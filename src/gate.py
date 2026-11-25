"""Retrain gate: promote a candidate model only if it beats the Production model.

RULE (both must hold, otherwise the candidate is rejected):
  1. WMAPE: candidate test_wmape must be at least WMAPE_MIN_RELATIVE_IMPROVEMENT
     (2%) *relatively* better (lower) than Production's:
         (prod_wmape - cand_wmape) / prod_wmape >= 0.02
  2. Bias:  |candidate test_bias| must not be worse than |Production test_bias|:
         abs(cand_bias) <= abs(prod_bias)
If nothing is Production yet, the candidate is promoted unconditionally and the
decision is flagged `first_promotion` (no comparison happened).

Metrics are READ from the MLflow runs logged by train.py / train_dl.py, never
recomputed. Missing or non-finite metrics raise GateError - they never default
to "promote".

Registry: MLflow 2.16.2 -> model *aliases* (the stages API is deprecated). The
single cross-family Production slot is the registered model CHAMPION_MODEL_NAME
with alias PRODUCTION_ALIAS; a promotion registers the candidate run's `model`
artifact as a new version there and moves the alias. The replaced version is
tagged status=archived. A rejection never touches the registry.

Deliberately imports only `mlflow` (no torch/xgboost) so it runs in the Airflow image.

PENDING MODE (used by the Airflow task): every processed run (promote OR reject) is tagged
gated=true. `run_gate_pending` gates every FINISHED xgboost/lstm run in the experiment that
is not yet tagged gated=true, oldest first, each against the Production model as it stands
after the previous one. Already-gated runs are never re-evaluated (so a decision cannot flip
because the registry moved on). Dry runs tag nothing.

CLI:  python -m src.gate --pending | --model-type xgboost|lstm | --run-id <id>   [--dry-run]
      python -m src.gate --mark-gated <run_id>      # backfill: mark a run evaluated elsewhere
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

# ---- gate configuration (named constants, not magic numbers) -----------------
WMAPE_MIN_RELATIVE_IMPROVEMENT = 0.02  # candidate must be >= 2% relatively better
BIAS_RULE = "abs(candidate_bias) <= abs(production_bias)"
FLOAT_TOL = 1e-12  # so "exactly 2%" is not lost to float rounding
CHAMPION_MODEL_NAME = "retailpulse-champion"
PRODUCTION_ALIAS = "production"
EXPERIMENT = "retailpulse-demand"  # keep in sync with src/train.py
REQUIRED_METRICS = ("test_wmape", "test_bias")
GATED_TAG = "gated"
GATEABLE_MODEL_TYPES = ("xgboost", "lstm")  # naive_seasonal has no model artifact to promote


class GateError(RuntimeError):
    """Gate could not make a trustworthy decision (missing run/metric/model)."""


@dataclass(frozen=True)
class Metrics:
    wmape: float
    bias: float


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str
    first_promotion: bool
    candidate: Metrics
    production: Metrics | None
    wmape_improvement: float | None  # relative; None when there was no comparison
    candidate_run_id: str | None = None
    production_run_id: str | None = None
    production_version: str | None = None

    def banner(self) -> str:
        p = self.production
        lines = [
            "=" * 72,
            f"GATE DECISION: {'PROMOTE' if self.promote else 'REJECT'}"
            + ("  (first promotion - no comparison)" if self.first_promotion else ""),
            f"  reason    : {self.reason}",
            f"  candidate : run {self.candidate_run_id}  test_wmape={self.candidate.wmape:.4f}  "
            f"test_bias={self.candidate.bias:+.4f}",
            "  production: "
            + (
                f"run {self.production_run_id} (v{self.production_version})  "
                f"test_wmape={p.wmape:.4f}  test_bias={p.bias:+.4f}"
                if p
                else "none"
            ),
            f"  rule      : WMAPE >= {WMAPE_MIN_RELATIVE_IMPROVEMENT:.0%} relatively better AND {BIAS_RULE}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- pure comparison logic ---------------------------------------------------
def compare_metrics(candidate: Metrics, production: Metrics | None, **ids) -> GateDecision:
    """Pure function: no MLflow, no I/O. `ids` fill the *_run_id/version fields."""
    if production is None:
        return GateDecision(
            promote=True,
            reason="First promotion: no Production model exists yet, so the candidate is promoted "
            "without comparison.",
            first_promotion=True,
            candidate=candidate,
            production=None,
            wmape_improvement=None,
            **ids,
        )
    if production.wmape <= 0:
        raise GateError(f"Production test_wmape is {production.wmape}; relative improvement undefined")

    improvement = (production.wmape - candidate.wmape) / production.wmape
    wmape_ok = improvement >= WMAPE_MIN_RELATIVE_IMPROVEMENT - FLOAT_TOL
    bias_ok = abs(candidate.bias) <= abs(production.bias) + FLOAT_TOL

    wmape_msg = (
        f"WMAPE {candidate.wmape:.4f} vs {production.wmape:.4f} is {improvement:+.2%} relative "
        f"({'>=' if wmape_ok else '<'} required {WMAPE_MIN_RELATIVE_IMPROVEMENT:.2%})"
    )
    bias_msg = (
        f"|bias| {abs(candidate.bias):.4f} vs {abs(production.bias):.4f} "
        f"({'not worse' if bias_ok else 'WORSE'})"
    )
    promote = wmape_ok and bias_ok
    reason = f"{'PROMOTE' if promote else 'REJECT'}: {wmape_msg}; {bias_msg}."
    return GateDecision(
        promote=promote,
        reason=reason,
        first_promotion=False,
        candidate=candidate,
        production=production,
        wmape_improvement=improvement,
        **ids,
    )


# ---- MLflow reads ------------------------------------------------------------
def fetch_metrics(client: MlflowClient, run_id: str) -> Metrics:
    try:
        run = client.get_run(run_id)
    except MlflowException as e:
        raise GateError(f"Cannot load run {run_id}: {e}") from e
    vals = {}
    for key in REQUIRED_METRICS:
        v = run.data.metrics.get(key)
        if v is None or not math.isfinite(v):
            raise GateError(f"Run {run_id} has no usable metric '{key}' (got {v!r}); refusing to decide.")
        vals[key] = float(v)
    return Metrics(wmape=vals["test_wmape"], bias=vals["test_bias"])


def get_production(client: MlflowClient, model_name: str):
    """Return the Production ModelVersion, or None only if the model/alias does not exist."""
    # Read the model's alias map instead of calling get_model_version_by_alias: a missing
    # alias surfaces as INVALID_PARAMETER_VALUE on the REST store (not RESOURCE_DOES_NOT_EXIST),
    # and matching on error text could mistake a real failure for "first promotion".
    try:
        rm = client.get_registered_model(model_name)
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None  # model never registered -> nothing is Production
        raise GateError(f"Could not read registered model {model_name}: {e}") from e
    version = (rm.aliases or {}).get(PRODUCTION_ALIAS)
    if version is None:
        return None
    try:
        return client.get_model_version(model_name, str(version))
    except MlflowException as e:
        raise GateError(f"Alias '{PRODUCTION_ALIAS}' points to v{version} of {model_name} but it cannot be read: {e}") from e


def resolve_candidate(client: MlflowClient, model_type: str) -> str:
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise GateError(f"Experiment '{EXPERIMENT}' not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.model_type = '{model_type}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise GateError(f"No FINISHED run with model_type='{model_type}' in '{EXPERIMENT}'")
    return runs[0].info.run_id


def find_ungated_runs(client: MlflowClient) -> list:
    """FINISHED xgboost/lstm runs in the experiment without gated=true, oldest first."""
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise GateError(f"Experiment '{EXPERIMENT}' not found")
    # The REST filter grammar has no tag IN/OR and unclear tag-absence semantics, so the server
    # only filters on status; model_type and gated are checked below.
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
        max_results=1000,
    )
    # Filter + sort again client-side: don't rely on server-side tag-absence semantics or ordering.
    pending = [
        r for r in runs
        if r.data.tags.get("model_type") in GATEABLE_MODEL_TYPES
        and r.data.tags.get(GATED_TAG) != "true"
    ]
    return sorted(pending, key=lambda r: r.info.start_time)


def mark_gated(client: MlflowClient, run_id: str, decision: GateDecision | None = None,
               note: str | None = None) -> None:
    client.set_tag(run_id, GATED_TAG, "true")
    client.set_tag(run_id, "gated_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if decision is not None:
        client.set_tag(run_id, "gate_decision", "promote" if decision.promote else "reject")
        client.set_tag(run_id, "gate_reason", decision.reason[:250])
    if note:
        client.set_tag(run_id, "gate_note", note[:250])


def evaluate_gate(
    candidate_run_id: str,
    production_model_name: str = CHAMPION_MODEL_NAME,
    client: MlflowClient | None = None,
) -> GateDecision:
    """READ-ONLY: compare the candidate run to the current Production model."""
    client = client or MlflowClient()
    cand = fetch_metrics(client, candidate_run_id)
    prod_version = get_production(client, production_model_name)
    if prod_version is None:
        return compare_metrics(cand, None, candidate_run_id=candidate_run_id)
    if not prod_version.run_id:
        raise GateError(f"Production version v{prod_version.version} has no source run; cannot read its metrics")
    prod = fetch_metrics(client, prod_version.run_id)
    return compare_metrics(
        cand, prod,
        candidate_run_id=candidate_run_id,
        production_run_id=prod_version.run_id,
        production_version=str(prod_version.version),
    )


# ---- registry writes (promote only) -----------------------------------------
def apply_decision(decision: GateDecision, model_name: str = CHAMPION_MODEL_NAME,
                   client: MlflowClient | None = None):
    """Promote -> new champion version + alias moved. Reject -> touch nothing (returns None)."""
    if not decision.promote:
        return None
    client = client or MlflowClient()
    run_id = decision.candidate_run_id
    run = client.get_run(run_id)
    try:
        client.create_registered_model(model_name)
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    mv = client.create_model_version(
        name=model_name, source=f"runs:/{run_id}/model", run_id=run_id,
        tags={
            "source_run_id": run_id,
            "source_model_type": run.data.tags.get("model_type", "unknown"),
            "test_wmape": f"{decision.candidate.wmape:.6f}",
            "test_bias": f"{decision.candidate.bias:.6f}",
            "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gate_reason": decision.reason[:250],
            "status": "production",
        },
    )
    for _ in range(60):  # wait until the version is READY
        mv = client.get_model_version(model_name, mv.version)
        if mv.status == "READY":
            break
        if mv.status == "FAILED_REGISTRATION":
            raise GateError(f"Registration of {model_name} v{mv.version} failed: {mv.status_message}")
        time.sleep(1)
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, mv.version)
    if decision.production_version:  # "archive" the previous Production version
        client.set_model_version_tag(model_name, decision.production_version, "status", "archived")
        client.set_model_version_tag(
            model_name, decision.production_version, "archived_at",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    return mv


def run_gate(
    candidate_run_id: str | None = None,
    model_type: str | None = None,
    dry_run: bool = False,
    production_model_name: str = CHAMPION_MODEL_NAME,
    client: MlflowClient | None = None,
) -> GateDecision:
    """Resolve candidate, decide, apply (unless dry_run), and log a clear banner."""
    if not candidate_run_id and not model_type:
        raise GateError("Provide candidate_run_id or model_type")
    client = client or MlflowClient()
    candidate_run_id = candidate_run_id or resolve_candidate(client, model_type)
    decision = evaluate_gate(candidate_run_id, production_model_name, client)
    logger.info("\n%s", decision.banner())
    if dry_run:
        logger.info("Dry run: registry NOT modified.")
    elif decision.promote:
        mv = apply_decision(decision, production_model_name, client)
        logger.info("Registry updated: %s v%s now has alias '%s'.", production_model_name, mv.version, PRODUCTION_ALIAS)
    else:
        logger.info("Registry left unchanged (candidate rejected).")
    if not dry_run:  # tag only after the registry action succeeded
        mark_gated(client, candidate_run_id, decision)
    return decision


def run_gate_pending(
    dry_run: bool = False,
    production_model_name: str = CHAMPION_MODEL_NAME,
    client: MlflowClient | None = None,
) -> list[GateDecision]:
    """Gate every ungated xgboost/lstm run, oldest first. One failing run does not block the
    others (it stays ungated and is retried next time), but the call raises at the end."""
    client = client or MlflowClient()
    pending = find_ungated_runs(client)
    if not pending:
        logger.info("No ungated candidate runs in '%s'; nothing to gate.", EXPERIMENT)
        return []
    logger.info(
        "%d ungated candidate run(s), oldest first: %s",
        len(pending),
        ", ".join(f"{r.data.tags.get('model_type')}:{r.info.run_id[:8]}" for r in pending),
    )
    decisions, errors = [], []
    for r in pending:
        try:
            decisions.append(run_gate(r.info.run_id, None, dry_run, production_model_name, client))
        except (GateError, MlflowException) as e:
            logger.error("Could not gate run %s: %s", r.info.run_id, e)
            errors.append(f"{r.info.run_id[:8]}: {e}")
    if errors:
        raise GateError(f"{len(errors)} run(s) could not be gated (left ungated): " + " | ".join(errors))
    return decisions


def main(argv=None):
    p = argparse.ArgumentParser(description="RetailPulse retrain gate")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pending", action="store_true", help="gate every ungated xgboost/lstm run, oldest first")
    g.add_argument("--run-id", help="candidate MLflow run id")
    g.add_argument("--model-type", choices=list(GATEABLE_MODEL_TYPES), help="use newest finished run of this type")
    g.add_argument("--mark-gated", metavar="RUN_ID", help="tag a run gated=true without evaluating it (backfill)")
    p.add_argument("--dry-run", action="store_true", help="decide but do not modify the registry or tags")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if os.environ.get("MLFLOW_TRACKING_URI") is None:
        logger.warning("MLFLOW_TRACKING_URI not set; using MLflow default (local ./mlruns)")
    if args.mark_gated:
        mark_gated(MlflowClient(), args.mark_gated, note="marked gated manually (evaluated outside the pending flow)")
        logger.info("Run %s tagged gated=true.", args.mark_gated)
        return None
    if args.pending:
        return run_gate_pending(dry_run=args.dry_run)
    return run_gate(args.run_id, args.model_type, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
