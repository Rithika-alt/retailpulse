"""Pending-run gating (gated=true tags), tested with an in-memory fake MLflow client."""
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST

from src import gate
from src.gate import CHAMPION_MODEL_NAME, GateError


def make_run(run_id, model_type, start, wmape, bias, gated=None, status="FINISHED", metrics=None):
    tags = {"model_type": model_type}
    if gated is not None:
        tags["gated"] = gated
    m = {"test_wmape": wmape, "test_bias": bias} if metrics is None else metrics
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start, status=status),
        data=SimpleNamespace(metrics=m, tags=tags),
    )


class FakeMlflow:
    """Just enough of MlflowClient for gate.run_gate_pending; records registry writes."""

    def __init__(self, runs):
        self.runs = {r.info.run_id: r for r in runs}
        self.models = {}  # name -> {"aliases": {}, "versions": {}}
        self.writes = []

    # experiments / runs
    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def search_runs(self, ids, filter_string=None, order_by=None, max_results=None):
        # deliberately unsorted (newest first) and unfiltered: gate must not depend on the server
        return sorted(self.runs.values(), key=lambda r: -r.info.start_time)

    def get_run(self, run_id):
        return self.runs[run_id]

    def set_tag(self, run_id, key, value):
        self.runs[run_id].data.tags[key] = value

    # registry
    def get_registered_model(self, name):
        if name not in self.models:
            raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace(aliases=dict(self.models[name]["aliases"]))

    def create_registered_model(self, name):
        if name in self.models:
            raise MlflowException("exists", error_code=RESOURCE_ALREADY_EXISTS)
        self.models[name] = {"aliases": {}, "versions": {}}
        self.writes.append(("create_registered_model", name))

    def create_model_version(self, name, source, run_id, tags=None):
        versions = self.models[name]["versions"]
        v = str(len(versions) + 1)
        versions[v] = SimpleNamespace(version=v, run_id=run_id, status="READY", tags=dict(tags or {}))
        self.writes.append(("create_model_version", name, run_id))
        return versions[v]

    def get_model_version(self, name, version):
        return self.models[name]["versions"][str(version)]

    def set_registered_model_alias(self, name, alias, version):
        self.models[name]["aliases"][alias] = str(version)
        self.writes.append(("set_alias", name, alias, str(version)))

    def set_model_version_tag(self, name, version, key, value):
        self.models[name]["versions"][str(version)].tags[key] = value
        self.writes.append(("version_tag", name, str(version), key))

    def production_run(self):
        m = self.models.get(CHAMPION_MODEL_NAME)
        if not m or "production" not in m["aliases"]:
            return None
        return m["versions"][m["aliases"]["production"]].run_id


def test_two_ungated_runs_are_both_gated_oldest_first():
    # xgboost older (0.40), lstm newer and clearly better (0.30, smaller |bias|)
    client = FakeMlflow([
        make_run("lstm1", "lstm", 200, 0.30, -0.01),
        make_run("xgb1", "xgboost", 100, 0.40, -0.05),
    ])
    decisions = gate.run_gate_pending(client=client)

    assert [d.candidate_run_id for d in decisions] == ["xgb1", "lstm1"]  # run order, not input order
    first, second = decisions
    assert first.promote and first.first_promotion  # nothing was Production yet
    assert second.promote and not second.first_promotion
    assert second.production_run_id == "xgb1"  # compared with Production as left by the first
    assert client.production_run() == "lstm1"
    for rid in ("xgb1", "lstm1"):
        assert client.runs[rid].data.tags["gated"] == "true"
        assert client.runs[rid].data.tags["gate_decision"] == "promote"


def test_order_matters_reversed_order_would_give_different_outcome():
    """Guard: if the newer run were gated first it would be the 'first promotion'."""
    client = FakeMlflow([make_run("lstm1", "lstm", 200, 0.30, -0.01), make_run("xgb1", "xgboost", 100, 0.40, -0.05)])
    d = gate.run_gate_pending(client=client)
    assert d[0].candidate_run_id == "xgb1" and d[0].first_promotion


def test_already_gated_run_is_skipped_and_never_reevaluated():
    # Production = xgb1. lstm1 was rejected earlier (gated=true); it would PASS today, but must not flip.
    client = FakeMlflow([
        make_run("xgb1", "xgboost", 100, 0.40, -0.05, gated="true"),
        make_run("lstm1", "lstm", 200, 0.30, -0.01, gated="true"),
    ])
    client.create_registered_model(CHAMPION_MODEL_NAME)
    client.create_model_version(CHAMPION_MODEL_NAME, "src", "xgb1")
    client.set_registered_model_alias(CHAMPION_MODEL_NAME, "production", "1")
    writes_before = list(client.writes)

    assert gate.run_gate_pending(client=client) == []
    assert client.writes == writes_before  # registry untouched
    assert client.production_run() == "xgb1"


def test_only_ungated_run_among_gated_ones_is_processed():
    client = FakeMlflow([
        make_run("xgb1", "xgboost", 100, 0.40, -0.05, gated="true"),
        make_run("xgb2", "xgboost", 300, 0.40, -0.05),
    ])
    client.create_registered_model(CHAMPION_MODEL_NAME)
    client.create_model_version(CHAMPION_MODEL_NAME, "src", "xgb1")
    client.set_registered_model_alias(CHAMPION_MODEL_NAME, "production", "1")
    decisions = gate.run_gate_pending(client=client)
    assert [d.candidate_run_id for d in decisions] == ["xgb2"]
    assert not decisions[0].promote  # identical retrain: 0% improvement
    assert client.runs["xgb2"].data.tags["gate_decision"] == "reject"
    assert client.production_run() == "xgb1"


def test_second_invocation_is_a_noop():
    client = FakeMlflow([make_run("xgb1", "xgboost", 100, 0.40, -0.05)])
    assert len(gate.run_gate_pending(client=client)) == 1
    writes = list(client.writes)
    assert gate.run_gate_pending(client=client) == []
    assert client.writes == writes


def test_dry_run_tags_nothing_and_writes_nothing():
    client = FakeMlflow([make_run("xgb1", "xgboost", 100, 0.40, -0.05)])
    decisions = gate.run_gate_pending(dry_run=True, client=client)
    assert len(decisions) == 1 and decisions[0].promote
    assert "gated" not in client.runs["xgb1"].data.tags
    assert client.writes == []


def test_naive_runs_and_unfinished_runs_are_not_candidates():
    client = FakeMlflow([
        make_run("naive1", "naive_seasonal", 50, 0.48, -0.05),
        make_run("xgb_running", "xgboost", 60, 0.40, -0.05, status="RUNNING"),
    ])
    # fake returns everything; the gate's own client-side type filter drops naive.
    pending = [r.info.run_id for r in gate.find_ungated_runs(client)]
    assert "naive1" not in pending


def test_run_with_missing_metrics_fails_loudly_but_does_not_block_others():
    client = FakeMlflow([
        make_run("bad", "lstm", 100, None, None, metrics={}),
        make_run("good", "xgboost", 200, 0.40, -0.05),
    ])
    with pytest.raises(GateError, match="bad"):
        gate.run_gate_pending(client=client)
    assert client.runs["good"].data.tags.get("gated") == "true"  # still processed
    assert "gated" not in client.runs["bad"].data.tags  # retried next time, never silently promoted
    assert client.production_run() == "good"
