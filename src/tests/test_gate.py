import math
import os
import urllib.request

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from src import gate
from src.gate import (
    WMAPE_MIN_RELATIVE_IMPROVEMENT,
    GateError,
    Metrics,
    apply_decision,
    compare_metrics,
    fetch_metrics,
)

PROD = Metrics(wmape=0.50, bias=-0.10)


def test_threshold_constant_is_two_percent():
    assert WMAPE_MIN_RELATIVE_IMPROVEMENT == 0.02


def test_clear_promote():
    d = compare_metrics(Metrics(0.40, -0.05), PROD)  # 20% better, smaller |bias|
    assert d.promote and not d.first_promotion
    assert d.wmape_improvement == pytest.approx(0.20)
    assert d.reason.startswith("PROMOTE")


def test_clear_reject():
    d = compare_metrics(Metrics(0.499, -0.05), PROD)  # 0.2% better only
    assert not d.promote
    assert "<" in d.reason and "required 2.00%" in d.reason


def test_borderline_exactly_two_percent_promotes():
    d = compare_metrics(Metrics(0.49, -0.10), PROD)  # exactly 2.0% better, equal |bias|
    assert d.wmape_improvement == pytest.approx(0.02)
    assert d.promote


def test_just_under_two_percent_rejects():
    assert not compare_metrics(Metrics(0.4901, -0.10), PROD).promote


def test_worse_bias_blocks_otherwise_passing_wmape():
    d = compare_metrics(Metrics(0.30, -0.20), PROD)  # 40% better WMAPE but |bias| 0.20 > 0.10
    assert not d.promote
    assert "WORSE" in d.reason


def test_bias_sign_does_not_matter_only_magnitude():
    assert compare_metrics(Metrics(0.40, +0.10), PROD).promote  # |+0.10| == |-0.10|


def test_no_production_is_unconditional_first_promotion():
    d = compare_metrics(Metrics(0.99, +9.0), None)
    assert d.promote and d.first_promotion and d.production is None
    assert d.wmape_improvement is None
    assert "First promotion" in d.reason


def test_real_phase_numbers_reject_lstm():
    d = compare_metrics(Metrics(0.3354, -0.0430), Metrics(0.3384, -0.0197))
    assert not d.promote
    assert d.wmape_improvement == pytest.approx(0.0089, abs=5e-4)
    assert "<" in d.reason and "WORSE" in d.reason  # fails both conditions


# ---- missing metrics must fail loudly, never default to promote ---------------
class FakeRun:
    def __init__(self, metrics):
        self.data = type("D", (), {"metrics": metrics, "tags": {}})()


class StubClient:
    def __init__(self, metrics=None, missing_run=False):
        self.metrics, self.missing_run, self.calls = metrics or {}, missing_run, []

    def get_run(self, run_id):
        if self.missing_run:
            raise MlflowException("no such run")
        return FakeRun(self.metrics)

    def __getattr__(self, name):  # any registry mutator is recorded
        def record(*a, **k):
            self.calls.append(name)
        return record


@pytest.mark.parametrize("metrics", [
    {}, {"test_wmape": 0.3}, {"test_bias": 0.1},
    {"test_wmape": float("nan"), "test_bias": 0.1}, {"test_wmape": 0.3, "test_bias": math.inf},
])
def test_missing_or_nonfinite_metrics_raise(metrics):
    with pytest.raises(GateError):
        fetch_metrics(StubClient(metrics), "abc")


def test_missing_run_raises():
    with pytest.raises(GateError):
        fetch_metrics(StubClient(missing_run=True), "abc")


def test_evaluate_gate_fails_when_production_metrics_missing(monkeypatch):
    class V:
        run_id, version = "prodrun", 3

    class C(StubClient):
        def get_run(self, run_id):
            return FakeRun({"test_wmape": 0.3, "test_bias": 0.1} if run_id == "cand" else {})

    monkeypatch.setattr(gate, "get_production", lambda client, name: V())
    with pytest.raises(GateError):
        gate.evaluate_gate("cand", client=C())


class RegClient:
    def __init__(self, model=None, error=None):
        self.model, self.error = model, error

    def get_registered_model(self, name):
        if self.error:
            raise self.error
        return self.model

    def get_model_version(self, name, version):
        return f"v{version}"


def _rm(aliases):
    return type("RM", (), {"aliases": aliases})()


def test_get_production_cases():
    missing = MlflowException("nope", error_code=RESOURCE_DOES_NOT_EXIST)
    assert gate.get_production(RegClient(error=missing), "m") is None  # model not registered
    assert gate.get_production(RegClient(model=_rm({})), "m") is None  # no alias yet
    assert gate.get_production(RegClient(model=_rm({"staging": "1"})), "m") is None
    assert gate.get_production(RegClient(model=_rm({"production": "3"})), "m") == "v3"


def test_get_production_other_registry_errors_are_not_treated_as_first_promotion():
    boom = MlflowException("server exploded", error_code=INTERNAL_ERROR)
    with pytest.raises(GateError):
        gate.get_production(RegClient(error=boom), "m")


def test_reject_never_touches_registry():
    d = compare_metrics(Metrics(0.499, -0.05), PROD, candidate_run_id="cand")
    client = StubClient()
    assert apply_decision(d, client=client) is None
    assert client.calls == []


# ---- integration: real Phase 1/2 runs (skipped without a reachable MLflow) ----
def _mlflow_up():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not uri.startswith("http"):
        return None
    try:
        urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=2)
        return uri
    except Exception:
        return None


@pytest.mark.skipif(_mlflow_up() is None, reason="no reachable MLflow server")
def test_real_runs_lstm_is_rejected_against_xgboost():
    from mlflow import MlflowClient, set_tracking_uri

    set_tracking_uri(_mlflow_up())
    client = MlflowClient()
    try:
        xgb = gate.resolve_candidate(client, "xgboost")
        lstm = gate.resolve_candidate(client, "lstm")
    except GateError as e:
        pytest.skip(f"Phase 1/2 runs not present: {e}")
    d = compare_metrics(gate.fetch_metrics(client, lstm), gate.fetch_metrics(client, xgb))
    assert d.promote is False
    assert d.wmape_improvement < WMAPE_MIN_RELATIVE_IMPROVEMENT  # ~0.9% < 2%
    assert "WORSE" in d.reason  # |bias| 0.043 vs 0.020
