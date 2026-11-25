"""Inference API tests: MLflow, S3 and the real model are all faked (no servers needed)."""
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import model_store as ms  # noqa: E402
from src.api.app import create_app  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402

VOCAB = {"item_id": ["ITEM_000", "ITEM_001"], "store_id": ["CA_1", "TX_1"], "cat_id": ["FOODS", "HOBBIES"]}
INFO = ms.ModelInfo(name="retailpulse-champion", version="1", run_id="abc123", model_type="xgboost")


def instance(**over):
    base = dict(item_id="ITEM_001", store_id="CA_1", cat_id="FOODS", lag_7=5.0, lag_14=4.0, lag_28=6.0,
                roll_mean_7=5.0, roll_std_7=1.0, roll_mean_28=5.5, roll_std_28=1.2, day_of_week=2,
                is_weekend=0, is_holiday=0, sell_price=3.5, price_change_flag=0, promo_flag=0)
    base.update(over)
    return base


class FakeModel:
    def __init__(self, value=7.25):
        self.value, self.seen = value, None

    def predict(self, frame):
        self.seen = frame
        return np.full(len(frame), self.value)


class FakeStore:
    def __init__(self, model=None, info=INFO, error=None):
        self.model, self.info, self.error = model or FakeModel(), info, error

    def get(self):
        if self.error:
            raise self.error
        return self.model, self.info, VOCAB

    def status(self):
        return {"model_loaded": True, "model": self.info.to_dict(), "last_error": None}


def client(store):
    return TestClient(create_app(store))


def test_categorical_list_matches_training():
    from src.train import CATEGORICAL

    assert ms.CATEGORICAL == CATEGORICAL


def test_health():
    r = client(FakeStore()).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model"]["version"] == "1"


def test_predict_happy_path_returns_prediction_and_model_info():
    store = FakeStore()
    r = client(store).post("/predict", json={"instances": [instance(), instance(item_id="ITEM_000")]})
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"] == [7.25, 7.25]
    assert body["model"] == {"name": "retailpulse-champion", "version": "1", "run_id": "abc123", "model_type": "xgboost"}
    frame = store.model.seen  # training column order, categoricals encoded with training vocabulary
    assert list(frame.columns) == FEATURE_COLUMNS + ["item_id", "store_id", "cat_id"]
    assert str(frame["item_id"].dtype) == "category" and list(frame["item_id"].cat.categories) == VOCAB["item_id"]


def test_negative_model_output_is_clipped_to_zero():
    r = client(FakeStore(model=FakeModel(-3.0))).post("/predict", json={"instances": [instance()]})
    assert r.json()["predictions"] == [0.0]


def test_unknown_category_is_422_not_silently_encoded():
    r = client(FakeStore()).post("/predict", json={"instances": [instance(item_id="NOPE")]})
    assert r.status_code == 422 and "item_id" in r.json()["detail"]


@pytest.mark.parametrize("bad", [{"lag_7": "abc"}, {"day_of_week": 9}, {"is_weekend": 2}])
def test_invalid_payload_is_422(bad):
    assert client(FakeStore()).post("/predict", json={"instances": [instance(**bad)]}).status_code == 422


def test_missing_feature_and_empty_batch_are_422():
    inst = instance()
    del inst["lag_7"]
    c = client(FakeStore())
    assert c.post("/predict", json={"instances": [inst]}).status_code == 422
    assert c.post("/predict", json={"instances": []}).status_code == 422


def test_registry_unavailable_is_503():
    c = client(FakeStore(error=ms.ModelUnavailable("registry: connection refused")))
    r = c.post("/predict", json={"instances": [instance()]})
    assert r.status_code == 503 and "connection refused" in r.json()["detail"]


def test_lstm_champion_is_501_with_clear_message():
    c = client(FakeStore(error=ms.UnsupportedModelType("Production model is 'lstm'")))
    r = c.post("/predict", json={"instances": [instance()]})
    assert r.status_code == 501 and "lstm" in r.json()["detail"]


# ---- ChampionModelStore: alias handling, TTL, reload on promotion --------------
class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def mv(version, model_type="xgboost"):
    return type("MV", (), {"version": str(version), "run_id": f"run{version}",
                           "tags": {"source_model_type": model_type}})()


def make_store(monkeypatch, versions, clock):
    """versions: a mutable one-item list holding the current production ModelVersion (or None)."""
    from src import gate

    monkeypatch.setattr(gate, "get_production", lambda client, name: versions[0])
    loads = []

    def loader(info):
        loads.append(info.version)
        return FakeModel(float(info.version))

    store = ms.ChampionModelStore(client=object(), ttl=30, clock=clock, model_loader=loader,
                                  vocab_loader=lambda: VOCAB)
    return store, loads


def test_store_loads_once_then_reloads_only_when_alias_moves(monkeypatch):
    clock, versions = Clock(), [mv(1)]
    store, loads = make_store(monkeypatch, versions, clock)

    assert store.get()[1].version == "1" and loads == ["1"]
    clock.t = 10  # inside TTL: no registry check at all, even if the alias moved
    versions[0] = mv(2)
    assert store.get()[1].version == "1" and loads == ["1"]
    clock.t = 31  # TTL expired: sees the promotion and reloads
    assert store.get()[1].version == "2" and loads == ["1", "2"]
    clock.t = 70  # TTL expired, alias unchanged: no reload
    assert store.get()[1].version == "2" and loads == ["1", "2"]


def test_store_no_production_alias_is_unavailable(monkeypatch):
    store, _ = make_store(monkeypatch, [None], Clock())
    with pytest.raises(ms.ModelUnavailable):
        store.get()


def test_store_lstm_champion_not_servable(monkeypatch):
    store, loads = make_store(monkeypatch, [mv(3, "lstm")], Clock())
    with pytest.raises(ms.UnsupportedModelType):
        store.get()
    assert loads == [] and store.status()["model"]["model_type"] == "lstm"


def test_store_registry_error_is_unavailable(monkeypatch):
    from src import gate

    def boom(client, name):
        raise gate.GateError("down")

    monkeypatch.setattr(gate, "get_production", boom)
    store = ms.ChampionModelStore(client=object(), model_loader=lambda i: None, vocab_loader=lambda: VOCAB)
    with pytest.raises(ms.ModelUnavailable):
        store.get()
