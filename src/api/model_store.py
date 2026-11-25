"""Loads whichever model currently holds the `production` alias on retailpulse-champion.

Reuses the MLflow registry logic from src/gate.py (CHAMPION_MODEL_NAME, get_production).
The alias is re-checked at most every `ttl` seconds; the model is reloaded only when the
aliased version changes, so a gate promotion is picked up without restarting the service.

Only XGBoost champions are served: this image has no torch and an LSTM needs a 28-day window.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import asdict, dataclass

import pandas as pd

from src import gate
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

CATEGORICAL = ["item_id", "store_id", "cat_id"]  # must match src/train.py CATEGORICAL
SERVABLE_TYPES = ("xgboost",)


class ModelUnavailable(RuntimeError):
    """Registry/artifact/vocabulary could not be reached or nothing is Production (HTTP 503)."""


class UnsupportedModelType(RuntimeError):
    """Production model family cannot be served by this image (HTTP 501)."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    run_id: str | None
    model_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_vocabulary(client=None, local_path: str | None = None) -> dict[str, list[str]]:
    """Category vocabularies = sorted unique ids in the raw data, i.e. what
    `astype("category")` produced in src/train.py. Raw data from S3 (LocalStack), else local file."""
    import os

    from src import storage

    local_path = local_path or os.environ.get("RETAILPULSE_RAW_LOCAL_PATH", "data/raw/sales.csv")
    usecols = lambda c: c in CATEGORICAL  # noqa: E731
    df = None
    errors = []
    try:
        client = client or storage.get_s3_client()
        body = client.get_object(Bucket=storage.raw_bucket(), Key=storage.DEFAULT_KEY)["Body"].read()
        df = pd.read_csv(io.BytesIO(body), usecols=usecols)
    except Exception as e:  # noqa: BLE001
        errors.append(f"s3: {e}")
    if df is None and local_path:
        try:
            df = pd.read_csv(local_path, usecols=usecols)
        except Exception as e:  # noqa: BLE001
            errors.append(f"local: {e}")
    if df is None:
        raise ModelUnavailable("Cannot build category vocabulary from raw data (" + "; ".join(errors) + ")")
    return {c: sorted(df[c].dropna().astype(str).unique()) for c in CATEGORICAL if c in df.columns}


def default_model_loader(info: ModelInfo):
    import mlflow.xgboost

    model = mlflow.xgboost.load_model(f"models:/{info.name}/{info.version}")
    if hasattr(model, "set_params"):
        model.set_params(enable_categorical=True)  # not persisted with the model
    return model


class ChampionModelStore:
    def __init__(self, client=None, ttl: float = 30.0, clock=time.monotonic,
                 model_loader=default_model_loader, vocab_loader=load_vocabulary,
                 model_name: str = gate.CHAMPION_MODEL_NAME):
        self._client = client
        self.ttl, self._clock = ttl, clock
        self._model_loader, self._vocab_loader = model_loader, vocab_loader
        self.model_name = model_name
        self._lock = threading.Lock()
        self._model = self._vocab = self._info = None
        self._checked_at: float | None = None
        self.last_error: str | None = None

    def _mlflow(self):
        if self._client is None:
            from mlflow import MlflowClient

            self._client = MlflowClient()
        return self._client

    def get(self):
        """Return (model, info, vocab); raises ModelUnavailable / UnsupportedModelType."""
        with self._lock:
            now = self._clock()
            if self._model is not None and self._checked_at is not None and now - self._checked_at < self.ttl:
                return self._model, self._info, self._vocab
            try:
                mv = gate.get_production(self._mlflow(), self.model_name)
            except Exception as e:  # noqa: BLE001 - GateError, MlflowException, connection errors
                self.last_error = f"registry: {e}"
                raise ModelUnavailable(self.last_error) from e
            if mv is None:
                self.last_error = f"no '{gate.PRODUCTION_ALIAS}' alias on {self.model_name}"
                raise ModelUnavailable(self.last_error)
            info = ModelInfo(
                name=self.model_name, version=str(mv.version), run_id=getattr(mv, "run_id", None),
                model_type=(getattr(mv, "tags", None) or {}).get("source_model_type", "unknown"),
            )
            self._checked_at = now
            if info.model_type not in SERVABLE_TYPES:
                self._model, self._info = None, info  # remember what is Production, serve nothing
                self.last_error = f"production model type '{info.model_type}' is not servable"
                raise UnsupportedModelType(
                    f"Production model {info.name} v{info.version} is '{info.model_type}'; this service "
                    f"serves only {SERVABLE_TYPES} (no torch / needs a 28-day window)."
                )
            if self._info is not None and self._model is not None and self._info.version == info.version:
                return self._model, self._info, self._vocab  # alias unchanged
            try:
                vocab = self._vocab or self._vocab_loader()
                model = self._model_loader(info)
            except ModelUnavailable:
                raise
            except Exception as e:  # noqa: BLE001
                self.last_error = f"load v{info.version}: {e}"
                raise ModelUnavailable(self.last_error) from e
            logger.info("Loaded %s v%s (%s)", info.name, info.version, info.model_type)
            self._model, self._info, self._vocab, self.last_error = model, info, vocab, None
            return self._model, self._info, self._vocab

    def status(self) -> dict:
        return {
            "model_loaded": self._model is not None,
            "model": self._info.to_dict() if self._info else None,
            "last_error": self.last_error,
        }


def build_frame(instances: list[dict], vocab: dict[str, list[str]]) -> pd.DataFrame:
    """Feature frame in training column order; categoricals use the training vocabulary."""
    df = pd.DataFrame(instances)
    for c in CATEGORICAL:
        if c not in vocab:
            continue
        bad = sorted(set(df[c].astype(str)) - set(vocab[c]))
        if bad:
            raise ValueError(f"Unknown {c}: {bad[:5]} (known values come from the training data)")
        df[c] = pd.Categorical(df[c].astype(str), categories=vocab[c])
    cols = FEATURE_COLUMNS + [c for c in CATEGORICAL if c in vocab]
    return df[cols]
