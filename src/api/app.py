"""RetailPulse inference API.

    uvicorn src.api.app:app --port 8000

GET  /health   liveness + which model is loaded (does not force a model load)
POST /predict  {"instances": [ {item_id, store_id, cat_id, <13 numeric features>} ]}
               -> predictions from the model aliased `production` on retailpulse-champion
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.model_store import (
    ChampionModelStore,
    ModelUnavailable,
    UnsupportedModelType,
    build_frame,
)

MAX_INSTANCES = 1000


class Instance(BaseModel):
    item_id: str
    store_id: str
    cat_id: str
    lag_7: float
    lag_14: float
    lag_28: float
    roll_mean_7: float
    roll_std_7: float
    roll_mean_28: float
    roll_std_28: float
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    is_holiday: int = Field(ge=0, le=1)
    sell_price: float = Field(ge=0)
    price_change_flag: int = Field(ge=0, le=1)
    promo_flag: int = Field(ge=0, le=1)


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=MAX_INSTANCES)


def create_app(store=None) -> FastAPI:
    app = FastAPI(title="RetailPulse inference API", version="0.4.0")
    app.state.store = store if store is not None else ChampionModelStore()

    @app.get("/health")
    def health():
        return {"status": "ok", **app.state.store.status()}

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            model, info, vocab = app.state.store.get()
        except UnsupportedModelType as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ModelUnavailable as e:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {e}") from e
        try:
            frame = build_frame([i.model_dump() for i in req.instances], vocab)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        preds = np.clip(np.asarray(model.predict(frame), dtype=float), 0, None)  # like train.py
        return {"predictions": [float(p) for p in preds], "model": info.to_dict()}

    return app


app = create_app()
