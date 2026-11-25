"""RetailPulse inference API.

    uvicorn src.api.app:app --port 8000          # local/dev (the container reads $PORT, default 8000)
    PORT=8080 ...                                # SageMaker requires 8080

GET  /health       liveness + which model is loaded (does not force a model load); local/compose use
POST /predict      {"instances": [ {item_id, store_id, cat_id, <13 numeric features>} ]}
                   -> predictions from the model aliased `production` on retailpulse-champion

SageMaker hosting contract (same prediction logic as /predict):
GET  /ping         200 only if the production model is loaded and servable, else 503
POST /invocations  application/json body: {"instances": [...]} or a bare list [...]
"""
from __future__ import annotations

import json

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from src.api.model_store import (
    ChampionModelStore,
    ModelUnavailable,
    UnsupportedModelType,
    build_frame,
)

MAX_INSTANCES = 1000
JSON_CONTENT_TYPES = ("application/json", "text/json")


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
    app = FastAPI(title="RetailPulse inference API", version="0.4.1")
    app.state.store = store if store is not None else ChampionModelStore()

    def run_prediction(req: PredictRequest) -> dict:
        """The single prediction code path behind /predict and /invocations."""
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

    @app.get("/health")
    def health():
        return {"status": "ok", **app.state.store.status()}

    @app.post("/predict")
    def predict(req: PredictRequest):
        return run_prediction(req)

    @app.get("/ping")
    def ping():
        """SageMaker health check. Stricter than /health: it loads the model if needed and only
        returns 200 when the production model is loaded and servable (non-200 = unhealthy)."""
        try:
            app.state.store.get()
        except (ModelUnavailable, UnsupportedModelType) as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return Response(status_code=200)

    @app.post("/invocations")
    async def invocations(request: Request):
        """SageMaker inference. The body is passed through as raw bytes by InvokeEndpoint, so we
        parse it ourselves: JSON `{"instances": [...]}` (as /predict) or a bare list of instances."""
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype and ctype not in JSON_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported Content-Type {ctype!r}; send application/json")
        try:
            payload = json.loads(await request.body())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Body is not valid JSON: {e}") from e
        if isinstance(payload, list):
            payload = {"instances": payload}
        try:
            req = PredictRequest.model_validate(payload)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=json.loads(e.json())) from e
        return await run_in_threadpool(run_prediction, req)

    return app


app = create_app()
