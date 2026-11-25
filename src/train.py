"""Train naive seasonal baseline + XGBoost; log both to one MLflow experiment."""
from __future__ import annotations

import os

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from src.evaluate import bias, time_split, wmape, wmape_by_group
from src.features import FEATURE_COLUMNS

EXPERIMENT = "retailpulse-demand"
REGISTERED_MODEL = "retailpulse-xgboost"
CATEGORICAL = ["item_id", "store_id", "cat_id"]

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "early_stopping_rounds": 30,
    "random_state": 42,
}


def _log_split_metrics(df: pd.DataFrame, pred, prefix: str) -> dict[str, float]:
    metrics = {f"{prefix}_wmape": wmape(df["sales"], pred), f"{prefix}_bias": bias(df["sales"], pred)}
    if "cat_id" in df.columns:  # secondary breakdown, logging only
        tmp = df.assign(_pred=pred)
        for cat, v in wmape_by_group(tmp, "cat_id", "sales", "_pred").items():
            metrics[f"{prefix}_wmape_cat_{cat}"] = v
    mlflow.log_metrics(metrics)
    return metrics


def run_training(features_path: str, tracking_uri: str | None = None) -> dict:
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)
    cat_cols = [c for c in CATEGORICAL if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    x_cols = FEATURE_COLUMNS + cat_cols

    tr, va, te = time_split(df["date"], 0.8, 0.1)
    train, val, test = df[tr], df[va], df[te]
    split_info = {
        "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "train_end": str(train["date"].max().date()),
        "val_end": str(val["date"].max().date()),
        "test_end": str(test["date"].max().date()),
    }
    results = {}

    # 1) Naive seasonal baseline: same weekday last week (== lag_7)
    with mlflow.start_run(run_name="naive_seasonal"):
        mlflow.set_tag("model_type", "naive_seasonal")
        mlflow.log_param("method", "lag_7 (same day last week)")
        mlflow.log_params(split_info)
        m = _log_split_metrics(val, val["lag_7"], "val")
        m.update(_log_split_metrics(test, test["lag_7"], "test"))
        results["naive_seasonal"] = m

    # 2) XGBoost on engineered features
    with mlflow.start_run(run_name="xgboost"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.log_params({k: v for k, v in XGB_PARAMS.items()})
        mlflow.log_param("features", ",".join(x_cols))
        mlflow.log_params(split_info)
        model = xgb.XGBRegressor(enable_categorical=True, **XGB_PARAMS)
        model.fit(train[x_cols], train["sales"], eval_set=[(val[x_cols], val["sales"])], verbose=False)
        mlflow.log_metric("best_iteration", model.best_iteration)
        m = _log_split_metrics(val, model.predict(val[x_cols]).clip(0), "val")
        m.update(_log_split_metrics(test, model.predict(test[x_cols]).clip(0), "test"))
        results["xgboost"] = m
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=REGISTERED_MODEL)

    return results


if __name__ == "__main__":
    import sys

    out = run_training(sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.parquet")
    for name, m in out.items():
        print(name, {k: round(v, 4) for k, v in m.items() if k.startswith("test_") and "cat" not in k})
