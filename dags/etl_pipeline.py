"""RetailPulse ETL DAG: raw -> features -> processed -> train. Manual trigger only."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = Path(os.environ.get("RETAILPULSE_DATA_DIR", "/opt/airflow/data"))
RAW_PATH = DATA_DIR / "raw" / "sales.csv"
STAGING_PATH = DATA_DIR / "processed" / "features.staging.parquet"
FINAL_PATH = DATA_DIR / "processed" / "features.parquet"


def generate_or_load_raw_data() -> str:
    from src.data_generation import generate_or_load

    return generate_or_load(RAW_PATH)


def build_features_task() -> str:
    import pandas as pd

    from src.data_generation import validate_raw
    from src.features import build_features

    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    validate_raw(df)
    feats = build_features(df)
    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(STAGING_PATH, index=False)
    return str(STAGING_PATH)


def save_processed() -> str:
    import pandas as pd

    from src.features import FEATURE_COLUMNS, KEYS

    feats = pd.read_parquet(STAGING_PATH)
    if feats[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("NaNs in feature table")
    if feats.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate keys in feature table")
    os.replace(STAGING_PATH, FINAL_PATH)
    return str(FINAL_PATH)


def train_models() -> dict:
    from src.train import run_training

    results = run_training(str(FINAL_PATH))
    return {k: {m: v for m, v in r.items() if m.startswith("test_") and "cat" not in m} for k, r in results.items()}


def fetch_raw_from_s3_task():
    """Phase 4: pull raw data from the S3 bucket (LocalStack) into data/raw/ when available.

    RETAILPULSE_RAW_SOURCE: auto (default; falls back with a warning) | s3 (strict) | local.
    The existing generate_or_load_raw_data task then just validates the downloaded file.
    """
    from src.storage import fetch_raw_from_s3

    return fetch_raw_from_s3(RAW_PATH)


def evaluate_and_gate() -> dict:
    """Gate every not-yet-gated candidate run (XGBoost from this DAG run, plus any LSTM run
    trained manually on the host since the last execution), oldest first.

    A rejection is a valid outcome (task succeeds, decision is logged). Missing metrics
    raise GateError, which fails this task (the affected run stays ungated).
    """
    from src.gate import run_gate_pending

    return [d.to_dict() for d in run_gate_pending()]


with DAG(
    dag_id="etl_pipeline",
    description="Raw sales -> features -> train baseline + XGBoost -> MLflow",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # manual trigger from the Airflow UI
    catchup=False,
    tags=["retailpulse"],
) as dag:
    t_raw = PythonOperator(task_id="generate_or_load_raw_data", python_callable=generate_or_load_raw_data)
    t_feat = PythonOperator(task_id="build_features", python_callable=build_features_task)
    t_save = PythonOperator(task_id="save_processed", python_callable=save_processed)
    t_train = PythonOperator(task_id="train_models", python_callable=train_models)

    t_gate = PythonOperator(task_id="evaluate_and_gate", python_callable=evaluate_and_gate)

    t_fetch = PythonOperator(task_id="fetch_raw_from_s3", python_callable=fetch_raw_from_s3_task)

    t_fetch >> t_raw >> t_feat >> t_save >> t_train >> t_gate
