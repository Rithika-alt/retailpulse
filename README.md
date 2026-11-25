# RetailPulse — Phase 1

Local pipeline: raw sales -> ETL -> leakage-safe feature table -> naive seasonal baseline + XGBoost -> WMAPE/bias in MLflow.
Scope: no cloud, no deep learning, no agents.

## Run it

```bash
# 1. Start the stack (Postgres, MLflow, Airflow webserver + scheduler)
docker compose up --build -d

# 2. Trigger the DAG: open http://localhost:8080 (admin / admin),
#    click the play button on `etl_pipeline` -> "Trigger DAG".
#    (CLI alternative: docker compose exec airflow-scheduler airflow dags trigger etl_pipeline)

# 3. Compare models: open http://localhost:5000 -> experiment `retailpulse-demand`
#    -> tick both runs (`naive_seasonal`, `xgboost`) -> Compare.
#    The XGBoost model is registered as `retailpulse-xgboost` (Models tab).

docker compose down        # stop (add -v to wipe Postgres + MLflow artifacts)
```

## Local dev / tests

```bash
py -3.11 -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -r requirements.txt
pytest                                              # includes the explicit leakage test
```

## Layout
- `src/data_generation.py` – synthetic M5-style data (50 items x 5 stores x 730 days). If `data/raw/sales.csv` already exists it is used instead (must be long format: `item_id, store_id, date, sell_price, sales`, optional `cat_id`). Wide-format M5 CSVs must be melted first.
- `src/features.py` – lags (7/14/28), rolling mean/std (7/28, computed on prior days only), calendar, price/promo flags.
- `src/evaluate.py` – WMAPE (global), bias (signed), per-category WMAPE, 80/10/10 time split.
- `src/train.py` – baseline (`lag_7`) + global XGBoost; params, metrics, model artifact logged to MLflow.
- `dags/etl_pipeline.py` – `generate_or_load_raw_data >> build_features >> save_processed >> train_models`, `schedule=None`.

## Notes
- Features assume contiguous daily rows per item-store (enforced; raises otherwise).
- Per-category WMAPE is logged only (`test_wmape_cat_*`); it drives no decision.

---

# Phase 2 — LSTM sequence model (Hugging Face Accelerate)

Third model family in the same MLflow experiment (`retailpulse-demand`, run `lstm_sequence_model`),
same split and same metric keys (`test_wmape`, `test_bias`, `*_wmape_cat_*`) as the naive and XGBoost runs.
Per-epoch `train_loss`, `val_loss`, `val_wmape_epoch` give a learning curve. Registered as `retailpulse-lstm`.

```bash
# host venv (torch/accelerate are NOT in requirements.txt: that file builds the Airflow image)
pip install -r requirements-dl.txt

# features must exist (trigger the DAG once), MLflow stack must be up
MLFLOW_TRACKING_URI=http://localhost:5000 accelerate launch src/train_dl.py
# PowerShell: $env:MLFLOW_TRACKING_URI="http://localhost:5000"; accelerate launch src/train_dl.py

# multi-GPU on one machine
accelerate launch --multi_gpu --num_processes 2 src/train_dl.py
```
No accelerate config is required (defaults = single process); `accelerate_config.yaml` is an explicit example.

- `src/windowing.py`: 28-day windows per item-store; the window for target date D holds rows D-28..D-1; D's own
  feature row (no sales) is passed separately. Windows lacking 28 contiguous prior days are dropped, not padded.
- Scaling uses per-series statistics from the training period only.
- Only the main process talks to MLflow, so multi-GPU does not create duplicate runs.
- Not part of the Airflow DAG (run it from the host).

---

# Phase 3 — Retrain gate + CI

## Model gate (`src/gate.py`)
A candidate is promoted to Production only if **both** hold (constants at the top of `src/gate.py`):
1. `test_wmape` is at least `WMAPE_MIN_RELATIVE_IMPROVEMENT` (2%) *relatively* better than Production: `(prod - cand) / prod >= 0.02`
2. `abs(candidate test_bias) <= abs(production test_bias)`

No Production model yet -> first promotion, promoted unconditionally (flagged `first_promotion`).
Metrics are read from the MLflow runs (`test_wmape`, `test_bias`), never recomputed. A missing or non-finite
metric raises `GateError` (the Airflow task fails); it never defaults to "promote". A reject touches nothing.

Registry: MLflow 2.16.2 -> **aliases** (stages are deprecated). Production = alias `production` on the registered
model `retailpulse-champion`, a single slot shared by every model family. Promote registers the candidate run's
`model` artifact as a new version there and moves the alias; the old version is tagged `status=archived`.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000     # PowerShell: $env:MLFLOW_TRACKING_URI="http://localhost:5000"
python -m src.gate --model-type xgboost --dry-run    # decide + print, do not touch the registry
python -m src.gate --model-type xgboost              # newest finished xgboost run
python -m src.gate --model-type lstm                 # e.g. gate the LSTM (trained on the host)
python -m src.gate --run-id <run_id>                 # a specific run
```
The DAG's final task `evaluate_and_gate` gates the newest XGBoost run (a reject is a successful task with the
decision in its log; missing metrics fail it). The LSTM is trained on the host, so gate it with the CLI.

## CI (`.github/workflows/ci.yml`)
Runs `pytest` on every push and pull request (CPU torch, so all tests including the DL ones run). It is separate
from the model gate. The MLflow integration test in `test_gate.py` skips itself when no server is reachable.
**This folder is not a git repository yet**: `git init` here (so `.github/` sits at the repo root), push to GitHub,
and the workflow will run.
