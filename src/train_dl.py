"""Train the LSTM sequence model with Hugging Face Accelerate; log to MLflow.

    accelerate launch src/train_dl.py            # 1 or many GPUs / CPU
    python -m src.train_dl                       # plain single process

No device is hardcoded: Accelerate places model, optimizer state and batches.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
from pathlib import Path

# `accelerate launch src/train_dl.py` puts src/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.dl_model import LSTMForecaster  # noqa: E402
from src.evaluate import time_split, wmape  # noqa: E402
from src.train import EXPERIMENT, _log_split_metrics  # noqa: E402  (same metric logging as Phase 1)
from src.windowing import SequenceData, build_sequence_data, split_positions  # noqa: E402

RUN_NAME = "lstm_sequence_model"
REGISTERED_MODEL = "retailpulse-lstm"


class WindowDataset(Dataset):
    """Lazy view over SequenceData restricted to a set of target positions."""

    def __init__(self, data: SequenceData, positions: np.ndarray):
        self.data, self.positions = data, positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        return self.data.get(int(self.positions[i]))


def predict(accelerator: Accelerator, model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Scaled predictions for every sample, sorted by position (same on all ranks)."""
    model.eval()
    preds, poss = [], []
    with torch.no_grad():
        for b in loader:
            p = model(b["x"], b["known"], b["item"], b["store"])
            p, pos = accelerator.gather_for_metrics((p, b["pos"]))
            preds.append(p.float().cpu())
            poss.append(pos.cpu())
    pos = torch.cat(poss).numpy()
    p = torch.cat(preds).numpy()
    order = np.argsort(pos)
    return pos[order], p[order]


def evaluate_split(data: SequenceData, pos: np.ndarray, pred_scaled: np.ndarray):
    """val_loss (scaled MSE) and the meta frame + raw predictions for metric logging."""
    y_scaled = data.M[data.target_rows[pos], 0]
    loss = float(np.mean((pred_scaled - y_scaled) ** 2))
    raw = data.to_raw(pred_scaled, pos)
    return loss, data.meta.iloc[pos].reset_index(drop=True), raw


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features.parquet")
    p.add_argument("--window", type=int, default=28)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed-precision", default="no", choices=["no", "fp16", "bf16"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    main_proc = accelerator.is_main_process

    # ---- data: identical split to Phase 1 (evaluate.time_split on feature-table dates)
    df = pd.read_parquet(args.features)
    tr, va, te = time_split(df["date"], 0.8, 0.1)
    train_end, val_end = df.loc[va, "date"].min(), df.loc[te, "date"].min()
    data = build_sequence_data(df, train_end, args.window)
    pos_tr, pos_va, pos_te = split_positions(data, train_end, val_end)
    if len(pos_va) != int(va.sum()) or len(pos_te) != int(te.sum()):
        raise RuntimeError(
            f"Window val/test rows ({len(pos_va)}/{len(pos_te)}) differ from the Phase 1 split "
            f"({int(va.sum())}/{int(te.sum())}); metrics would not be comparable."
        )

    def loader(pos, shuffle):
        return DataLoader(WindowDataset(data, pos), batch_size=args.batch_size, shuffle=shuffle)

    model = LSTMForecaster(
        data.n_seq_features, data.n_known_features, data.n_items, data.n_stores,
        hidden=args.hidden, layers=args.layers, dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, train_dl, val_dl, test_dl = accelerator.prepare(
        model, optimizer, loader(pos_tr, True), loader(pos_va, False), loader(pos_te, False)
    )
    loss_fn = torch.nn.MSELoss()

    if main_proc:
        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(EXPERIMENT)
    run = mlflow.start_run(run_name=RUN_NAME) if main_proc else contextlib.nullcontext()
    with run:
        if main_proc:
            mlflow.set_tag("model_type", "lstm")
            mlflow.log_params({
                "window": args.window, "hidden": args.hidden, "layers": args.layers,
                "dropout": args.dropout, "lr": args.lr, "batch_size": args.batch_size,
                "epochs_max": args.epochs, "patience": args.patience, "seed": args.seed,
                "mixed_precision": args.mixed_precision,
                "num_processes": accelerator.num_processes, "device": str(accelerator.device.type),
                "train_rows": len(pos_tr), "val_rows": len(pos_va), "test_rows": len(pos_te),
                "train_end": str((train_end - pd.Timedelta(days=1)).date()),
                "val_end": str((val_end - pd.Timedelta(days=1)).date()),
                "rows_without_full_window": data.n_dropped,
                "loss": "MSE on per-series-scaled sales",
            })

        best_val, best_state, stale = float("inf"), None, 0
        for epoch in range(args.epochs):
            model.train()
            tot = torch.zeros(2, device=accelerator.device)
            for b in train_dl:
                optimizer.zero_grad()
                loss = loss_fn(model(b["x"], b["known"], b["item"], b["store"]), b["y"])
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                n = b["y"].shape[0]
                tot += torch.tensor([loss.detach().float() * n, n], device=accelerator.device)
            tot = accelerator.reduce(tot, reduction="sum")
            train_loss = float(tot[0] / tot[1])

            pos, p = predict(accelerator, model, val_dl)
            val_loss, vmeta, vraw = evaluate_split(data, pos, p)
            val_wmape = wmape(vmeta["sales"], vraw)
            if main_proc:
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, "val_wmape_epoch": val_wmape},
                    step=epoch,
                )
                print(f"epoch {epoch:2d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_wmape={val_wmape:.4f}")

            # val_loss is computed from gathered preds -> identical on all ranks -> synced stopping
            if val_loss < best_val - 1e-5:
                best_val, stale = val_loss, 0
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in accelerator.unwrap_model(model).state_dict().items()}
                )
            else:
                stale += 1
                if stale >= args.patience:
                    break

        accelerator.unwrap_model(model).load_state_dict(best_state)
        results = {}
        for name, dl in (("val", val_dl), ("test", test_dl)):
            pos, p = predict(accelerator, model, dl)
            _, meta, raw = evaluate_split(data, pos, p)
            if main_proc:
                results[name] = _log_split_metrics(meta, raw, name)  # same keys as Phase 1

        if main_proc:
            mlflow.log_metric("best_val_loss", best_val)
            mlflow.log_text(data.scales.to_csv(index=False), "scalers/series_scales.csv")
            mlflow.pytorch.log_model(
                accelerator.unwrap_model(model), artifact_path="model",
                code_paths=["src"], registered_model_name=REGISTERED_MODEL,
            )
            t = results["test"]
            print(f"TEST wmape={t['test_wmape']:.4f} bias={t['test_bias']:.4f}")

    accelerator.wait_for_everyone()
    accelerator.end_training()
    return results if main_proc else None


if __name__ == "__main__":
    main()
