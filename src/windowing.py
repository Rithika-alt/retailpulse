"""Turn the Phase 1 feature table into fixed-length input windows (numpy only).

For a target date D of one item-store series:
  * x      = the WINDOW rows D-28 .. D-1 (sales + every feature column), shape (W, F)
  * known  = the feature row of D itself WITHOUT sales (calendar, price/promo, lags,
             rolling stats - all leak-free by the Phase 1 contract), shape (K,)
  * y      = sales on D (scaled)
Leakage rule: nothing in x/known may depend on sales on or after D.

Short series / gaps: windows without 28 contiguous preceding days are DROPPED
(not padded) - zero padding would be indistinguishable from a stockout.

Scaling: per-series sales/price scales are fit on rows strictly before `train_end`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, KEYS

WINDOW = 28
SEQ_COLUMNS = ["sales"] + FEATURE_COLUMNS
KNOWN_COLUMNS = list(FEATURE_COLUMNS)
SALES_SCALED = ["sales", "lag_7", "lag_14", "lag_28",
                "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28"]
PRICE_SCALED = ["sell_price"]


def fit_scales(df: pd.DataFrame, train_end) -> pd.DataFrame:
    """Per-series mean sales / price from rows with date < train_end only."""
    train = df[df["date"] < pd.Timestamp(train_end)]
    stats = (
        train.groupby(KEYS)
        .agg(sales_scale=("sales", "mean"), price_scale=("sell_price", "mean"))
        .reset_index()
    )
    stats["sales_scale"] = stats["sales_scale"].where(stats["sales_scale"] > 0, 1.0)
    stats["price_scale"] = stats["price_scale"].where(stats["price_scale"] > 0, 1.0)
    return stats


@dataclass
class SequenceData:
    M: np.ndarray            # (n_rows, F) transformed [sales] + FEATURE_COLUMNS
    sales_scale: np.ndarray  # (n_rows,) per-row series sales scale
    item_idx: np.ndarray     # (n_rows,) int64
    store_idx: np.ndarray    # (n_rows,) int64
    target_rows: np.ndarray  # row index (into M) of each valid target
    meta: pd.DataFrame       # aligned with target_rows: ids, date, raw sales
    scales: pd.DataFrame
    window: int
    n_items: int
    n_stores: int
    n_dropped: int           # rows that could not form a full window

    @property
    def n_seq_features(self) -> int:
        return self.M.shape[1]

    @property
    def n_known_features(self) -> int:
        return self.M.shape[1] - 1

    def __len__(self) -> int:
        return len(self.target_rows)

    def get(self, i: int) -> dict:
        r = self.target_rows[i]
        return {
            "x": self.M[r - self.window : r],
            "known": self.M[r, 1:],
            "item": self.item_idx[r],
            "store": self.store_idx[r],
            "y": self.M[r, 0],
            "pos": np.int64(i),
        }

    def batch(self, positions) -> dict:
        """Materialise windows for several target positions."""
        rows = self.target_rows[np.asarray(positions, dtype=np.int64)]
        idx = rows[:, None] + np.arange(-self.window, 0)[None, :]
        return {
            "x": self.M[idx],
            "known": self.M[rows, 1:],
            "item": self.item_idx[rows],
            "store": self.store_idx[rows],
            "y": self.M[rows, 0],
        }

    def to_raw(self, pred_scaled, positions) -> np.ndarray:
        """Scaled predictions -> unit sales, clipped at 0."""
        rows = self.target_rows[np.asarray(positions, dtype=np.int64)]
        return np.clip(np.asarray(pred_scaled, dtype=float) * self.sales_scale[rows], 0, None)


def build_sequence_data(df: pd.DataFrame, train_end, window: int = WINDOW) -> SequenceData:
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    scales = fit_scales(df, train_end)
    df = df.merge(scales, on=KEYS, how="left")
    df["sales_scale"] = df["sales_scale"].fillna(scales["sales_scale"].mean())
    df["price_scale"] = df["price_scale"].fillna(scales["price_scale"].mean())

    ss = df["sales_scale"].to_numpy(float)
    ps = df["price_scale"].to_numpy(float)
    M = np.empty((len(df), len(SEQ_COLUMNS)), dtype=np.float32)
    for j, c in enumerate(SEQ_COLUMNS):
        v = df[c].to_numpy(float)
        if c in SALES_SCALED:
            v = v / ss
        elif c in PRICE_SCALED:
            v = v / ps
        elif c == "day_of_week":
            v = v / 6.0
        M[:, j] = v

    series = df.groupby(KEYS, sort=False).ngroup().to_numpy()
    dates = df["date"].to_numpy("datetime64[D]")
    n = len(df)
    i = np.arange(n)
    prev = np.clip(i - window, 0, None)
    valid = (
        (i >= window)
        & (series == series[prev])
        & ((dates - dates[prev]) == np.timedelta64(window, "D"))
    )
    target_rows = i[valid]

    item_cat = pd.Categorical(df["item_id"], categories=sorted(df["item_id"].unique()))
    store_cat = pd.Categorical(df["store_id"], categories=sorted(df["store_id"].unique()))
    meta_cols = KEYS + (["cat_id"] if "cat_id" in df.columns else []) + ["date", "sales"]
    return SequenceData(
        M=M,
        sales_scale=ss.astype(np.float32),
        item_idx=np.asarray(item_cat.codes, dtype=np.int64),
        store_idx=np.asarray(store_cat.codes, dtype=np.int64),
        target_rows=target_rows,
        meta=df.loc[target_rows, meta_cols].reset_index(drop=True),
        scales=scales,
        window=window,
        n_items=len(item_cat.categories),
        n_stores=len(store_cat.categories),
        n_dropped=int((~valid).sum()),
    )


def split_positions(data: SequenceData, train_end, val_end):
    """Positions of targets in train / val / test by TARGET date (never shuffled)."""
    d = data.meta["date"]
    train = np.flatnonzero((d < pd.Timestamp(train_end)).to_numpy())
    val = np.flatnonzero(((d >= pd.Timestamp(train_end)) & (d < pd.Timestamp(val_end))).to_numpy())
    test = np.flatnonzero((d >= pd.Timestamp(val_end)).to_numpy())
    return train, val, test
