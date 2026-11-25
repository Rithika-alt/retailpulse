"""Metrics and time-based splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(actual, predicted) -> float:
    """sum|a-p| / sum|a| over the whole set (volume-weighted, not per-item mean)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denom = np.abs(a).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(a - p).sum() / denom)


def bias(actual, predicted) -> float:
    """mean(predicted - actual); positive = over-forecast."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(p - a))


def wmape_by_group(df: pd.DataFrame, group_col: str, actual_col: str, pred_col: str) -> dict[str, float]:
    return {
        str(k): wmape(g[actual_col], g[pred_col]) for k, g in df.groupby(group_col, sort=True, observed=True)
    }


def time_split(dates: pd.Series, train_frac: float = 0.8, val_frac: float = 0.1):
    """Boolean masks (train, val, test) split on unique dates; never shuffled."""
    dates = pd.to_datetime(pd.Series(dates))
    unique = np.sort(dates.unique())
    n = len(unique)
    train_end = unique[int(n * train_frac)]  # first val date
    val_end = unique[int(n * (train_frac + val_frac))]  # first test date
    train = dates < train_end
    val = (dates >= train_end) & (dates < val_end)
    test = dates >= val_end
    return train.to_numpy(), val.to_numpy(), test.to_numpy()
