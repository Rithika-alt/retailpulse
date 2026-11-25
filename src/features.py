"""Leakage-safe feature engineering.

Contract: every feature in the row for date D is computed only from sales
strictly before D (and from price/calendar info, which is known in advance).
Input must have one row per (item_id, store_id, date) with no date gaps;
lags are row-based shifts, so gaps would silently break the time semantics.
"""
from __future__ import annotations

import pandas as pd

KEYS = ["item_id", "store_id"]
LAGS = (7, 14, 28)
WINDOWS = (7, 28)

# Fixed (month, day) holidays - deliberately simple.
HOLIDAYS = {(1, 1), (2, 14), (7, 4), (10, 31), (11, 11), (12, 24), (12, 25), (12, 31)}

LAG_COLS = [f"lag_{k}" for k in LAGS]
ROLL_COLS = [f"roll_{s}_{w}" for w in WINDOWS for s in ("mean", "std")]
CALENDAR_COLS = ["day_of_week", "is_weekend", "is_holiday"]
PRICE_COLS = ["sell_price", "price_change_flag", "promo_flag"]
FEATURE_COLUMNS = LAG_COLS + ROLL_COLS + CALENDAR_COLS + PRICE_COLS


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(KEYS + ["date"]).reset_index(drop=True)
    if out.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate (item_id, store_id, date) rows")
    gaps = out.groupby(KEYS, sort=False)["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError("Each item-store series must have contiguous daily dates")
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(KEYS, sort=False)["sales"]
    for k in LAGS:
        out[f"lag_{k}"] = g.shift(k)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean/std over the W days *before* D (series shifted by 1 first)."""
    out = df.copy()
    prior = out.groupby(KEYS, sort=False)["sales"].shift(1)
    grouped = prior.groupby([out["item_id"], out["store_id"]], sort=False)
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=w).mean())
        out[f"roll_std_{w}"] = grouped.transform(lambda s: s.rolling(w, min_periods=w).std())
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out["date"]
    out["day_of_week"] = d.dt.dayofweek.astype("int8")
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype("int8")
    out["is_holiday"] = [int((m, day) in HOLIDAYS) for m, day in zip(d.dt.month, d.dt.day)]
    out["is_holiday"] = out["is_holiday"].astype("int8")
    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price is known ahead of time, so current price is not leakage.

    price_change_flag: price differs from the price 7 days earlier.
    promo_flag: price is >10% below the median of the prior 28 days.
    """
    out = df.copy()
    g = out.groupby(KEYS, sort=False)["sell_price"]
    out["price_change_flag"] = (out["sell_price"] != g.shift(7)).astype("int8")
    prior = g.shift(1)
    med = prior.groupby([out["item_id"], out["store_id"]], sort=False).transform(
        lambda s: s.rolling(28, min_periods=7).median()
    )
    out["promo_flag"] = (out["sell_price"] < 0.9 * med).astype("int8")
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature table; drops warm-up rows lacking full lag/rolling history."""
    out = _prepare(df)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_calendar_features(out)
    out = add_price_features(out)
    out = out.dropna(subset=LAG_COLS + ROLL_COLS).reset_index(drop=True)
    id_cols = KEYS + (["cat_id"] if "cat_id" in out.columns else []) + ["date"]
    return out[id_cols + ["sales"] + FEATURE_COLUMNS]
