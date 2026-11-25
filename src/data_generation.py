"""Synthetic M5-style sales generator.

Schema (long format): item_id, store_id, cat_id, date, sell_price, sales
Patterns: weekly seasonality, mild yearly seasonality + trend, promo windows
(price -20%, demand ~1.8x), and stockout-like zero-sales runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["item_id", "store_id", "cat_id", "date", "sell_price", "sales"]
REQUIRED_COLUMNS = ["item_id", "store_id", "date", "sell_price", "sales"]

STORES = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"]
CATEGORIES = ["FOODS", "HOUSEHOLD", "HOBBIES"]
# Monday=0 ... Sunday=6
WEEKDAY_MULT = np.array([0.85, 0.85, 0.90, 0.95, 1.10, 1.35, 1.20])


def generate(
    n_items: int = 50,
    n_stores: int = 5,
    start: str = "2023-01-01",
    days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    weekday = dates.dayofweek.to_numpy()
    yearly = 1 + 0.10 * np.sin(2 * np.pi * t / 365.25)

    stores = STORES[:n_stores]
    frames = []
    for i in range(n_items):
        item_id = f"ITEM_{i:03d}"
        cat_id = CATEGORIES[i % len(CATEGORIES)]
        item_price = float(np.round(rng.lognormal(mean=1.8, sigma=0.5), 2))
        item_rate = float(rng.lognormal(mean=1.6, sigma=0.7))
        for store in stores:
            rate = item_rate * rng.uniform(0.6, 1.5)
            price = np.full(days, item_price * rng.uniform(0.95, 1.05))
            trend = 1 + rng.uniform(-0.1, 0.2) * t / days

            promo = np.zeros(days, dtype=bool)
            for _ in range(3):
                s = int(rng.integers(30, days - 15))
                promo[s : s + int(rng.integers(5, 11))] = True
            price = np.where(promo, price * 0.8, price).round(2)

            lam = rate * WEEKDAY_MULT[weekday] * yearly * trend
            lam = np.where(promo, lam * 1.8, lam)
            sales = rng.poisson(lam).astype(int)

            for _ in range(2):  # stockout runs
                s = int(rng.integers(30, days - 10))
                sales[s : s + int(rng.integers(3, 8))] = 0

            frames.append(
                pd.DataFrame(
                    {
                        "item_id": item_id,
                        "store_id": store,
                        "cat_id": cat_id,
                        "date": dates,
                        "sell_price": price,
                        "sales": sales,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)[RAW_COLUMNS]


def validate_raw(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")


def generate_or_load(raw_path: str | os.PathLike, **gen_kwargs) -> str:
    """Use an existing long-format CSV at raw_path, else generate one there.

    Returns the path. A real M5 export must already be melted to the long
    schema above (wide d_1..d_N format is not handled in Phase 1).
    """
    path = Path(raw_path)
    if path.exists():
        validate_raw(pd.read_csv(path, nrows=5))
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(**gen_kwargs).to_csv(path, index=False)
    return str(path)


if __name__ == "__main__":
    out = generate_or_load(Path("data/raw/sales.csv"))
    print(f"Raw data at {out}")
