import numpy as np
import pandas as pd
import pytest

from src.evaluate import bias, time_split, wmape
from src.features import FEATURE_COLUMNS, KEYS, build_features


def make_df(days=90, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days)
    frames = []
    for item in ("A", "B"):
        for store in ("S1", "S2"):
            frames.append(pd.DataFrame({
                "item_id": item, "store_id": store, "cat_id": "FOODS", "date": dates,
                "sell_price": 5.0, "sales": rng.integers(0, 20, days),
            }))
    return pd.concat(frames, ignore_index=True)


def test_lags_match_manual_shift():
    df = make_df()
    feats = build_features(df)
    row = feats.iloc[0]
    src = df[(df.item_id == row.item_id) & (df.store_id == row.store_id)].set_index("date")["sales"]
    for k in (7, 14, 28):
        assert row[f"lag_{k}"] == src[row.date - pd.Timedelta(days=k)]


def test_lags_do_not_cross_series():
    df = make_df()
    df.loc[df.item_id == "B", "sales"] += 1000
    feats = build_features(df)
    a = feats[feats.item_id == "A"]
    assert a[["lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28"]].max().max() < 100


def test_rolling_excludes_current_day():
    df = make_df()
    feats = build_features(df)
    row = feats.iloc[5]
    src = df[(df.item_id == row.item_id) & (df.store_id == row.store_id)].set_index("date")["sales"]
    prior7 = src[row.date - pd.Timedelta(days=7): row.date - pd.Timedelta(days=1)]
    assert row.roll_mean_7 == pytest.approx(prior7.mean())
    assert row.roll_std_7 == pytest.approx(prior7.std())


@pytest.mark.parametrize("offset", [0, 10, 40])
def test_no_leakage_features_at_D_ignore_sales_on_or_after_D(offset):
    """Features for date D must not change if sales on/after D are altered."""
    df = make_df(days=100)
    D = pd.Timestamp("2023-01-01") + pd.Timedelta(days=40 + offset)
    corrupted = df.copy()
    corrupted.loc[corrupted.date >= D, "sales"] = 1_000_000
    base = build_features(df)
    mod = build_features(corrupted)
    a = base[base.date == D].sort_values(KEYS).reset_index(drop=True)
    b = mod[mod.date == D].sort_values(KEYS).reset_index(drop=True)
    assert len(a) == 4
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS])
    # and the whole table up to D is untouched
    pd.testing.assert_frame_equal(
        base[base.date < D].reset_index(drop=True)[FEATURE_COLUMNS],
        mod[mod.date < D].reset_index(drop=True)[FEATURE_COLUMNS],
    )


def test_calendar_flags():
    df = make_df(days=400)
    feats = build_features(df).set_index("date")
    assert feats.loc["2023-12-25", "is_holiday"].eq(1).all()
    assert feats.loc["2023-12-20", "is_holiday"].eq(0).all()
    assert feats.loc["2023-12-23", "is_weekend"].eq(1).all()  # Saturday
    assert feats.loc["2023-12-25", "day_of_week"].eq(0).all()  # Monday


def test_price_change_and_promo_flags():
    df = make_df()
    d0 = pd.Timestamp("2023-01-01")
    mask = (df.date >= d0 + pd.Timedelta(days=50)) & (df.date < d0 + pd.Timedelta(days=55))
    df.loc[mask, "sell_price"] = 3.5  # 30% promo
    feats = build_features(df).set_index("date")
    assert feats.loc[d0 + pd.Timedelta(days=50), "promo_flag"].eq(1).all()
    assert feats.loc[d0 + pd.Timedelta(days=40), "promo_flag"].eq(0).all()
    # price differs from 7 days earlier on promo start; same later when price unchanged
    assert feats.loc[d0 + pd.Timedelta(days=50), "price_change_flag"].eq(1).all()
    assert feats.loc[d0 + pd.Timedelta(days=45), "price_change_flag"].eq(0).all()
    # returns to 5.0 after promo: 7d earlier it was 5.0 on day 56-> compare day 57 vs 50 (3.5)
    assert feats.loc[d0 + pd.Timedelta(days=57), "price_change_flag"].eq(1).all()


def test_rejects_date_gaps():
    df = make_df().drop(index=10)
    with pytest.raises(ValueError):
        build_features(df)


def test_metrics_and_split():
    assert wmape([10, 0], [8, 2]) == pytest.approx(4 / 10)
    assert bias([10, 0], [8, 2]) == pytest.approx(0.0)
    dates = pd.Series(pd.date_range("2023-01-01", periods=100).repeat(3))
    tr, va, te = time_split(dates)
    assert tr.sum() + va.sum() + te.sum() == 300
    assert dates[tr].max() < dates[va].min() and dates[va].max() < dates[te].min()
    assert (tr.sum(), va.sum(), te.sum()) == (240, 30, 30)
