import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, KEYS, build_features
from src.tests.test_features import make_df
from src.windowing import SEQ_COLUMNS, build_sequence_data, fit_scales, split_positions

W = 28
START = pd.Timestamp("2023-01-01")


def day(n):
    return START + pd.Timedelta(days=n)


@pytest.fixture(scope="module")
def table():
    return build_features(make_df(days=140, seed=1))


def pos_of(data, item, store, date):
    m = data.meta
    hit = m.index[(m.item_id == item) & (m.store_id == store) & (m.date == date)]
    assert len(hit) == 1
    return int(hit[0])


def test_shapes_and_counts(table):
    data = build_sequence_data(table, day(100), W)
    b = data.batch(np.arange(len(data)))
    n = len(data)
    assert b["x"].shape == (n, W, len(SEQ_COLUMNS))
    assert b["known"].shape == (n, len(FEATURE_COLUMNS))
    assert b["y"].shape == (n,)
    # table has 112 rows/series (days 28..139); targets need 28 prior rows -> 84/series, 4 series
    assert n == 4 * (112 - W)
    assert data.n_dropped == 4 * W


def test_window_matches_manual_slice_and_ends_at_D_minus_1(table):
    data = build_sequence_data(table, day(100), W)
    D = day(90)
    p = pos_of(data, "A", "S1", D)
    b = data.batch([p])
    series = table[(table.item_id == "A") & (table.store_id == "S1")].sort_values("date").reset_index(drop=True)
    end = series.index[series.date == D][0]
    ss = data.scales.query("item_id=='A' and store_id=='S1'").sales_scale.iloc[0]
    manual = series.loc[end - W : end - 1, "sales"].to_numpy() / ss
    np.testing.assert_allclose(b["x"][0, :, 0], manual, rtol=1e-5)
    assert series.date[end - 1] == D - pd.Timedelta(days=1)  # last window day is D-1
    np.testing.assert_allclose(b["y"][0], series.loc[end, "sales"] / ss, rtol=1e-5)


@pytest.mark.parametrize("d", [60, 90, 120])
def test_no_leakage_window_for_D_ignores_sales_on_or_after_D(d):
    """Altering sales on/after D must not change the inputs for target D."""
    table = build_features(make_df(days=160, seed=2))
    D = day(d)
    cutoff = min(day(70), D)  # scales fit on rows < D, so they cannot see the mutation
    corrupted_raw = make_df(days=160, seed=2)
    corrupted_raw.loc[corrupted_raw.date >= D, "sales"] = 1_000_000
    corrupted = build_features(corrupted_raw)

    base = build_sequence_data(table, cutoff, W)
    mod = build_sequence_data(corrupted, cutoff, W)
    for item, store in [("A", "S1"), ("B", "S2")]:
        pb, pm = pos_of(base, item, store, D), pos_of(mod, item, store, D)
        a, c = base.batch([pb]), mod.batch([pm])
        np.testing.assert_array_equal(a["x"], c["x"])
        np.testing.assert_array_equal(a["known"], c["known"])
        assert a["y"][0] != c["y"][0]  # only the target itself reflects the change


def test_windows_do_not_cross_series(table):
    t = table.copy()
    t.loc[t.item_id == "B", "sales"] += 10_000
    data = build_sequence_data(t, day(100), W)
    ss = data.scales.set_index(KEYS).sales_scale
    for (item, store) in [("A", "S1"), ("A", "S2")]:
        p = np.flatnonzero(((data.meta.item_id == item) & (data.meta.store_id == store)).to_numpy())
        x = data.batch(p)["x"][:, :, 0] * ss[(item, store)]
        assert x.max() < 100  # no B values leaked into A windows


def test_series_shorter_than_window_is_dropped_not_padded():
    raw = make_df(days=140, seed=3)
    short = raw[(raw.item_id == "A") & (raw.store_id == "S1")].iloc[:50].assign(item_id="NEW")
    table = build_features(pd.concat([raw, short], ignore_index=True))
    data = build_sequence_data(table, day(100), W)
    assert "NEW" not in set(data.meta.item_id)  # 22 feature rows <= 28 -> no window
    assert data.n_items == 3  # but it is still in the vocabulary


def test_date_gap_drops_every_window_spanning_it(table):
    t = table.drop(index=table.index[(table.item_id == "A") & (table.store_id == "S1") & (table.date == day(80))])
    data = build_sequence_data(t, day(100), W)
    dates = data.meta.loc[(data.meta.item_id == "A") & (data.meta.store_id == "S1"), "date"]
    # targets whose 28-day window includes day 80 (i.e. D in 81..108) must be gone, and day 80 itself
    banned = {day(n) for n in range(80, 109)}
    assert banned.isdisjoint(set(dates))


def test_scales_use_train_rows_only(table):
    cutoff = day(100)
    base = fit_scales(table, cutoff)
    mutated = table.copy()
    mutated.loc[mutated.date >= cutoff, ["sales", "sell_price"]] = 9_999
    pd.testing.assert_frame_equal(base, fit_scales(mutated, cutoff))


def test_split_positions_are_time_ordered_and_disjoint(table):
    data = build_sequence_data(table, day(100), W)
    tr, va, te = split_positions(data, day(100), day(120))
    d = data.meta.date
    assert d.iloc[tr].max() < day(100) <= d.iloc[va].min() and d.iloc[va].max() < day(120) <= d.iloc[te].min()
    assert len(tr) + len(va) + len(te) == len(data)
