import numpy as np
import pandas as pd

from src.config import FEATURES
from src.features import build_dataset, respiratory_score, score_to_class
from src.ingest import make_synthetic


def _row(**kw):
    base = dict(pm2_5_mean=20, o3_max=50, no2_mean=30, temp_min=20)
    base.update(kw)
    return pd.DataFrame([base])


def test_score_bands():
    assert respiratory_score(_row(pm2_5_mean=10)).iloc[0] == 0
    assert respiratory_score(_row(pm2_5_mean=45)).iloc[0] == 1
    assert respiratory_score(_row(pm2_5_mean=150)).iloc[0] == 4
    # extra triggers add one point each
    assert respiratory_score(_row(pm2_5_mean=150, o3_max=120, no2_mean=90, temp_min=5)).iloc[0] == 7


def test_score_to_class():
    s = pd.Series([0, 1, 2, 3, 4, 7, np.nan])
    c = score_to_class(s)
    assert c.iloc[:6].tolist() == [0, 0, 1, 1, 2, 2]
    assert np.isnan(c.iloc[6])


def test_api_row_matches_training_features():
    """The serving path must build the same score_today feature as training."""
    from src.features import row_from_record
    raw = make_synthetic(60)
    ds = build_dataset(raw)
    last_date = ds.index[-1]
    rec = raw[raw["date"] == last_date.strftime("%Y-%m-%d")].iloc[0].to_dict()
    row = row_from_record(rec)
    assert row.loc[0, "score_today"] == ds.loc[last_date, "score_today"]
    assert list(row.columns) == FEATURES


def test_missing_inputs_are_not_labelled_low():
    assert np.isnan(respiratory_score(_row(pm2_5_mean=np.nan)).iloc[0])


def test_dataset_shape_and_columns():
    ds = build_dataset(make_synthetic(300))
    assert set(FEATURES + ["target", "persistence"]) <= set(ds.columns)
    assert ds["target"].isin([0, 1, 2]).all()
    assert ds["target"].nunique() == 3, "synthetic data should contain all three classes"
    assert not ds.isna().any().any()


def test_no_label_leakage_into_features():
    """Changing tomorrow's values must not change today's features."""
    raw = make_synthetic(120)
    a = build_dataset(raw)
    changed = raw.copy()
    changed.loc[changed.index[-30], "pm2_5_mean"] += 500
    b = build_dataset(changed)
    day_before = pd.Timestamp(raw["date"].iloc[-31])
    pd.testing.assert_series_equal(a.loc[day_before, FEATURES], b.loc[day_before, FEATURES])
    # ...but the label (tomorrow's risk) is allowed to change
    assert b.loc[day_before, "target"] == 2
