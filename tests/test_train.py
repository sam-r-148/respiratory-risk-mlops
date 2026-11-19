import json

import pandas as pd

from src import train
from src.ingest import make_synthetic


def test_should_promote_logic():
    assert train.should_promote(0.5, None, 0.0)            # no champion yet
    assert train.should_promote(0.71, 0.70, 0.0)           # better
    assert not train.should_promote(0.70, 0.70, 0.0)       # tie keeps champion
    assert not train.should_promote(0.705, 0.70, 0.01)     # not better by enough
    assert not train.should_promote(0.60, 0.70, 0.0)       # worse


def test_first_run_promotes_and_second_identical_run_does_not(tmp_path):
    data = tmp_path / "daily.csv"
    make_synthetic(500).to_csv(data, index=False)
    kw = dict(data_path=data, model_dir=tmp_path / "champ", reports_dir=tmp_path / "rep")

    first = train.run(**kw)
    assert first["promoted"] is True
    assert (tmp_path / "champ" / "model.joblib").exists()
    meta = json.loads((tmp_path / "champ" / "metadata.json").read_text())
    assert meta["challenger"]["f1_macro"] == first["challenger"]["f1_macro"]
    assert (tmp_path / "champ" / "reference_features.csv").exists()

    second = train.run(**kw)  # same data, same seed -> tie -> keep champion
    assert second["promoted"] is False
    assert second["champion"]["f1_macro"] == first["challenger"]["f1_macro"]


def test_model_beats_naive_baseline_on_synthetic(tmp_path):
    data = tmp_path / "daily.csv"
    make_synthetic(600).to_csv(data, index=False)
    res = train.run(data_path=data, model_dir=tmp_path / "c", reports_dir=tmp_path / "r")
    assert res["challenger"]["f1_macro"] > 0.5


def test_too_little_data_is_refused(tmp_path):
    data = tmp_path / "daily.csv"
    make_synthetic(40).to_csv(data, index=False)
    try:
        train.run(data_path=data, model_dir=tmp_path / "c", reports_dir=tmp_path / "r")
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "usable rows" in str(e)


def test_drift_report_runs_after_training(tmp_path):
    from src import drift
    data = tmp_path / "daily.csv"
    make_synthetic(500).to_csv(data, index=False)
    kw = dict(model_dir=tmp_path / "c", reports_dir=tmp_path / "r")
    train.run(data_path=data, **kw)
    report = drift.run(data_path=data, **kw)
    assert "share_drifted" in report and 0 <= report["share_drifted"] <= 1
    assert (tmp_path / "r" / "drift.json").exists()


def test_drift_detects_injected_shift(tmp_path):
    """A large pollution regime shift in the latest weeks should be flagged far more than clean data."""
    from src import drift
    clean = make_synthetic(900, seed=21)
    shifted = clean.copy()
    shifted.loc[len(shifted) - 45:, ["pm2_5_mean", "pm10_mean", "no2_mean"]] *= 1.6
    shifted.loc[len(shifted) - 45:, "temp_mean"] += 5
    shares = {}
    for name, df in (("clean", clean), ("shifted", shifted)):
        d = tmp_path / name
        d.mkdir()
        df.to_csv(d / "daily.csv", index=False)
        train.run(data_path=d / "daily.csv", model_dir=d / "c", reports_dir=d / "r")
        shares[name] = drift.run(data_path=d / "daily.csv", model_dir=d / "c", reports_dir=d / "r")["share_drifted"]
    assert shares["shifted"] > shares["clean"]
