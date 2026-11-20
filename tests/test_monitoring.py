"""Tests for the monitoring + retraining workflow (Tasks A-E)."""
import json
import os
import subprocess
import sys

import pandas as pd
from sklearn.metrics import f1_score

from src import config, evaluate, train, validate
from src.ingest import make_synthetic


def _paths(tmp_path):
    return dict(data_path=tmp_path / "daily.csv", model_dir=tmp_path / "champion",
                reports_dir=tmp_path / "reports")


def _log_lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "reports" / "decision_log.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- Task A: validation
def test_validate_good_data_reports_size_and_target_distribution():
    rep = validate.validate(make_synthetic(300))
    assert rep["ok"] and rep["errors"] == []
    assert rep["n_rows"] == 300
    td = rep["target_distribution"]
    assert set(td) == set(config.CLASSES)
    assert sum(v["count"] for v in td.values()) == rep["usable_labelled_rows"]
    assert abs(sum(v["percent"] for v in td.values()) - 100) < 0.5


def test_validate_flags_missing_required_column():
    rep = validate.validate(make_synthetic(300).drop(columns=["pm2_5_mean"]))
    assert not rep["ok"]
    assert rep["missing_columns"] == ["pm2_5_mean"]


def test_validate_counts_missing_values_and_duplicate_dates():
    df = make_synthetic(300)
    df.loc[5, "no2_mean"] = None
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # 3 repeated records
    rep = validate.validate(df)
    assert rep["missing_values"]["no2_mean"]["count"] == 1
    assert rep["duplicate_dates"] == 3 and rep["duplicate_rows"] == 3
    assert rep["ok"]  # warnings only: the feature code handles both
    assert len(rep["warnings"]) == 2


def test_validate_rejects_too_little_data_and_bad_values():
    assert not validate.validate(make_synthetic(50))["ok"]
    bad = make_synthetic(300)
    bad["temp_mean"] = bad["temp_mean"].astype(object)
    bad.loc[3, "temp_mean"] = "hot"
    assert "temp_mean" in " ".join(validate.validate(bad)["errors"])
    assert not validate.validate(make_synthetic(300).iloc[0:0])["ok"]


def test_validate_run_writes_report(tmp_path):
    make_synthetic(300).to_csv(tmp_path / "daily.csv", index=False)
    rep = validate.run(tmp_path / "daily.csv", tmp_path / "reports")
    saved = json.loads((tmp_path / "reports" / "data_validation.json").read_text())
    assert saved["ok"] and saved["n_rows"] == rep["n_rows"] == 300


# ---------------------------------------------------------------- Task B: evaluation
def test_classification_metrics_values():
    y_true, y_pred = [0, 1, 2, 2], [0, 1, 2, 1]
    m = evaluate.classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["High"]["precision"] == 1.0 and m["per_class"]["High"]["recall"] == 0.5
    assert m["per_class"]["Moderate"]["precision"] == 0.5 and m["per_class"]["Moderate"]["recall"] == 1.0
    assert sum(map(sum, m["confusion_matrix"]["matrix"])) == 4
    # identical to the formula used by the quality gate
    assert abs(m["f1_macro"] - f1_score(y_true, y_pred, average="macro")) < 1e-12
    for k in ("precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted"):
        assert 0 <= m[k] <= 1


def test_evaluate_skips_when_there_is_no_champion(tmp_path):
    make_synthetic(300).to_csv(tmp_path / "daily.csv", index=False)
    rep = evaluate.run(tmp_path / "daily.csv", tmp_path / "champion", tmp_path / "reports")
    assert rep["skipped"] and "no champion" in rep["reason"]
    assert (tmp_path / "reports" / "evaluation_report.json").exists()


def test_evaluate_scores_champion_only_on_days_after_its_training(tmp_path):
    full = make_synthetic(500)
    full.iloc[:400].to_csv(tmp_path / "daily.csv", index=False)
    first = train.run(**_paths(tmp_path))
    assert first["promoted"]

    full.to_csv(tmp_path / "daily.csv", index=False)  # a new batch of 100 labelled days arrives
    rep = evaluate.run(tmp_path / "daily.csv", tmp_path / "champion", tmp_path / "reports")
    assert not rep["skipped"]
    assert rep["champion"]["train_end"] == first["train_end"]
    assert pd.Timestamp(rep["new_data"]["from"]) > pd.Timestamp(first["train_end"])
    m = rep["champion_metrics"]
    assert sum(v["support"] for v in m["per_class"].values()) == rep["new_data"]["n_rows"]
    assert sum(rep["new_data"]["target_distribution"].values()) == rep["new_data"]["n_rows"]
    assert 0 <= m["f1_macro"] <= 1 and "persistence_baseline_metrics" in rep
    assert (tmp_path / "reports" / "evaluation_report.json").exists()


# ---------------------------------------------------------------- Tasks C + D: candidate and gate
def test_candidate_is_always_saved_separately_and_every_decision_is_logged(tmp_path):
    make_synthetic(500).to_csv(tmp_path / "daily.csv", index=False)
    kw = _paths(tmp_path)
    first = train.run(**kw)
    champion_bytes = (tmp_path / "champion" / "model.joblib").read_bytes()
    assert first["promoted"] and first["decision"] == "PROMOTED"
    # a promoted champion is an exact copy of the candidate
    assert champion_bytes == (tmp_path / "candidate" / "model.joblib").read_bytes()

    second = train.run(**kw)  # same data -> tie -> retained
    assert second["decision"] == "RETAINED" and not second["eligible_for_promotion"]
    for name in train.ARTIFACT_FILES:  # candidate still saved, in its own folder
        assert (tmp_path / "candidate" / name).exists()
    assert (tmp_path / "champion" / "model.joblib").read_bytes() == champion_bytes  # untouched

    log = _log_lines(tmp_path)
    assert [e["decision"] for e in log] == ["PROMOTED", "RETAINED"]
    assert log[1]["gate_metric"] == "f1_macro" and log[1]["champion_f1_macro"] is not None
    assert "retained" in log[1]["reason"].lower()


def test_auto_promote_off_marks_eligible_but_leaves_champion_alone(tmp_path):
    make_synthetic(500).to_csv(tmp_path / "daily.csv", index=False)
    res = train.run(**_paths(tmp_path), auto_promote=False)
    assert res["eligible_for_promotion"] is True and res["promoted"] is False
    assert res["decision"] == "ELIGIBLE_NOT_PROMOTED"
    assert not (tmp_path / "champion" / "model.joblib").exists()
    assert (tmp_path / "candidate" / "model.joblib").exists()
    assert _log_lines(tmp_path)[0]["decision"] == "ELIGIBLE_NOT_PROMOTED"


def test_decision_is_written_before_the_champion_is_overwritten(tmp_path, monkeypatch):
    make_synthetic(500).to_csv(tmp_path / "daily.csv", index=False)
    seen = []
    real_copy = train.shutil.copy2

    def spy(src, dst, *a, **k):
        log = tmp_path / "reports" / "decision_log.jsonl"
        seen.append(log.exists() and len(log.read_text().splitlines()) == 1)
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(train.shutil, "copy2", spy)
    train.run(**_paths(tmp_path))
    assert seen and all(seen)


# ---------------------------------------------------------------- Task E: one-command workflow
def test_pipeline_runs_all_tasks_end_to_end(tmp_path):
    make_synthetic(500).to_csv(tmp_path / "daily.csv", index=False)
    env = {**os.environ, "DATA_PATH": str(tmp_path / "daily.csv"),
           "MODEL_DIR": str(tmp_path / "models" / "champion"), "REPORTS_DIR": str(tmp_path / "reports"),
           "MLFLOW_TRACKING_URI": f"file:{tmp_path / 'mlruns'}"}
    proc = subprocess.run([sys.executable, "-m", "src.pipeline"], cwd=config.ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for marker in ("TASK A", "TASK B", "TASK C", "TASK D", "decision", "EVIDENCE"):
        assert marker in proc.stdout
    for rel in ("reports/data_validation.json", "reports/evaluation_report.json",
                "reports/decision_log.jsonl", "reports/latest_run.json",
                "models/candidate/model.joblib", "models/champion/model.joblib"):
        assert (tmp_path / rel).exists(), rel
