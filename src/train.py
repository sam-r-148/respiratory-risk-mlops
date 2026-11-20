"""Tasks C + D - retrain a candidate model and apply the quality gate.

Task C (retrain)
  * The training table is data/daily.csv, which already contains the previous training data PLUS
    every newly ingested batch, so the candidate is trained on existing + new data combined.
  * It uses the same feature pipeline (src/features.py) as the champion and the API.
  * The candidate is ALWAYS saved separately in models/candidate/ and tracked in MLflow.

Task D (quality gate)
  * Gate metric: macro-F1 on the most recent HOLDOUT_DAYS days. The candidate, the champion and the
    naive baseline are all scored on that same window, which neither model was trained on.
  * Rule: the candidate is ELIGIBLE if there is no champion, or its macro-F1 is strictly higher than
    the champion's by at least MIN_IMPROVEMENT. Otherwise the champion is retained.
  * The decision is appended to reports/decision_log.jsonl BEFORE the champion folder is touched, so
    the champion is never overwritten without a recorded decision.
  * With AUTO_PROMOTE=false an eligible candidate is only marked eligible; a human promotes it.

Usage:  python -m src.train
"""
import json
import os
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src import config
from src.features import build_dataset

warnings.filterwarnings("ignore", category=UserWarning)

ARTIFACT_FILES = ("model.joblib", "metadata.json", "reference_features.csv")


def candidate_models() -> dict:
    rs = config.RANDOM_STATE
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=rs, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200,
            class_weight="balanced", random_state=rs),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    return {"f1_macro": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred))}


def should_promote(challenger_f1: float, champion_f1: float | None, min_improvement: float) -> bool:
    """Eligible if there is no champion, or the challenger is strictly better by min_improvement."""
    return champion_f1 is None or challenger_f1 > champion_f1 + min_improvement


def _load_champion(model_dir):
    path = Path(model_dir) / "model.joblib"
    return joblib.load(path) if path.exists() else None


def _decision_reason(eligible: bool, auto_promote: bool, ch_f1: float, champ_f1: float | None,
                     min_improvement: float, champion_existed: bool) -> str:
    if champ_f1 is None:
        reason = ("Champion could not be evaluated; candidate treated as the new champion."
                  if champion_existed else "No champion exists yet; the first candidate becomes the champion.")
    elif eligible:
        reason = (f"Candidate macro-F1 {ch_f1:.4f} is higher than champion macro-F1 {champ_f1:.4f} "
                  f"by more than the required margin {min_improvement}.")
    else:
        reason = (f"Candidate macro-F1 {ch_f1:.4f} does not exceed champion macro-F1 {champ_f1:.4f} "
                  f"+ margin {min_improvement}; champion retained.")
    if eligible and not auto_promote:
        reason += " AUTO_PROMOTE is off: candidate is eligible but the champion was left unchanged."
    return reason


def run(data_path=config.DATA_PATH, model_dir=config.MODEL_DIR, reports_dir=config.REPORTS_DIR,
        holdout_days=config.HOLDOUT_DAYS, min_improvement=config.MIN_IMPROVEMENT,
        candidate_dir=None, auto_promote=config.AUTO_PROMOTE) -> dict:
    model_dir, reports_dir = Path(model_dir), Path(reports_dir)
    candidate_dir = Path(candidate_dir) if candidate_dir else config.candidate_dir_for(model_dir)

    daily = pd.read_csv(data_path)
    ds = build_dataset(daily)
    if len(ds) < config.MIN_ROWS:
        raise SystemExit(f"Only {len(ds)} usable rows (need {config.MIN_ROWS}). "
                         "Backfill more history: python -m src.ingest --start-date 2023-01-01")

    split = ds.index.max() - pd.Timedelta(days=holdout_days)
    train, hold = ds[ds.index <= split], ds[ds.index > split]
    X_tr, y_tr = train[config.FEATURES], train["target"]
    X_ho, y_ho = hold[config.FEATURES], hold["target"]

    # 1) choose the model type with time-ordered cross-validation on TRAIN only
    cv = TimeSeriesSplit(n_splits=3)
    cv_scores = {name: float(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="f1_macro").mean())
                 for name, m in candidate_models().items()}
    best_name = max(cv_scores, key=cv_scores.get)
    challenger = candidate_models()[best_name].fit(X_tr, y_tr)

    # 2) score candidate, champion and the naive baseline on the SAME holdout window
    ch = evaluate(challenger, X_ho, y_ho)
    baseline_f1 = float(f1_score(y_ho, hold["persistence"], average="macro"))
    champion = _load_champion(model_dir)
    champ = None
    if champion is not None:
        try:
            champ = evaluate(champion, X_ho, y_ho)
        except Exception as e:  # e.g. feature set changed -> old model unusable
            print(f"[train] champion could not be evaluated ({e}); treating as absent")

    # 3) quality gate
    champ_f1 = champ["f1_macro"] if champ else None
    eligible = should_promote(ch["f1_macro"], champ_f1, min_improvement)
    promote = bool(eligible and auto_promote)
    decision = "PROMOTED" if promote else ("ELIGIBLE_NOT_PROMOTED" if eligible else "RETAINED")
    reason = _decision_reason(eligible, auto_promote, ch["f1_macro"], champ_f1, min_improvement,
                              champion_existed=champion is not None)

    result = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(ds)), "n_train": int(len(train)), "n_holdout": int(len(hold)),
        "train_end": str(train.index.max().date()), "holdout_end": str(hold.index.max().date()),
        "cv_f1_macro": cv_scores, "selected_model": best_name,
        "challenger": ch, "champion": champ, "persistence_baseline_f1": baseline_f1,
        "gate_metric": config.GATE_METRIC, "min_improvement": min_improvement,
        "eligible_for_promotion": bool(eligible), "auto_promote": bool(auto_promote),
        "promoted": promote, "decision": decision, "decision_reason": reason,
    }

    # 4) ALWAYS save the candidate separately (Task C)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(challenger, candidate_dir / "model.joblib")
    X_tr.tail(1500).to_csv(candidate_dir / "reference_features.csv", index_label="date")
    meta = {**result, "features": config.FEATURES, "classes": config.CLASSES, "city": config.CITY}
    (candidate_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 5) record the decision BEFORE the champion can be touched (Task D)
    reports_dir.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": result["run_at"], "decision": decision, "eligible_for_promotion": bool(eligible),
             "gate_metric": config.GATE_METRIC, "candidate_f1_macro": ch["f1_macro"],
             "champion_f1_macro": champ_f1, "min_improvement": min_improvement,
             "persistence_baseline_f1": baseline_f1, "selected_model": best_name,
             "n_train": result["n_train"], "n_holdout": result["n_holdout"],
             "train_end": result["train_end"], "holdout_end": result["holdout_end"],
             "reason": reason, "candidate_path": str(candidate_dir)}
    with open(reports_dir / "decision_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    # 6) promote: copy the candidate files into the champion folder
    if promote:
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_FILES:
            shutil.copy2(candidate_dir / name, model_dir / name)

    (reports_dir / "latest_run.json").write_text(json.dumps(result, indent=2))
    return result


def log_to_mlflow(result: dict, candidate_dir=None, reports_dir=config.REPORTS_DIR) -> None:
    """Experiment tracking. Skipped silently if mlflow is not installed."""
    try:
        import mlflow
    except ImportError:
        print("[train] mlflow not installed; skipping experiment tracking")
        return
    candidate_dir = Path(candidate_dir) if candidate_dir else config.candidate_dir_for(config.MODEL_DIR)
    reports_dir = Path(reports_dir)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{config.ROOT / 'mlruns'}"))
    mlflow.set_experiment("respiratory-risk")
    with mlflow.start_run(run_name=result["run_at"]):
        mlflow.log_params({"selected_model": result["selected_model"], "n_train": result["n_train"],
                           "n_holdout": result["n_holdout"], "train_end": result["train_end"],
                           "gate_metric": result["gate_metric"], "min_improvement": result["min_improvement"]})
        mlflow.log_metrics({"challenger_f1": result["challenger"]["f1_macro"],
                            "challenger_accuracy": result["challenger"]["accuracy"],
                            "persistence_f1": result["persistence_baseline_f1"],
                            **({"champion_f1": result["champion"]["f1_macro"]} if result["champion"] else {})})
        mlflow.set_tags({"decision": result["decision"], "promoted": str(result["promoted"]),
                         "eligible_for_promotion": str(result["eligible_for_promotion"])})
        if (candidate_dir / "model.joblib").exists():  # the candidate is tracked every run
            mlflow.log_artifact(str(candidate_dir / "model.joblib"), artifact_path="candidate")
        for report in ("latest_run.json", "data_validation.json", "evaluation_report.json",
                       "decision_log.jsonl"):
            if (reports_dir / report).exists():
                mlflow.log_artifact(str(reports_dir / report), artifact_path="reports")


def main() -> None:
    result = run()
    log_to_mlflow(result)
    ch, champ = result["challenger"]["f1_macro"], result["champion"]
    print(f"[train] model={result['selected_model']}  candidate F1={ch:.3f}  "
          f"champion F1={'n/a' if not champ else format(champ['f1_macro'], '.3f')}  "
          f"naive baseline F1={result['persistence_baseline_f1']:.3f}")
    print(f"[train] decision: {result['decision']} - {result['decision_reason']}")
    gh_out = os.getenv("GITHUB_OUTPUT")  # lets the GitHub Actions workflow react to the decision
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"promoted={str(result['promoted']).lower()}\n")


if __name__ == "__main__":
    main()
