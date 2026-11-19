"""Train a challenger model, compare it to the current champion on the same recent
holdout window, and promote it only if it is better.

Usage:  python -m src.train
"""
import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src import config
from src.features import build_dataset

warnings.filterwarnings("ignore", category=UserWarning)


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
    """Promote if there is no champion, or the challenger is strictly better by min_improvement."""
    return champion_f1 is None or challenger_f1 > champion_f1 + min_improvement


def _load_champion(model_dir):
    path = model_dir / "model.joblib"
    return joblib.load(path) if path.exists() else None


def run(data_path=config.DATA_PATH, model_dir=config.MODEL_DIR, reports_dir=config.REPORTS_DIR,
        holdout_days=config.HOLDOUT_DAYS, min_improvement=config.MIN_IMPROVEMENT) -> dict:
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

    # 2) score challenger, champion and the naive baseline on the SAME holdout window
    ch = evaluate(challenger, X_ho, y_ho)
    baseline_f1 = float(f1_score(y_ho, hold["persistence"], average="macro"))
    champion = _load_champion(model_dir)
    champ = None
    if champion is not None:
        try:
            champ = evaluate(champion, X_ho, y_ho)
        except Exception as e:  # e.g. feature set changed -> old model unusable
            print(f"[train] champion could not be evaluated ({e}); treating as absent")
    promote = should_promote(ch["f1_macro"], champ["f1_macro"] if champ else None, min_improvement)

    result = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(ds)), "n_train": int(len(train)), "n_holdout": int(len(hold)),
        "train_end": str(train.index.max().date()), "holdout_end": str(hold.index.max().date()),
        "cv_f1_macro": cv_scores, "selected_model": best_name,
        "challenger": ch, "champion": champ, "persistence_baseline_f1": baseline_f1,
        "promoted": bool(promote),
    }

    if promote:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(challenger, model_dir / "model.joblib")
        X_tr.tail(1500).to_csv(model_dir / "reference_features.csv", index_label="date")
        meta = {**result, "features": config.FEATURES, "classes": config.CLASSES, "city": config.CITY}
        (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latest_run.json").write_text(json.dumps(result, indent=2))
    return result


def log_to_mlflow(result: dict, model_dir=config.MODEL_DIR) -> None:
    """Experiment tracking. Skipped silently if mlflow is not installed."""
    try:
        import mlflow
    except ImportError:
        print("[train] mlflow not installed; skipping experiment tracking")
        return
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{config.ROOT / 'mlruns'}"))
    mlflow.set_experiment("respiratory-risk")
    with mlflow.start_run(run_name=result["run_at"]):
        mlflow.log_params({"selected_model": result["selected_model"], "n_train": result["n_train"],
                           "n_holdout": result["n_holdout"], "train_end": result["train_end"]})
        mlflow.log_metrics({"challenger_f1": result["challenger"]["f1_macro"],
                            "challenger_accuracy": result["challenger"]["accuracy"],
                            "persistence_f1": result["persistence_baseline_f1"],
                            **({"champion_f1": result["champion"]["f1_macro"]} if result["champion"] else {})})
        mlflow.set_tag("promoted", str(result["promoted"]))
        if result["promoted"] and (model_dir / "model.joblib").exists():
            mlflow.log_artifact(str(model_dir / "model.joblib"))
        mlflow.log_artifact(str(config.REPORTS_DIR / "latest_run.json"))


def main() -> None:
    result = run()
    log_to_mlflow(result)
    ch, champ = result["challenger"]["f1_macro"], result["champion"]
    print(f"[train] model={result['selected_model']}  challenger F1={ch:.3f}  "
          f"champion F1={'n/a' if not champ else format(champ['f1_macro'], '.3f')}  "
          f"naive baseline F1={result['persistence_baseline_f1']:.3f}")
    print(f"[train] {'PROMOTED new champion' if result['promoted'] else 'KEPT existing champion'}")
    gh_out = os.getenv("GITHUB_OUTPUT")  # lets the GitHub Actions workflow react to the decision
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"promoted={str(result['promoted']).lower()}\n")


if __name__ == "__main__":
    main()
