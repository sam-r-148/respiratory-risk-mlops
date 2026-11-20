"""Task B - evaluate the EXISTING (champion) model on new labelled data.

"New labelled data" = every labelled day that falls AFTER the champion's training end date
(metadata.json -> train_end). The champion has never seen those days, so its score on them shows
whether it is still reliable. The same window is also scored with the naive "tomorrow = today"
baseline for context.

Classification metrics: accuracy, precision, recall, F1 (macro, weighted and per class) and the
confusion matrix. The macro-F1 here uses exactly the same formula as the quality gate in train.py.

Usage:  python -m src.evaluate
Output: reports/evaluation_report.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src import config
from src.features import build_dataset

LABELS = [0, 1, 2]
MIN_RELIABLE_ROWS = 30  # below this the metrics are noisy; we still report them with a warning


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall, F1 (macro / weighted / per class) and the confusion matrix."""
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out["per_class"] = {config.CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                            "f1": float(f[i]), "support": int(s[i])}
                        for i in range(len(LABELS))}
    out["confusion_matrix"] = {"labels": config.CLASSES,
                               "rows_are": "actual", "columns_are": "predicted",
                               "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}
    return out


def _skipped(reason: str) -> dict:
    return {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "skipped": True, "reason": reason}


def evaluate_champion(data_path, model_dir) -> dict:
    model_dir = Path(model_dir)
    model_path, meta_path = model_dir / "model.joblib", model_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        return _skipped("no champion model yet (run: python -m src.train)")

    meta = json.loads(meta_path.read_text())
    ds = build_dataset(pd.read_csv(data_path))
    train_end = pd.Timestamp(meta["train_end"])
    new = ds[ds.index > train_end]
    if new.empty:
        return _skipped(f"no labelled days after the champion's training end ({meta['train_end']})")

    X, y = new[config.FEATURES], new["target"]
    try:
        pred = joblib.load(model_path).predict(X)
    except Exception as e:  # e.g. the feature set changed since the champion was trained
        return _skipped(f"champion could not score the new data ({e})")

    counts = y.value_counts().reindex(LABELS, fill_value=0)
    warnings = []
    if len(new) < MIN_RELIABLE_ROWS:
        warnings.append(f"only {len(new)} new labelled rows (< {MIN_RELIABLE_ROWS}); metrics are noisy")
    missing_classes = [config.CLASSES[i] for i in LABELS if counts[i] == 0]
    if missing_classes:
        warnings.append(f"class(es) absent from the new data: {missing_classes}")

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skipped": False,
        "champion": {"model": meta.get("selected_model"), "trained_at": meta.get("run_at"),
                     "train_end": meta["train_end"]},
        "new_data": {"n_rows": int(len(new)),
                     "from": str(new.index.min().date()), "to": str(new.index.max().date()),
                     "target_distribution": {config.CLASSES[i]: int(counts[i]) for i in LABELS}},
        "champion_metrics": classification_metrics(y, pred),
        "persistence_baseline_metrics": classification_metrics(y, new["persistence"]),
        "warnings": warnings,
    }


def print_report(rep: dict) -> None:
    if rep["skipped"]:
        print(f"[evaluate] skipped: {rep['reason']}")
        return
    m, nd = rep["champion_metrics"], rep["new_data"]
    print(f"[evaluate] champion ({rep['champion']['model']}, trained through {rep['champion']['train_end']}) "
          f"on {nd['n_rows']} NEW labelled days ({nd['from']} -> {nd['to']})")
    print(f"[evaluate] accuracy={m['accuracy']:.3f}  precision(macro)={m['precision_macro']:.3f}  "
          f"recall(macro)={m['recall_macro']:.3f}  F1(macro)={m['f1_macro']:.3f}")
    for cls, v in m["per_class"].items():
        print(f"[evaluate]   {cls:<9} precision={v['precision']:.3f}  recall={v['recall']:.3f}  "
              f"f1={v['f1']:.3f}  support={v['support']}")
    print(f"[evaluate] naive baseline (tomorrow = today) F1(macro)="
          f"{rep['persistence_baseline_metrics']['f1_macro']:.3f}")
    for w in rep["warnings"]:
        print(f"[evaluate] WARNING: {w}")


def run(data_path=config.DATA_PATH, model_dir=config.MODEL_DIR, reports_dir=config.REPORTS_DIR) -> dict:
    rep = evaluate_champion(data_path, model_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "evaluation_report.json").write_text(json.dumps(rep, indent=2))
    print_report(rep)
    return rep


if __name__ == "__main__":
    run()
