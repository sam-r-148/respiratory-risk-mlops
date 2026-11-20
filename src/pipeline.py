"""Task E - run the whole monitoring + retraining workflow with one command and print the evidence.

    python -m src.pipeline            # use the data already in data/daily.csv
    python -m src.pipeline --ingest   # first pull the newest batch from Open-Meteo

Steps
  A  validate the incoming data           -> reports/data_validation.json
  B  evaluate the existing champion       -> reports/evaluation_report.json
  C  retrain a candidate (kept separate)  -> models/candidate/  (+ MLflow run)
  D  quality gate + recorded decision     -> reports/decision_log.jsonl, reports/latest_run.json

The champion in models/champion/ is only replaced when the gate says so, and only after the decision
has been written to the decision log.
"""
import argparse
import subprocess
import sys

from src import config, evaluate, train, validate


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    p = argparse.ArgumentParser(description="Monitoring and retraining workflow (Tasks A-D)")
    p.add_argument("--ingest", action="store_true", help="pull the newest data batch first")
    a = p.parse_args()

    if a.ingest:
        banner("STEP 0 - Load the new data batch (Open-Meteo)")
        subprocess.run([sys.executable, "-m", "src.ingest"], check=True)

    banner("TASK A - Prepare and validate the new data")
    v = validate.run()
    if not v["ok"]:
        raise SystemExit("[pipeline] data validation failed - stopping before any model is touched")

    banner("TASK B - Evaluate the existing (champion) model on new labelled data")
    ev = evaluate.run()

    banner("TASK C - Retrain a candidate on existing + new data")
    banner("TASK D - Quality gate (metric: macro-F1 on the same holdout window)")
    result = train.run()
    train.log_to_mlflow(result)
    ch, champ = result["challenger"], result["champion"]
    print(f"[train] selected model        : {result['selected_model']}")
    print(f"[train] training rows / holdout: {result['n_train']} / {result['n_holdout']} "
          f"(training ends {result['train_end']}, holdout ends {result['holdout_end']})")
    print(f"[train] candidate  macro-F1   : {ch['f1_macro']:.4f}   accuracy {ch['accuracy']:.4f}")
    print(f"[train] champion   macro-F1   : "
          f"{'n/a' if not champ else format(champ['f1_macro'], '.4f')}")
    print(f"[train] naive baseline macro-F1: {result['persistence_baseline_f1']:.4f}")
    print(f"[gate ] decision              : {result['decision']}")
    print(f"[gate ] reason                : {result['decision_reason']}")

    banner("EVIDENCE - files produced by this run")
    cand = config.candidate_dir_for(config.MODEL_DIR)
    files = [("Data validation report", config.REPORTS_DIR / "data_validation.json"),
             ("Evaluation report (champion)", config.REPORTS_DIR / "evaluation_report.json"),
             ("Candidate model artifact", cand / "model.joblib"),
             ("Candidate metadata", cand / "metadata.json"),
             ("Quality-gate decision log", config.REPORTS_DIR / "decision_log.jsonl"),
             ("Latest run summary", config.REPORTS_DIR / "latest_run.json"),
             ("Current champion model", config.MODEL_DIR / "model.joblib")]
    for label, path in files:
        try:
            shown = path.relative_to(config.ROOT)
        except ValueError:
            shown = path
        print(f"  [{'x' if path.exists() else ' '}] {label:<30} {shown}")
    if ev.get("skipped"):
        print(f"\n(Task B was skipped: {ev['reason']})")
    print("\nMLflow UI: mlflow ui --backend-store-uri mlruns   (experiment 'respiratory-risk')")


if __name__ == "__main__":
    main()
