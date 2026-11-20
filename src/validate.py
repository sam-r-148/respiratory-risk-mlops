"""Task A - prepare and validate a batch of incoming data.

Checks, in order:
  1. required columns are present
  2. missing values (per column)
  3. duplicate records / IDs (the record ID here is the calendar date)
  4. dataset size and the target (risk class) distribution

Errors stop the pipeline (exit code 1); warnings are recorded and the pipeline continues, because the
feature code handles them (duplicate dates are collapsed, gaps of up to 2 days are interpolated).

Usage:  python -m src.validate [--data data/daily.csv]
Output: reports/data_validation.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.features import build_dataset


def validate(daily: pd.DataFrame) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    n_rows = int(len(daily))
    if n_rows == 0:
        errors.append("dataset is empty")

    # 1) required columns
    missing_cols = [c for c in config.REQUIRED_COLS if c not in daily.columns]
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")
    present = [c for c in config.REQUIRED_COLS if c in daily.columns]

    # 2) missing values
    missing_values = {}
    for c in present:
        n = int(daily[c].isna().sum())
        if n:
            missing_values[c] = {"count": n, "percent": round(100 * n / max(n_rows, 1), 2)}
    if missing_values:
        warnings.append(f"missing values in {len(missing_values)} column(s): {sorted(missing_values)}")
    empty_cols = [c for c in present if n_rows and daily[c].isna().all()]
    if empty_cols:
        errors.append(f"columns with no values at all: {empty_cols}")

    non_numeric = []
    for c in config.BASE_COLS:
        if c in daily.columns:
            coerced = pd.to_numeric(daily[c], errors="coerce")
            if int((coerced.isna() & daily[c].notna()).sum()):
                non_numeric.append(c)
    if non_numeric:
        errors.append(f"non-numeric values in numeric columns: {non_numeric}")

    # 3) duplicates (date is the record ID) and date parsing
    duplicate_dates = 0
    date_from = date_to = None
    if "date" in daily.columns:
        dates = pd.to_datetime(daily["date"], errors="coerce")
        n_bad_dates = int(dates.isna().sum())
        if n_bad_dates:
            errors.append(f"{n_bad_dates} row(s) have a missing or unparseable date")
        valid = dates.dropna()
        duplicate_dates = int(valid.duplicated().sum())
        if duplicate_dates:
            warnings.append(f"{duplicate_dates} duplicate date(s); duplicates are ignored when building features")
        if len(valid):
            date_from, date_to = str(valid.min().date()), str(valid.max().date())
    duplicate_rows = int(daily.duplicated().sum()) if n_rows else 0

    # 4) size and target distribution (only meaningful if the structure above is sound)
    usable_rows = None
    target_distribution = None
    if not errors:
        try:
            ds = build_dataset(daily)
        except Exception as e:  # pragma: no cover - defensive
            errors.append(f"could not build the labelled dataset: {e}")
        else:
            usable_rows = int(len(ds))
            counts = ds["target"].value_counts().reindex([0, 1, 2], fill_value=0)
            target_distribution = {
                config.CLASSES[i]: {"count": int(counts[i]),
                                    "percent": round(100 * int(counts[i]) / max(usable_rows, 1), 1)}
                for i in range(3)}
            if usable_rows < config.MIN_ROWS:
                errors.append(f"only {usable_rows} usable labelled rows (need at least {config.MIN_ROWS})")

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "n_columns": int(daily.shape[1]),
        "date_from": date_from,
        "date_to": date_to,
        "usable_labelled_rows": usable_rows,
        "required_columns": config.REQUIRED_COLS,
        "missing_columns": missing_cols,
        "missing_values": missing_values,
        "duplicate_dates": duplicate_dates,
        "duplicate_rows": duplicate_rows,
        "target_distribution": target_distribution,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def print_report(rep: dict) -> None:
    print(f"[validate] rows={rep['n_rows']}  columns={rep['n_columns']}  "
          f"dates={rep['date_from']} -> {rep['date_to']}")
    print(f"[validate] usable labelled rows (after building features/targets): {rep['usable_labelled_rows']}")
    print(f"[validate] missing required columns: {rep['missing_columns'] or 'none'}")
    mv = rep["missing_values"]
    print("[validate] missing values: " + ("none" if not mv else
          ", ".join(f"{k}={v['count']} ({v['percent']}%)" for k, v in mv.items())))
    print(f"[validate] duplicate dates: {rep['duplicate_dates']}   duplicate rows: {rep['duplicate_rows']}")
    td = rep["target_distribution"]
    if td:
        print("[validate] target distribution: " +
              "  ".join(f"{k}={v['count']} ({v['percent']}%)" for k, v in td.items()))
    for w in rep["warnings"]:
        print(f"[validate] WARNING: {w}")
    for e in rep["errors"]:
        print(f"[validate] ERROR: {e}")
    print(f"[validate] result: {'OK' if rep['ok'] else 'FAILED'}")


def run(data_path=config.DATA_PATH, reports_dir=config.REPORTS_DIR) -> dict:
    data_path = Path(data_path)
    if not data_path.exists():
        raise SystemExit(f"[validate] data file not found: {data_path}  (run: python -m src.ingest)")
    rep = validate(pd.read_csv(data_path))
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_validation.json").write_text(json.dumps(rep, indent=2))
    print_report(rep)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the incoming data batch (Task A)")
    p.add_argument("--data", type=Path, default=config.DATA_PATH)
    a = p.parse_args()
    rep = run(a.data)
    if not rep["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
