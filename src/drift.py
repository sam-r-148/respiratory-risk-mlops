"""Data-drift check: compare recent feature distributions with the training reference
saved next to the champion model (KS test + Population Stability Index).

Usage:  python -m src.drift
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import config
from src.features import build_dataset

SKIP = {"month_sin", "month_cos"}  # seasonal by design, would always "drift"


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def run(data_path=config.DATA_PATH, model_dir=config.MODEL_DIR, reports_dir=config.REPORTS_DIR,
        window_days: int = 45) -> dict:
    ref_path = model_dir / "reference_features.csv"
    if not ref_path.exists():
        print("[drift] no reference data yet (no champion trained) - skipping")
        return {"skipped": True}
    ref = pd.read_csv(ref_path, parse_dates=["date"], index_col="date")
    cur = build_dataset(pd.read_csv(data_path)).tail(window_days)

    # Compare like with like: pollution is strongly seasonal, so use reference days from the
    # same time of year (+-30 days around the window's centre) in earlier data.
    # Falls back to the whole reference if there are too few same-season days (first year).
    ref = ref[ref.index < cur.index.min()]
    centre = cur.index[len(cur) // 2].dayofyear
    gap = np.abs(ref.index.dayofyear.to_numpy() - centre)
    gap = np.minimum(gap, 365 - gap)
    same_season = ref[gap <= 30]
    seasonal = len(same_season) >= 20
    if seasonal:
        ref = same_season

    features = {}
    for f in config.FEATURES:
        if f in SKIP:
            continue
        p = float(ks_2samp(ref[f], cur[f]).pvalue)
        score = psi(ref[f].to_numpy(), cur[f].to_numpy())
        features[f] = {"ks_pvalue": round(p, 4), "psi": round(score, 3),
                       "drifted": bool(p < 0.01 and score > 0.2)}
    share = sum(v["drifted"] for v in features.values()) / len(features)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "window_days": window_days, "seasonal_reference": seasonal,
              "reference_rows": int(len(ref)), "share_drifted": round(share, 3),
              "dataset_drift": share >= 0.4, "features": features}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "drift.json").write_text(json.dumps(report, indent=2))
    drifted = [k for k, v in features.items() if v["drifted"]]
    print(f"[drift] {share:.0%} of features drifted over last {window_days} days "
          f"({', '.join(drifted) or 'none'}); dataset_drift={report['dataset_drift']}")
    return report


if __name__ == "__main__":
    run()
