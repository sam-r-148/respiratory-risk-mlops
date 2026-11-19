"""Feature engineering and the respiratory-risk label.

IMPORTANT: the label is a RULE-BASED PROXY, not a clinical outcome. It is built from
tomorrow's pollution and weather using commonly cited air-quality bands. The model
forecasts this risk level from today's conditions.
"""
import numpy as np
import pandas as pd

from src.config import BASE_COLS, FEATURES

SCORE_INPUTS = ["pm2_5_mean", "o3_max", "no2_mean", "temp_min"]


def respiratory_score(df: pd.DataFrame) -> pd.Series:
    """0-7 points: PM2.5 band (0-4) + ozone + NO2 + cold-air trigger (1 each)."""
    pm = df["pm2_5_mean"]
    score = pd.Series(
        np.select([pm < 30, pm < 60, pm < 90, pm < 120], [0, 1, 2, 3], default=4),
        index=df.index, dtype="float",
    )
    score += (df["o3_max"] > 100).astype(int)
    score += (df["no2_mean"] > 80).astype(int)
    score += (df["temp_min"] < 8).astype(int)
    return score.where(df[SCORE_INPUTS].notna().all(axis=1))


def score_to_class(score: pd.Series) -> pd.Series:
    """0 = Low (score<=1), 1 = Moderate (2-3), 2 = High (>=4)."""
    cls = pd.Series(np.select([score <= 1, score <= 3], [0, 1], default=2),
                    index=score.index, dtype="float")
    return cls.where(score.notna())


def _prepare(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df[BASE_COLS] = df[BASE_COLS].interpolate(limit=2, limit_area="inside")  # fill tiny gaps only
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_today"] = respiratory_score(out)
    out["pm2_5_lag1"] = out["pm2_5_mean"].shift(1)
    out["pm2_5_roll3"] = out["pm2_5_mean"].rolling(3).mean()
    out["pm2_5_roll7"] = out["pm2_5_mean"].rolling(7).mean()
    month = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def build_dataset(daily: pd.DataFrame) -> pd.DataFrame:
    """Row for day t: features from day t (and earlier), target = risk class of day t+1.
    Also returns `persistence` = today's own class, used as a naive baseline."""
    df = _prepare(daily)
    feats = _add_features(df)
    today_score = respiratory_score(df)
    feats["target"] = score_to_class(today_score.shift(-1))
    feats["persistence"] = score_to_class(today_score)
    ds = feats.dropna(subset=FEATURES + ["target", "persistence"])
    ds = ds[FEATURES + ["target", "persistence"]].copy()
    ds["target"] = ds["target"].astype(int)
    ds["persistence"] = ds["persistence"].astype(int)
    return ds


def row_from_record(record: dict) -> pd.DataFrame:
    """Build one model-input row from an API request (lags default to today's PM2.5)."""
    r = dict(record)
    date = pd.Timestamp(r.pop("date", None) or pd.Timestamp.now())
    pm = r["pm2_5_mean"]
    r["score_today"] = float(respiratory_score(pd.DataFrame([{k: r[k] for k in SCORE_INPUTS}])).iloc[0])
    for k in ("pm2_5_lag1", "pm2_5_roll3", "pm2_5_roll7"):
        if r.get(k) is None:
            r[k] = pm
    r["month_sin"] = float(np.sin(2 * np.pi * date.month / 12))
    r["month_cos"] = float(np.cos(2 * np.pi * date.month / 12))
    return pd.DataFrame([{f: r[f] for f in FEATURES}])
