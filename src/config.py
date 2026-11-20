"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "daily.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models" / "champion"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))

# City to monitor (defaults to Gurugram). Change via env vars to monitor another city.
CITY = os.getenv("CITY", "Gurugram")
LAT = float(os.getenv("LAT", "28.4595"))
LON = float(os.getenv("LON", "77.0266"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")

CLASSES = ["Low", "Moderate", "High"]

# Daily columns produced by ingest.py
BASE_COLS = [
    "pm2_5_mean", "pm10_mean", "o3_max", "no2_mean",
    "temp_mean", "temp_min", "temp_max",
    "humidity_mean", "wind_mean", "precip_sum",
]
# Columns the model actually sees (today's conditions -> tomorrow's risk)
FEATURES = BASE_COLS + [
    "score_today", "pm2_5_lag1", "pm2_5_roll3", "pm2_5_roll7", "month_sin", "month_cos",
]

HOLDOUT_DAYS = int(os.getenv("HOLDOUT_DAYS", "60"))      # most recent days used for gating
MIN_ROWS = int(os.getenv("MIN_ROWS", "100"))             # refuse to train on less data
MIN_IMPROVEMENT = float(os.getenv("MIN_IMPROVEMENT", "0.0"))  # challenger must beat champion by this much (macro-F1)
RANDOM_STATE = 42

# Columns every incoming data batch must contain (checked by src/validate.py)
REQUIRED_COLS = ["date"] + BASE_COLS

# Quality gate. The gate metric is macro-F1 (see README / WORKFLOW.md for the justification).
GATE_METRIC = "f1_macro"
# If false, an eligible candidate is only *marked* eligible and the champion is left untouched
# until a human promotes it. Default true = automatic promotion (always recorded in the decision log).
AUTO_PROMOTE = os.getenv("AUTO_PROMOTE", "true").strip().lower() not in ("0", "false", "no")


def candidate_dir_for(model_dir) -> Path:
    """The candidate model lives next to the champion (models/candidate next to models/champion)."""
    return Path(model_dir).parent / "candidate"
