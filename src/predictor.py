"""Framework-free prediction logic (kept separate from FastAPI so it is easy to test)."""
import json
from pathlib import Path

import joblib

from src.features import row_from_record

ADVICE = {
    "Low": "Air conditions look good. Normal outdoor activity is fine for most people.",
    "Moderate": "Sensitive groups (asthma, COPD, children, elderly) should limit prolonged outdoor exertion.",
    "High": "High risk day. Sensitive groups should stay indoors, keep reliever medication handy "
            "and consider a mask (N95) outdoors.",
}
DISCLAIMER = ("Educational project. This is a rule-based risk proxy forecast from air quality and "
              "weather, not a medical diagnosis.")


class Predictor:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "model.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.classes = self.meta["classes"]

    def info(self) -> dict:
        m = self.meta
        return {"model": m["selected_model"], "trained_at": m["run_at"], "train_end": m["train_end"],
                "holdout_f1_macro": m["challenger"]["f1_macro"], "city": m.get("city"),
                "classes": self.classes}

    def predict(self, record: dict) -> dict:
        row = row_from_record(record)
        proba = self.model.predict_proba(row)[0]
        labels = [self.classes[int(c)] for c in self.model.classes_]
        best = int(proba.argmax())
        risk = labels[best]
        return {
            "risk": risk,
            "risk_level": int(self.model.classes_[best]),
            "probabilities": {l: round(float(p), 4) for l, p in zip(labels, proba)},
            "advice": ADVICE[risk],
            "model_trained_at": self.meta["run_at"],
            "disclaimer": DISCLAIMER,
        }
