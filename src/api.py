"""FastAPI service.   Run:  uvicorn src.api:app --reload   then open http://127.0.0.1:8000/docs"""
import os
from datetime import date as Date
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.predictor import Predictor

app = FastAPI(title="Respiratory Risk Alert API",
              description="Forecasts tomorrow's respiratory risk level (Low / Moderate / High) "
                          "from today's air quality and weather. Educational proxy, not medical advice.",
              version="1.0.0")


@lru_cache(maxsize=1)
def get_predictor() -> Optional[Predictor]:
    model_dir = os.getenv("MODEL_DIR", str(config.MODEL_DIR))
    try:
        return Predictor(model_dir)
    except FileNotFoundError:
        return None


class Conditions(BaseModel):
    """Today's daily conditions."""
    pm2_5_mean: float = Field(..., ge=0, le=2000, description="PM2.5 daily mean, ug/m3")
    pm10_mean: float = Field(..., ge=0, le=3000, description="PM10 daily mean, ug/m3")
    o3_max: float = Field(..., ge=0, le=1000, description="Ozone daily max, ug/m3")
    no2_mean: float = Field(..., ge=0, le=1000, description="NO2 daily mean, ug/m3")
    temp_mean: float = Field(..., ge=-60, le=60, description="deg C")
    temp_min: float = Field(..., ge=-60, le=60)
    temp_max: float = Field(..., ge=-60, le=60)
    humidity_mean: float = Field(..., ge=0, le=100, description="%")
    wind_mean: float = Field(..., ge=0, le=200, description="km/h")
    precip_sum: float = Field(..., ge=0, le=2000, description="mm")
    pm2_5_lag1: Optional[float] = Field(None, ge=0, le=2000, description="yesterday's PM2.5 (optional)")
    pm2_5_roll3: Optional[float] = Field(None, ge=0, le=2000, description="3-day mean PM2.5 (optional)")
    pm2_5_roll7: Optional[float] = Field(None, ge=0, le=2000, description="7-day mean PM2.5 (optional)")
    date: Optional[Date] = Field(None, description="defaults to today; used for seasonality")

    model_config = {"json_schema_extra": {"example": {
        "pm2_5_mean": 135.0, "pm10_mean": 240.0, "o3_max": 55.0, "no2_mean": 62.0,
        "temp_mean": 14.0, "temp_min": 8.5, "temp_max": 22.0, "humidity_mean": 72.0,
        "wind_mean": 4.0, "precip_sum": 0.0}}}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_predictor() is not None}


@app.get("/model-info")
def model_info():
    p = get_predictor()
    if p is None:
        raise HTTPException(503, "No trained model found. Run the training pipeline first.")
    return p.info()


@app.post("/predict")
def predict(c: Conditions):
    p = get_predictor()
    if p is None:
        raise HTTPException(503, "No trained model found. Run the training pipeline first.")
    record = c.model_dump()
    if record["date"] is not None:
        record["date"] = record["date"].isoformat()
    return p.predict(record)
