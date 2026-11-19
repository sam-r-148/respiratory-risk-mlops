"""Pull hourly air quality + weather from Open-Meteo (free, no API key), aggregate to
daily values and append to data/daily.csv.

Usage:
    python -m src.ingest                       # incremental update (first run: ~1 year of history)
    python -m src.ingest --start-date 2023-06-01
    python -m src.ingest --synthetic --days 730   # offline demo/test data (NOT real)
"""
import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import config

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WX_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"  # supports past_days (max 92)

AQ_VARS = ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "us_aqi"]
WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]

MAX_PAST_DAYS = 90        # stay under the API limit of 92
MIN_HOURS_PER_DAY = 20    # drop incomplete days


def today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=config.TIMEZONE).normalize().tz_localize(None)


def _get_hourly(url: str, params: dict, retries: int = 3) -> pd.DataFrame:
    base = {"latitude": config.LAT, "longitude": config.LON, "timezone": config.TIMEZONE}
    for attempt in range(retries):
        try:
            r = requests.get(url, params={**base, **params}, timeout=60)
            r.raise_for_status()
            js = r.json()
            if "hourly" not in js:
                raise ValueError(js.get("reason", "response has no 'hourly' block"))
            df = pd.DataFrame(js["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_air_quality(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days_back = (today() - start).days
    params = {"hourly": ",".join(AQ_VARS)}
    if days_back <= MAX_PAST_DAYS:
        params.update(past_days=max(days_back, 1), forecast_days=1)
    else:
        params.update(start_date=start.date().isoformat(), end_date=end.date().isoformat())
    return _get_hourly(AQ_URL, params)


def fetch_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    archive_end = min(end, today() - timedelta(days=7))  # archive lags a few days
    if start <= archive_end:
        parts.append(_get_hourly(WX_ARCHIVE_URL, {
            "hourly": ",".join(WX_VARS),
            "start_date": start.date().isoformat(),
            "end_date": archive_end.date().isoformat(),
        }))
    recent_start = max(start, today() - timedelta(days=MAX_PAST_DAYS))
    if recent_start <= end:
        parts.append(_get_hourly(WX_FORECAST_URL, {
            "hourly": ",".join(WX_VARS),
            "past_days": max((today() - recent_start).days, 1),
            "forecast_days": 1,
        }))
    df = pd.concat(parts).drop_duplicates("time", keep="last").sort_values("time")
    return df.reset_index(drop=True)


def to_daily(aq: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    aq = aq.assign(date=aq["time"].dt.floor("D"))
    wx = wx.assign(date=wx["time"].dt.floor("D"))
    aq_d = aq.groupby("date").agg(
        pm2_5_mean=("pm2_5", "mean"), pm10_mean=("pm10", "mean"),
        o3_max=("ozone", "max"), no2_mean=("nitrogen_dioxide", "mean"),
        aqi_mean=("us_aqi", "mean"), aq_hours=("pm2_5", "count"),
    )
    wx_d = wx.groupby("date").agg(
        temp_mean=("temperature_2m", "mean"), temp_min=("temperature_2m", "min"),
        temp_max=("temperature_2m", "max"), humidity_mean=("relative_humidity_2m", "mean"),
        wind_mean=("wind_speed_10m", "mean"), precip_sum=("precipitation", "sum"),
        wx_hours=("temperature_2m", "count"),
    )
    daily = aq_d.join(wx_d, how="inner")
    daily = daily[(daily["aq_hours"] >= MIN_HOURS_PER_DAY) & (daily["wx_hours"] >= MIN_HOURS_PER_DAY)]
    daily = daily[daily.index < today()]  # today is incomplete
    return daily.drop(columns=["aq_hours", "wx_hours"]).reset_index()


def merge_and_save(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = new.assign(date=pd.to_datetime(new["date"]))
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        new = pd.concat([old, new])
    out = new.drop_duplicates("date", keep="last").sort_values("date")
    out.assign(date=out["date"].dt.strftime("%Y-%m-%d")).to_csv(path, index=False)
    return out


def make_synthetic(days: int, seed: int = 7) -> pd.DataFrame:
    """Plausible fake North-India-style data. For tests and offline demos only."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=today() - timedelta(days=1), periods=days)
    doy = dates.dayofyear.to_numpy()
    season = np.cos(2 * np.pi * (doy - 10) / 365)  # +1 in winter, -1 in summer
    x = np.zeros(days)
    for i in range(1, days):
        x[i] = 0.75 * x[i - 1] + rng.normal(0, 0.35)
    pm25 = np.clip((65 + 50 * season) * np.exp(x), 5, 500)
    temp_mean = 25 - 12 * season + rng.normal(0, 2, days)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "pm2_5_mean": pm25,
        "pm10_mean": pm25 * 1.8 + rng.normal(0, 10, days),
        "o3_max": np.clip(70 - 35 * season + rng.normal(0, 18, days), 10, 250),
        "no2_mean": np.clip(35 + 25 * season + 12 * x + rng.normal(0, 6, days), 3, 200),
        "aqi_mean": np.clip(pm25 * 1.5, 10, 500),
        "temp_mean": temp_mean,
        "temp_min": temp_mean - 5 - rng.random(days) * 2,
        "temp_max": temp_mean + 6 + rng.random(days) * 2,
        "humidity_mean": np.clip(55 + 12 * season + rng.normal(0, 10, days), 15, 98),
        "wind_mean": np.clip(8 - 2.5 * x + rng.normal(0, 1.5, days), 1, 25),
        "precip_sum": rng.exponential(4, days) * (rng.random(days) < 0.15),
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, help="how many past days to fetch")
    p.add_argument("--start-date", help="YYYY-MM-DD, fetch from this date (uses date-range API)")
    p.add_argument("--synthetic", action="store_true", help="generate fake data instead of calling APIs")
    p.add_argument("--out", type=Path, default=config.DATA_PATH)
    a = p.parse_args()

    if a.synthetic:
        df = merge_and_save(make_synthetic(a.days or 730), a.out)
        print(f"[ingest] wrote {len(df)} SYNTHETIC rows to {a.out}")
        return

    end = today() - timedelta(days=1)
    if a.start_date:
        start = pd.Timestamp(a.start_date)
    elif a.days:
        start = end - timedelta(days=a.days - 1)
    elif a.out.exists():
        last = pd.read_csv(a.out, parse_dates=["date"])["date"].max()
        start = last - timedelta(days=2)  # overlap a little to refresh late-arriving values
    else:
        start = end - timedelta(days=364)  # first run: one year

    try:
        daily = to_daily(fetch_air_quality(start, end), fetch_weather(start, end))
    except Exception as e:
        if start < today() - timedelta(days=MAX_PAST_DAYS) and not a.start_date:
            print(f"[ingest] long-range fetch failed ({e}); falling back to last {MAX_PAST_DAYS} days")
            start = today() - timedelta(days=MAX_PAST_DAYS)
            daily = to_daily(fetch_air_quality(start, end), fetch_weather(start, end))
        else:
            raise
    df = merge_and_save(daily, a.out)
    print(f"[ingest] fetched {len(daily)} days; file now has {len(df)} days "
          f"({df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
