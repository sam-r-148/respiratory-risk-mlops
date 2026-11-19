# Respiratory Risk Alert: an end-to-end MLOps project

Forecasts **tomorrow's respiratory risk level (Low / Moderate / High)** for a city from today's
air quality and weather, with a pipeline that **pulls live data daily, retrains automatically,
promotes a new model only if it is better, and redeploys it as a Docker image**.

> **Disclaimer:** educational project. The target is a **rule-based risk proxy** built from air-quality
> bands (see `src/features.py`), *not* a clinical outcome or medical diagnosis.

Default city: Gurugram (change `CITY`, `LAT`, `LON` env vars to monitor another city).

## Architecture

```
 Open-Meteo Air Quality API ─┐
                             ├─► ingest.py ─► data/daily.csv ─► features.py ─► train.py ──► gate ──► models/champion/
 Open-Meteo Weather API ─────┘   (daily agg)   (versioned in git)  (t -> t+1)   (MLflow)   (better     model.joblib
                                                                                           than       metadata.json
                      GitHub Actions cron (daily) triggers everything ─────────────────►   champion?)  reference_features.csv
                                                                                              │
                      drift.py (KS + PSI vs training reference) ─► reports/drift.json         ▼
                                                                         promoted? ─► Docker build ─► ghcr.io image ─► FastAPI /predict
```

| Stage | File | What it does |
|---|---|---|
| Ingest | `src/ingest.py` | Hourly PM2.5, PM10, NO2, ozone + temperature, humidity, wind, rain from Open-Meteo (free, **no API key**) → daily table. Incremental. |
| Features / label | `src/features.py` | Features from day *t*; label = risk class of day *t+1*. Includes a no-leakage test. |
| Train + gate | `src/train.py` | Picks RF vs HistGradientBoosting by time-series CV, scores on the last 60 days, **promotes only if macro-F1 beats the current champion**. Logs to MLflow. |
| Drift | `src/drift.py` | KS test + PSI per feature, recent window vs same-season reference. |
| Serve | `src/api.py` | FastAPI: `GET /health`, `GET /model-info`, `POST /predict`. |
| CI | `.github/workflows/ci.yml` | pytest, Docker build, container smoke test. |
| Retrain / CD | `.github/workflows/retrain.yml` | Daily cron: ingest → drift → train/gate → commit data+model → push image if promoted. |

## How the label works (be ready to explain this)

`score = PM2.5 band (0-4) + [ozone > 100] + [NO2 > 80] + [min temp < 8°C]`
→ **Low** (0-1), **Moderate** (2-3), **High** (4+).

The model does **not** just read today's AQI: it *forecasts tomorrow's* score from today's pollution,
yesterday/3-day/7-day PM2.5 history, today's weather and the season. It is compared against a
**persistence baseline** ("tomorrow = today") reported in every run (`persistence_baseline_f1`),
so you can show whether ML adds value over the naive guess.

## Run it locally

Use **Python 3.12** to match CI/Docker (the pinned versions are for 3.12).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) get REAL data (first run backfills ~1 year; needs internet)
python -m src.ingest
#    optionally a longer history for a better model:
#    python -m src.ingest --start-date 2023-01-01
#    (no internet / just want to try it? python -m src.ingest --synthetic  -> FAKE data, demo only)

# 2) train (creates models/champion/ the first time)
python -m src.train

# 3) look at experiments
mlflow ui                          # http://127.0.0.1:5000

# 4) drift report (needs a trained champion)
python -m src.drift

# 5) tests
pytest -q

# 6) serve the API
uvicorn src.api:app --reload       # http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "pm2_5_mean": 135, "pm10_mean": 240, "o3_max": 55, "no2_mean": 62,
  "temp_mean": 14, "temp_min": 8.5, "temp_max": 22,
  "humidity_mean": 72, "wind_mean": 4, "precip_sum": 0 }'
```

### Docker

```bash
docker build -t respiratory-risk .
docker run -p 8000:8000 respiratory-risk
```
(The image copies `models/champion`, so run steps 1-2 above first.)

## Put it on GitHub (this satisfies the "CI pipeline runs successfully" requirement too)

1. Run `ingest` + `train` locally so `data/daily.csv` and `models/champion/` exist.
2. Create a new GitHub repo, then:
   ```bash
   git init && git add . && git commit -m "Respiratory risk MLOps project"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
3. The **CI** workflow runs on push. Check the *Actions* tab for a green run.
4. **Settings → Actions → General → Workflow permissions → "Read and write permissions"** (needed so the
   retrain job can commit data/model back).
5. *Actions → Daily retrain → Run workflow* to trigger it manually once; after that it runs every day.
   Screenshot a green run for your report.
6. When a model is promoted, the image appears under your repo's **Packages** (`ghcr.io/<you>/<repo>`).

## Design decisions & honest limitations

- **Why a gate?** A daily retrain can produce a worse model by chance. The champion/challenger gate
  (strictly better macro-F1 on the same recent holdout) prevents regressions reaching production.
- **Why a time-based split?** Random splits leak the future into training for time series.
- **Proxy label.** Without real hospital/clinic data the target is a rule-based index. The ML
  contribution is *forecasting* it a day ahead. Say this explicitly in your viva.
- **Small holdout is noisy.** 60 days of macro-F1 can swing; `MIN_IMPROVEMENT` (env var) adds a margin.
- **Drift check is a heuristic.** Daily pollution is autocorrelated and seasonal, so KS/PSI can raise
  false alarms; the report is informational and the daily retrain runs regardless.
- **Champion is trained without the last 60 days** (they are the gate's holdout). A natural upgrade is
  refitting on all data after the gate passes.
- **Open-Meteo air quality is modelled (CAMS) data**, not a ground sensor reading.

## Ideas to extend (pick one or two for extra marks)

- Replace `drift.py` with an **Evidently** report, or add Prometheus + Grafana for API metrics.
- Add **DVC** for the dataset and **MLflow Model Registry** stages instead of the file-based champion.
- Multi-city support (loop over a list of coordinates, one model per city).
- A small Streamlit page calling `/predict` with live data.
- Real labels: public hospital/OPD respiratory-admission counts for a city, if you can find them.
