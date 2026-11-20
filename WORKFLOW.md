# Model monitoring and retraining workflow (assessment Tasks A-E)

The deployed model forecasts tomorrow's respiratory risk (Low / Moderate / High). New data arrives every
day, so this workflow checks the incoming data, measures how the current model is doing on it, retrains a
candidate, and decides - with a written record - whether the candidate replaces the current champion.

Run everything with one command:

```bash
python -m src.pipeline            # uses data/daily.csv as it is
python -m src.pipeline --ingest   # first pulls the newest batch from Open-Meteo
```

The same steps run automatically every day in GitHub Actions (`.github/workflows/retrain.yml`).

```
new data -> A validate -> B evaluate champion -> C retrain candidate -> D quality gate -> promote / retain
              |              |                     |                      |
   data_validation.json  evaluation_report.json  models/candidate/   decision_log.jsonl
```

## Task A - Prepare new data (`src/validate.py`)

* Loads the data batch (`data/daily.csv`, extended daily by `src/ingest.py`).
* Checks that all required columns exist (`date` plus the 10 weather / air-quality columns).
* Counts missing values per column, and checks non-numeric values and unparseable dates.
* Checks duplicate records - the record ID is the calendar date - and fully duplicated rows.
* Prints the dataset size, date range and the **target distribution** (Low / Moderate / High).
* Errors (missing columns, empty data, too few rows) stop the pipeline. Missing values and duplicates are
  warnings, because the feature code ignores duplicate dates and interpolates gaps of up to 2 days.
* Output: `reports/data_validation.json`.

## Task B - Evaluate the existing model (`src/evaluate.py`)

* Loads the champion from `models/champion/`.
* "New labelled data" = every labelled day **after the champion's training end date**. The champion has
  never seen those days, so the score shows whether it is still reliable.
* Reports accuracy, precision, recall and F1 (macro, weighted and per class) and the confusion matrix.
* Also scores the naive baseline ("tomorrow = today") on the same days for context.
* Output: `reports/evaluation_report.json`.

## Task C - Retrain (`src/train.py`)

* `data/daily.csv` already holds the earlier training data **plus** every new batch, so the candidate is
  trained on existing + new data combined, with the same feature pipeline (`src/features.py`) as the champion
  and the API.
* Two model types are compared with time-ordered cross-validation; the better one is fitted.
* The candidate is **always saved separately** in `models/candidate/` (model, metadata, reference features),
  never directly into the champion folder.
* Each run is tracked in MLflow (parameters, metrics, the candidate model artifact, the reports).

## Task D - Quality gate (`src/train.py`)

* **Metric: macro-F1** on the most recent 60 labelled days. The candidate, the champion and the naive
  baseline are scored on that same window, which neither model was trained on.
* **Rule:** the candidate is *eligible for promotion* if there is no champion yet, or its macro-F1 is
  strictly higher than the champion's by at least `MIN_IMPROVEMENT` (default 0). Otherwise the champion is
  retained. A tie keeps the champion.
* **Never overwritten silently:** the decision (eligible / retained, both scores, the reason) is appended to
  `reports/decision_log.jsonl` *before* the champion folder is touched. A test enforces this order.
* `AUTO_PROMOTE=false` makes an eligible candidate only *marked* eligible; the champion stays until a
  human promotes it. The default (`true`) promotes automatically, with the decision recorded.

### Why macro-F1?

* The classes are very imbalanced. In our real data, High is about 63% of days, Moderate about 35% and Low
  under 2%. Accuracy would reward a model that mostly predicts High.
* Macro-F1 averages the F1 of each class equally, so missing a class is penalised.
* F1 balances precision (false alarms) and recall (missed high-risk days), and both matter for a health alert.
* Limitations: a 60-day holdout is small and noisy (the recent holdout has no Low days at all, so its
  macro-F1 averages over Moderate and High), and the margin `MIN_IMPROVEMENT` can be raised to avoid
  promoting on noise.

## Task E - What to demonstrate

| Item | Where |
|---|---|
| Successful execution | `python -m src.pipeline`, and the green *Daily retrain* run in the GitHub Actions tab |
| Evaluation report | `reports/evaluation_report.json` |
| Candidate model artifact | `models/candidate/model.joblib` (local) and the `retrain-N` artifact of each Actions run |
| Quality-gate decision | the last line of `reports/decision_log.jsonl`, and `decision` in `reports/latest_run.json` |
| Data validation | `reports/data_validation.json` |
| MLflow run | `mlflow ui --backend-store-uri mlruns` (experiment `respiratory-risk`) |

## Files

| File | Role |
|---|---|
| `src/validate.py` | Task A |
| `src/evaluate.py` | Task B |
| `src/train.py` | Tasks C and D |
| `src/pipeline.py` | Task E - runs A to D in order and prints the evidence |
| `reports/decision_log.jsonl` | Append-only history of every gate decision |
