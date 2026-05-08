# CLAUDE.md

## Project Overview

Credit Card Fraud Detection — end-to-end ML pipeline with a FastAPI serving layer, MLflow experiment tracking, Docker containerisation, and GitHub Actions CI/CD that pushes to Docker Hub on every push to `main`.

The dataset (V1–V28) contains PCA-transformed features — the real feature names are anonymised by the original data provider.

## Key Commands

**Run full pipeline (validate → tune → train → export):**
```bash
python scripts/run_pipeline.py
```

**Start API locally:**
```bash
uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```

**View MLflow experiments:**
```bash
mlflow ui
```

**Build and run Docker locally:**
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

**Pull from Docker Hub:**
```bash
docker pull milanhub007/credit-card-fraud-ml:latest
docker run -p 8000:8000 milanhub007/credit-card-fraud-ml:latest
```

## Architecture

```
run_pipeline.py
    → load_data.py          # loads data/creditcard.csv, validates expected columns
    → validate_data.py      # Great Expectations checks (nulls, ranges, fraud rate sanity)
    → preprocess.py         # drops Time, scales Amount, saves scaler.pkl to artifacts/
    → tune.py               # Optuna search (5 trials) over RandomForest params
    → RandomForestClassifier training on SMOTE-resampled data
    → MLflow logging (params, metrics, model artifact)
    → export_model.py       # copies best MLflow run artifacts to src/serving/
```

The API (`src/app/app.py`) calls `src/serving/inference.py` which loads three artifacts at import time:
- `src/serving/model.pkl` — trained RandomForest
- `src/serving/preprocessing.pkl` — feature column list and target name
- `src/serving/scaler.pkl` — StandardScaler fitted on Amount

**These artifacts must exist before starting the API.** Run `run_pipeline.py` first to generate them, or pull the Docker image which already contains them.

## CI/CD

`.github/workflows/docker.yml` triggers on every push to `main`:
1. Checks out code
2. Logs in to Docker Hub using `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` GitHub secrets
3. Builds the Docker image
4. Pushes to `milanhub007/credit-card-fraud-ml:latest`

GitHub Actions cache (`cache-from/cache-to: type=gha`) is used to speed up repeat builds.

## Important Decisions

- **SMOTE applied only to training split** — applying before splitting leaks synthetic samples into evaluation and inflates scores.
- **`Time` column dropped** — accepted in the API schema for input compatibility but removed in `inference.py` before prediction. It was also excluded during training.
- **`Amount` scaled at inference using the training scaler** — must use the same scaler fitted during training, not a new one, to stay consistent with the training distribution.
- **Artifacts loaded at module level in `inference.py`** — intentional, loads once on startup rather than on every request.
- **F1 as the Optuna objective** — accuracy is misleading on this dataset (~0.17% fraud rate); F1 balances precision/recall appropriately.
- **`pywin32` filtered during Docker build** — `pywin32` is a Windows-only package in `requirements.txt`. The Dockerfile filters it out with `grep -v "^pywin32"` before running `pip install`.
- **Non-root user in Docker** — container runs as `appuser` (not root) for security.

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `scripts/run_pipeline.py` | Orchestrates the full training pipeline |
| `src/data/load_data.py` | Loads CSV, validates expected columns exist |
| `src/data/preprocess.py` | Deduplication, median imputation, Amount scaling, saves scaler |
| `src/validate/validate_data.py` | Great Expectations suite — nulls, value ranges, fraud rate sanity |
| `src/train/tune.py` | Optuna hyperparameter search over RandomForest params |
| `src/train/train.py` | Standalone training with fixed params (separate from pipeline) |
| `src/train/evaluate.py` | Confusion matrix and classification report |
| `src/serving/export_model.py` | Pulls best MLflow run and copies artifacts to `src/serving/` |
| `src/serving/inference.py` | Loads artifacts at import time, transforms input, returns prediction + probability |
| `src/app/app.py` | FastAPI routes — GET `/` health check, POST `/predict` |
| `.github/workflows/docker.yml` | GitHub Actions CI/CD — builds and pushes Docker image to Docker Hub |
| `Dockerfile` | Python 3.9-slim image, filters pywin32, runs as non-root, healthcheck on port 8000 |
| `.dockerignore` | Excludes venv, notebooks, mlruns, data, test files from Docker build context |

## Model Performance (latest run)

| Metric | Score |
|--------|-------|
| F1 Score | 0.8182 |
| Precision | 0.8889 |
| Recall | 0.7579 |

Metrics are sourced from `src/serving/mlflow_metadata/metrics/`.

## Dependencies

Full install (training + serving): `requirements.txt`

Runtime only (what the Docker image needs): `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pandas`, `numpy`, `joblib`

Note: `pywin32` in `requirements.txt` is Windows-only and is filtered out during Docker build.
