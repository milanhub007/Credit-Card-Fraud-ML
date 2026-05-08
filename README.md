# Credit Card Fraud Detection

An end-to-end machine learning pipeline that detects fraudulent credit card transactions using a Random Forest classifier, served via a FastAPI REST API and containerised with Docker. CI/CD is handled through GitHub Actions which automatically builds and pushes the Docker image to Docker Hub on every push to `main`.

## Overview

- **Data validation** with Great Expectations before every training run
- **Hyperparameter tuning** with Optuna
- **Class imbalance** handled with SMOTE
- **Experiment tracking** with MLflow
- **REST API** served with FastAPI + Uvicorn
- **Containerised** with Docker
- **CI/CD** with GitHub Actions → Docker Hub

## Project Structure

```
├── .github/
│   └── workflows/
│       └── docker.yml           # CI/CD — builds and pushes image to Docker Hub on push to main
├── data/
│   └── creditcard.csv           # Dataset (284,807 transactions, ~150MB)
├── notebooks/
│   └── main.ipynb               # Jupyter notebook
├── scripts/
│   └── run_pipeline.py          # Main entry point — runs full training pipeline
├── src/
│   ├── app/
│   │   ├── app.py               # FastAPI application
│   │   └── test_api.py          # API tests
│   ├── data/
│   │   ├── load_data.py         # CSV loading with column validation
│   │   └── preprocess.py        # Deduplication, scaling, saves scaler.pkl
│   ├── serving/
│   │   ├── inference.py         # Loads artifacts at startup, runs predictions
│   │   ├── export_model.py      # Exports best MLflow run to src/serving/
│   │   ├── model.pkl            # Trained RandomForest model
│   │   ├── preprocessing.pkl    # Feature column list
│   │   ├── scaler.pkl           # Fitted StandardScaler for Amount
│   │   └── mlflow_metadata/     # Metadata from the best MLflow run
│   ├── train/
│   │   ├── train.py             # Standalone training with fixed params
│   │   ├── tune.py              # Optuna hyperparameter search
│   │   └── evaluate.py          # Confusion matrix and classification report
│   └── validate/
│       └── validate_data.py     # Great Expectations data quality checks
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── CLAUDE.md
```

## Dataset

The dataset contains 284,807 credit card transactions with 492 frauds (~0.17% fraud rate).

- **V1–V28** — PCA-transformed features (real feature names are anonymised)
- **Amount** — transaction amount
- **Class** — target label (0 = legitimate, 1 = fraud)

Download `creditcard.csv` from Kaggle:
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place it at: `data/creditcard.csv`

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/milanhub007/Credit-Card-Fraud-ML.git
cd Credit-Card-Fraud-ML
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Pipeline

Runs data validation, Optuna tuning, training, evaluation, and exports the model artifacts to `src/serving/`:

```bash
python scripts/run_pipeline.py
```

View MLflow experiment results:
```bash
mlflow ui
```

## Running the API Locally

```bash
uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000/docs` (Swagger UI).

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict fraud on a transaction |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.53, "V4": 1.38,
       "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
       "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
       "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
       "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
       "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
```

**Example response:**
```json
{
  "fraud": 0,
  "fraud_probability": 0.03
}
```

## Running with Docker

**Pull from Docker Hub:**
```bash
docker pull milanhub007/credit-card-fraud-ml:latest
docker run -p 8000:8000 milanhub007/credit-card-fraud-ml:latest
```

**Or build locally:**
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## CI/CD

Every push to `main` triggers a GitHub Actions workflow that:
1. Builds the Docker image
2. Pushes it to Docker Hub as `milanhub007/credit-card-fraud-ml:latest`

No manual `docker build` or `docker push` needed.

## Model Performance

| Metric | Score |
|--------|-------|
| F1 Score | 0.8182 |
| Precision | 0.8889 |
| Recall | 0.7579 |

Recall is prioritised — missing a fraudulent transaction is more costly than a false alarm.

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn, imbalanced-learn, Optuna |
| Tracking | MLflow |
| Validation | Great Expectations |
| API | FastAPI, Uvicorn, Pydantic |
| Container | Docker |
| CI/CD | GitHub Actions, Docker Hub |
