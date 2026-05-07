import os
import joblib
import pandas as pd

SERVING_DIR = os.path.dirname(__file__)

# Artifacts are loaded once at import time, so they are not reloaded on every request
model = joblib.load(
    os.path.join(SERVING_DIR, "model.pkl")
)

preprocessing = joblib.load(
    os.path.join(SERVING_DIR, "preprocessing.pkl")
)

FEATURE_COLS = preprocessing["feature_columns"]

scaler = joblib.load(
    os.path.join(SERVING_DIR, "scaler.pkl")
)


def transform_input(df):

    df = df.copy()

    # Time is not predictive after PCA transformation and was excluded during training
    df = df.drop(
        columns=["Time"],
        errors="ignore"
    )

    # Amount must be scaled to match the distribution the model was trained on
    if "Amount" in df.columns:
        df["Amount"] = scaler.transform(
            df[["Amount"]]
        )

    # reindex ensures column order and names exactly match training data; missing columns get 0
    df = df.reindex(
        columns=FEATURE_COLS,
        fill_value=0
    )

    return df


def predict(input_dict):

    df = pd.DataFrame([input_dict])

    df = transform_input(df)

    prediction = model.predict(df)[0]

    # [0][1] selects the probability of the positive class (fraud = 1)
    probability = model.predict_proba(df)[0][1]

    return {
        "fraud": int(prediction),
        "fraud_probability": float(probability)
    }
