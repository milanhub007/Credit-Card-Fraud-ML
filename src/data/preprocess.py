import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> pd.DataFrame:
    df = df.copy()
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    df = df.drop(columns=["Time"], errors="ignore")

    # Fill any missing values with column median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df[numeric_cols] = df[numeric_cols].astype("float64")

    # Scale amount column
    scaler = StandardScaler()

    if "Amount" in df.columns:
        df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Convert numeric columns
    df[numeric_cols] = df[numeric_cols].astype("float64")

    # Save scaler artifact
    os.makedirs(artifacts_dir, exist_ok=True)

    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print(
        f"[preprocess] Shape: {df.shape} | "
        f"Fraud rate: {df['Class'].mean():.4%}"
    )

    return df
