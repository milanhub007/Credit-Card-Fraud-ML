import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "../..", "data", "creditcard.csv")


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Read creditcard.csv and return a raw DataFrame.
    Raises FileNotFoundError if the file is missing.
    """
    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place creditcard.csv inside the data/ folder."
        )

    df = pd.read_csv(filepath)

    # Quick sanity check
    expected = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    print(f"[load_data] {len(df):,} rows loaded  |  Fraud rate: {df['Class'].mean():.4%}")
    return df





