import os
import sys
import time
import argparse
import joblib
import warnings
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from src.data.load_data import load_data
from src.validate.validate_data import validate_data
from src.data.preprocess import preprocess
from src.train.tune import tune_model
from src.serving.export_model import export_latest_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Needed so src.* imports resolve when this script is run directly from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = os.path.join(project_root, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)
    serving_dir = os.path.join(project_root, "src", "serving")
    os.makedirs(serving_dir, exist_ok=True)
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # MLflow requires forward slashes in the URI even on Windows
    mlruns_uri = "file:///" + mlruns_path.replace("\\", "/")
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run():

        mlflow.log_param("model", "RandomForest")

        df = load_data()
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        is_valid, failed, cleaned_data = validate_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

        df = preprocess(df, artifacts_dir)

        target = "Class"
        X = df.drop(columns=[target])
        y = df[target]

        feature_cols = list(X.columns)

        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }

        preprocessing_path = os.path.join(artifacts_dir, "preprocessing.pkl")
        joblib.dump(preprocessing_artifact, preprocessing_path)
        mlflow.log_artifact(preprocessing_path)

        # stratify=y preserves the fraud/non-fraud ratio in both splits (dataset is heavily imbalanced)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        # SMOTE applied only to training data to avoid leaking synthetic samples into evaluation
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        print("Running Optuna tuning...")
        best_params = tune_model(X_res, y_res)
        mlflow.log_params(best_params)

        model = RandomForestClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1  # use all available CPU cores
        )

        print("Training model...")
        start = time.time()
        model.fit(X_res, y_res)
        train_time = time.time() - start
        mlflow.log_metric("train_time", train_time)

        preds = model.predict(X_test)

        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model", input_example=X_test.head(3)
        )

        model_path = os.path.join(artifacts_dir, "model.pkl")
        joblib.dump(model, model_path)
        print("Model logged to MLflow")

        export_latest_model()

        print("\nPipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
