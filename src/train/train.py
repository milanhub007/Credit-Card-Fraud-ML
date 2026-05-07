import mlflow
import pandas as pd
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, f1_score, precision_score


def train_model(data: pd.DataFrame, target_col: str):
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # best_params = tune_model(X_train, y_train)
    best_params = {
        'n_estimators': 87,
        'max_depth': 7,
        'min_samples_split': 3,
        'min_samples_leaf': 3,
        'max_features': 'log2',
        'bootstrap': False,
        'random_state': 42
    }

    # Smote for imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(**best_params, n_jobs=-1)
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run():
        # Train model
        model.fit(X_res, y_res)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        rec = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)

        mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("Precision", precision)

        mlflow.sklearn.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(data, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Precision{precision:.4f}, Recall{rec:.4f}, F1{f1:.4f}")

    return model, X_test, y_test
