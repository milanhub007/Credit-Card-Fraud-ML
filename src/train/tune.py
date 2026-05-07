import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


def tune_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # SMOTE only on train split — applying it to val would inflate scores with synthetic samples
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
        model.fit(X_res, y_res)
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    # F1 is the right metric here — accuracy would be misleading on the imbalanced fraud dataset
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)

    print("Best params:", study.best_params)
    return study.best_params
