import os
import shutil
import joblib
import mlflow


SERVING_DIR = os.path.dirname(__file__)

PROJECT_ROOT = os.path.abspath(
    os.path.join(SERVING_DIR, "../..")
)

MLRUNS_DIR = os.path.join(
    PROJECT_ROOT,
    "mlruns"
)


def export_latest_model():

    mlflow.set_tracking_uri(
        "file:///" + MLRUNS_DIR.replace("\\", "/")
    )

    experiment = mlflow.get_experiment_by_name(
        "fraud_detection"
    )

    experiment_id = experiment.experiment_id

    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    latest_run = runs[0]

    run_id = latest_run.info.run_id

    run_dir = os.path.join(
        MLRUNS_DIR,
        experiment_id,
        run_id
    )

    model_uri = f"runs:/{run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)

    # SAVE MODEL
    joblib.dump(
        model,
        os.path.join(
            SERVING_DIR,
            "model.pkl"
        )
    )

    # COPY ARTIFACTS
    artifacts_src = os.path.join(
        PROJECT_ROOT,
        "artifacts"
    )

    files_to_copy = [
        "preprocessing.pkl",
        "scaler.pkl"
    ]

    for file_name in files_to_copy:

        src = os.path.join(
            artifacts_src,
            file_name
        )

        dst = os.path.join(
            SERVING_DIR,
            file_name
        )

        if os.path.exists(src):
            shutil.copy(src, dst)

    # COPY MLFLOW METADATA
    metadata_dst = os.path.join(
        SERVING_DIR,
        "mlflow_metadata"
    )

    if os.path.exists(metadata_dst):
        shutil.rmtree(metadata_dst)

    os.makedirs(metadata_dst, exist_ok=True)

    items_to_copy = [
        "artifacts",
        "metrics",
        "params",
        "tags",
        "meta.yaml"
    ]

    for item in items_to_copy:

        src = os.path.join(
            run_dir,
            item
        )

        dst = os.path.join(
            metadata_dst,
            item
        )

        if os.path.isdir(src):
            shutil.copytree(src, dst)

        elif os.path.isfile(src):
            shutil.copy2(src, dst)
