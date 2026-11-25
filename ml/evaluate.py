import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml.train import DATA_DIR, setup_mlflow

MODEL_NAME = "WineQualityPredictor"


def load_evaluation_data(version):
    """Load the held-out evaluation dataset."""
    if version not in ("v1", "v2"):
        raise ValueError(f"Unsupported dataset version: {version}")

    df = pd.read_csv(DATA_DIR / version / "test.csv")

    X = df.drop(columns="quality")
    y = df["quality"]

    return X, y


def evaluate_model(model, X, y):
    """Evaluate a model using regression metrics."""
    predictions = model.predict(X)

    return {
        "mae": mean_absolute_error(y, predictions),
        "rmse": mean_squared_error(y, predictions) ** 0.5,
        "r2": r2_score(y, predictions),
    }


def load_champion(model_name=MODEL_NAME):
    """Load the champion and retrieve its training dataset version."""
    setup_mlflow()

    client = MlflowClient()

    champion_info = client.get_model_version_by_alias(
        name=model_name,
        alias="champion",
    )

    run = client.get_run(champion_info.run_id)

    dataset_version = run.data.params.get("dataset_version")

    if not dataset_version:
        raise ValueError("Champion dataset version is missing from MLflow.")

    model_uri = f"models:/{model_name}@champion"
    model = mlflow.pyfunc.load_model(model_uri)

    return {
        "model": model,
        "registry_version": champion_info.version,
        "dataset_version": dataset_version,
        "run_id": champion_info.run_id,
    }


def evaluate_champion_and_candidate(
    candidate_model,
    candidate_version,
    champion=None,
):
    """Compare champion and candidate on identical held-out observations."""

    if champion is None:
        champion = load_champion()

    champion_version = champion["dataset_version"]

    X_champion, y_champion = load_evaluation_data(champion_version)
    X_candidate, y_candidate = load_evaluation_data(candidate_version)

    # Ensure both versions contain the same evaluation observations.
    common_columns = X_champion.columns.intersection(X_candidate.columns)

    if not X_champion[common_columns].reset_index(drop=True).equals(X_candidate[common_columns].reset_index(drop=True)):
        raise ValueError("Champion and candidate evaluation observations differ.")

    if not y_champion.reset_index(drop=True).equals(y_candidate.reset_index(drop=True)):
        raise ValueError("Champion and candidate evaluation targets differ.")

    champion_metrics = evaluate_model(
        champion["model"],
        X_champion,
        y_champion,
    )

    candidate_metrics = evaluate_model(
        candidate_model,
        X_candidate,
        y_candidate,
    )

    return {
        "champion": champion_metrics,
        "candidate": candidate_metrics,
        "champion_registry_version": champion["registry_version"],
    }
