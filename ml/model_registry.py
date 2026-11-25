import mlflow
from mlflow.tracking import MlflowClient

from ml.train import setup_mlflow


MODEL_NAME = "WineQualityPredictor"


def get_champion_info(model_name=MODEL_NAME):
    """Retrieve the currently approved champion."""
    setup_mlflow()

    client = MlflowClient()

    return client.get_model_version_by_alias(
        name=model_name,
        alias="champion",
    )


def register_candidate(run_id, model_name=MODEL_NAME):
    """Register a trained candidate without promoting it."""
    setup_mlflow()

    client = MlflowClient()
    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise ValueError("Training run did not finish successfully.")

    model_outputs = run.outputs.model_outputs

    if len(model_outputs) != 1:
        raise ValueError("Expected exactly one logged model.")

    model_id = model_outputs[0].model_id

    registered = mlflow.register_model(
        model_uri=f"models:/{model_id}",
        name=model_name,
        await_registration_for=60,
    )

    return registered.version

