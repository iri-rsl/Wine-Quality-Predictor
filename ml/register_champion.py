import mlflow
from mlflow.tracking import MlflowClient

from ml.train import setup_mlflow

RUN_ID = "d1b5d7e2c48347aea30b0da19abaff19"
MODEL_NAME = "WineQualityPredictor"


def register_initial_champion():
    setup_mlflow()

    client = MlflowClient()

    # Verify that the selected run exists and succeeded
    run = client.get_run(RUN_ID)

    if run.info.status != "FINISHED":
        raise ValueError("The selected training run did not finish successfully.")

    if run.data.params.get("model_type") not in (None, "xgboost"):
        raise ValueError("The selected run is not an XGBoost experiment.")

    # Retrieve the model logged during this run
    model_outputs = run.outputs.model_outputs

    if not model_outputs:
        raise ValueError("No logged model found for this run.")

    if len(model_outputs) != 1:
        raise ValueError("Expected exactly one logged model.")

    model_id = model_outputs[0].model_id

    # MLflow 3 model URI
    model_uri = f"models:/{model_id}"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        await_registration_for=60,
    )

    version = registered_model.version

    # Assign the champion alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )

    # Verify the alias
    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion",
    )

    print(f"Registered model: {MODEL_NAME}")
    print(f"Registered version: {version}")
    print(f"Champion version: {champion.version}")
    print(f"Champion run ID: {champion.run_id}")


if __name__ == "__main__":
    register_initial_champion()
