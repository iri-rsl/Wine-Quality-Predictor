import argparse

from mlflow.tracking import MlflowClient

from ml.train import setup_mlflow

MODEL_NAME = "WineQualityPredictor"


def rollback(version, model_name=MODEL_NAME):
    setup_mlflow()
    client = MlflowClient()

    target = client.get_model_version(model_name, str(version))

    # Only versions explicitly marked as approved can be restored.
    if target.tags.get("approved") != "true":
        raise ValueError(f"Model version {version} is not marked as approved.")

    previous = client.get_model_version_by_alias(model_name, "champion")

    client.set_registered_model_alias(model_name, "champion", str(version))

    restored = client.get_model_version_by_alias(model_name, "champion")

    if str(restored.version) != str(version):
        raise RuntimeError("Rollback verification failed.")

    print(f"Previous champion: {previous.version}")
    print(f"Restored champion: {restored.version}")

    return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    rollback(args.version)
