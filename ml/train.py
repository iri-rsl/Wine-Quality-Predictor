import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"

EXPERIMENT_NAME = "Wine Quality Predictor"


def load_data(version):
    version_dir = DATA_DIR / version

    train_df = pd.read_csv(version_dir / "train.csv")
    validation_df = pd.read_csv(version_dir / "validation.csv")

    X_train = train_df.drop(columns="quality")
    y_train = train_df["quality"]

    X_validation = validation_df.drop(columns="quality")
    y_validation = validation_df["quality"]

    return X_train, y_train, X_validation, y_validation


def create_model(model_name):
    if model_name == "linear_regression":
        return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])

    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_dvc_hash():
    dvc_file = ROOT_DIR / "data" / "processed.dvc"

    if not dvc_file.exists():
        return "unknown"

    content = dvc_file.read_text(encoding="utf-8")
    match = re.search(r"md5:\s*([a-f0-9]+)", content)

    return match.group(1) if match else "unknown"


def setup_mlflow():
    load_dotenv(ROOT_DIR / ".env")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not all([tracking_uri, username, password]):
        raise ValueError("MLflow credentials are not fully configured.")

    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(EXPERIMENT_NAME)


def train(model_name, version, comparison_group="initial_model_comparison"):
    setup_mlflow()

    X_train, y_train, X_validation, y_validation = load_data(version)

    model = create_model(model_name)

    run_name = f"{model_name}_{version}"

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        mae = mean_absolute_error(y_validation, predictions)
        rmse = mean_squared_error(y_validation, predictions) ** 0.5
        r2 = r2_score(y_validation, predictions)

        mlflow.log_params(
            {
                "model": model_name,
                "dataset_version": version,
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
                "feature_count": X_train.shape[1],
            }
        )

        if model_name == "linear_regression":
            mlflow.log_params(
                {
                    "scaler": "StandardScaler",
                    "fit_intercept": True,
                }
            )

            mlflow.sklearn.log_model(model, name="model")

        elif model_name == "xgboost":
            mlflow.log_params(
                {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "min_child_weight": 3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                }
            )

            mlflow.xgboost.log_model(model, name="model")

        mlflow.log_metrics(
            {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

        # Save validation predictions as an evaluation artifact
        evaluation_df = pd.DataFrame(
            {
                "actual_quality": y_validation.to_numpy(),
                "predicted_quality": predictions,
            }
        )

        evaluation_df["residual"] = evaluation_df["actual_quality"] - evaluation_df["predicted_quality"]

        evaluation_df["absolute_error"] = evaluation_df["residual"].abs()

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluation_path = Path(temp_dir) / "validation_predictions.csv"

            evaluation_df.to_csv(evaluation_path, index=False)

            mlflow.log_artifact(str(evaluation_path), artifact_path="evaluation")

        mlflow.set_tags(
            {
                "git_commit": get_git_commit(),
                "dvc_data_hash": get_dvc_hash(),
                "comparison_group": comparison_group,
            }
        )

        print(f"\nCompleted: {run_name}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"Run ID: {run.info.run_id}")

    return {
        "model": model,
        "run_id": run.info.run_id,
        "model_name": model_name,
        "dataset_version": version,
        "validation_metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["linear_regression", "xgboost"])

    parser.add_argument("--version", choices=["v1", "v2"])

    parser.add_argument("--all", action="store_true", help="Run all model and dataset combinations")

    args = parser.parse_args()

    if args.all:
        combinations = [
            ("linear_regression", "v1"),
            ("xgboost", "v1"),
            ("linear_regression", "v2"),
            ("xgboost", "v2"),
        ]

        for model_name, version in combinations:
            train(model_name, version)

    elif args.model and args.version:
        train(args.model, args.version)

    else:
        parser.error("Provide both --model and --version, or use --all.")
