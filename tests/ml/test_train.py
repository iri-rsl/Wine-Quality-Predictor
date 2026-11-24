from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml import train as training


def test_load_data_reads_train_and_validation_splits(tmp_path, monkeypatch):
    version_dir = tmp_path / "v1"
    version_dir.mkdir()

    pd.DataFrame({"alcohol": [10.0, 11.0], "quality": [5, 6]}).to_csv(
        version_dir / "train.csv", index=False
    )
    pd.DataFrame({"alcohol": [12.0], "quality": [7]}).to_csv(
        version_dir / "validation.csv", index=False
    )

    monkeypatch.setattr(training, "DATA_DIR", tmp_path)

    X_train, y_train, X_validation, y_validation = training.load_data("v1")

    assert list(X_train.columns) == ["alcohol"]
    assert y_train.tolist() == [5, 6]
    assert X_validation["alcohol"].tolist() == [12.0]
    assert y_validation.tolist() == [7]


def test_create_model_returns_scaled_linear_regression_pipeline():
    model = training.create_model("linear_regression")

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["regressor"], LinearRegression)


def test_create_model_returns_configured_xgboost_regressor():
    model = training.create_model("xgboost")

    assert isinstance(model, XGBRegressor)
    assert model.get_params()["n_estimators"] == 300
    assert model.get_params()["learning_rate"] == 0.05
    assert model.get_params()["max_depth"] == 4
    assert model.get_params()["random_state"] == 42


def test_create_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model: other"):
        training.create_model("other")


def test_get_git_commit_returns_commit_hash(monkeypatch):
    run_git = Mock(return_value=SimpleNamespace(stdout="abc123\n"))
    monkeypatch.setattr(training.subprocess, "run", run_git)

    assert training.get_git_commit() == "abc123"


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(),
        training.subprocess.CalledProcessError(1, ["git"]),
    ],
)
def test_get_git_commit_returns_unknown_when_git_is_unavailable(monkeypatch, error):
    def fail_to_run(*args, **kwargs):
        raise error

    monkeypatch.setattr(training.subprocess, "run", fail_to_run)

    assert training.get_git_commit() == "unknown"


def test_get_dvc_hash_reads_hash_from_dvc_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "processed.dvc").write_text("outs:\n- md5: abc123\n", encoding="utf-8")
    monkeypatch.setattr(training, "ROOT_DIR", tmp_path)

    assert training.get_dvc_hash() == "abc123"


@pytest.mark.parametrize(
    "file_content",
    [
        None,
        "outs:\n- md5: NOT_A_HASH\n",
    ],
)
def test_get_dvc_hash_returns_unknown_when_hash_is_unavailable(
    tmp_path, monkeypatch, file_content
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    if file_content is not None:
        (data_dir / "processed.dvc").write_text(file_content, encoding="utf-8")

    monkeypatch.setattr(training, "ROOT_DIR", tmp_path)

    assert training.get_dvc_hash() == "unknown"


def test_setup_mlflow_requires_all_credentials(monkeypatch):
    monkeypatch.setattr(training, "load_dotenv", Mock())

    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="MLflow credentials are not fully configured"):
        training.setup_mlflow()


def test_setup_mlflow_configures_tracking(monkeypatch):
    monkeypatch.setattr(training, "load_dotenv", Mock())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "password")

    set_tracking_uri = Mock()
    set_experiment = Mock()
    monkeypatch.setattr(training.mlflow, "set_tracking_uri", set_tracking_uri)
    monkeypatch.setattr(
        training.mlflow,
        "get_tracking_uri",
        Mock(return_value="https://mlflow.example"),
    )
    monkeypatch.setattr(training.mlflow, "set_experiment", set_experiment)

    training.setup_mlflow()

    set_tracking_uri.assert_called_once_with("https://mlflow.example")
    set_experiment.assert_called_once_with(training.EXPERIMENT_NAME)


@pytest.mark.parametrize(
    ("model_name", "model_logger"),
    [
        ("linear_regression", "sklearn"),
        ("xgboost", "xgboost"),
    ],
)
def test_train_fits_model_and_logs_results(monkeypatch, model_name, model_logger):
    X_train = pd.DataFrame({"alcohol": [10.0, 11.0]})
    y_train = pd.Series([3, 4])
    X_validation = pd.DataFrame({"alcohol": [12.0, 13.0]})
    y_validation = pd.Series([3, 4])

    model = Mock()
    model.predict.return_value = np.array([2.0, 5.0])

    monkeypatch.setattr(training, "setup_mlflow", Mock())
    monkeypatch.setattr(
        training,
        "load_data",
        Mock(return_value=(X_train, y_train, X_validation, y_validation)),
    )
    monkeypatch.setattr(training, "create_model", Mock(return_value=model))
    monkeypatch.setattr(training, "get_git_commit", Mock(return_value="commit"))
    monkeypatch.setattr(training, "get_dvc_hash", Mock(return_value="datahash"))
    monkeypatch.setattr(
        training.mlflow,
        "start_run",
        Mock(
            return_value=nullcontext(
                SimpleNamespace(info=SimpleNamespace(run_id="run-id"))
            )
        ),
    )
    monkeypatch.setattr(training.mlflow, "log_params", Mock())
    monkeypatch.setattr(training.mlflow, "log_metrics", Mock())
    monkeypatch.setattr(training.mlflow, "set_tags", Mock())

    logged_artifact = {}

    def capture_artifact(path, artifact_path):
        logged_artifact["data"] = pd.read_csv(path)
        logged_artifact["artifact_path"] = artifact_path

    monkeypatch.setattr(training.mlflow, "log_artifact", capture_artifact)

    model_log = Mock()
    model_logger_module = getattr(training.mlflow, model_logger)
    monkeypatch.setattr(model_logger_module, "log_model", model_log)

    result = training.train(model_name, "v1")

    assert result is model
    model.fit.assert_called_once_with(X_train, y_train)
    model.predict.assert_called_once_with(X_validation)
    training.mlflow.log_metrics.assert_called_once_with(
        {"mae": 1.0, "rmse": 1.0, "r2": -3.0}
    )
    training.mlflow.set_tags.assert_called_once_with(
        {
            "git_commit": "commit",
            "dvc_data_hash": "datahash",
            "comparison_group": "initial_model_comparison",
        }
    )
    model_log.assert_called_once_with(model, name="model")
    assert logged_artifact["artifact_path"] == "evaluation"
    assert logged_artifact["data"]["residual"].tolist() == [1.0, -1.0]
    assert logged_artifact["data"]["absolute_error"].tolist() == [1.0, 1.0]
