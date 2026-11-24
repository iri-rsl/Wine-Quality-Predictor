from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml import register_champion as registration


def make_run(
    status="FINISHED",
    model_type="xgboost",
    model_outputs=None,
):
    if model_outputs is None:
        model_outputs = [SimpleNamespace(model_id="model-id")]

    return SimpleNamespace(
        info=SimpleNamespace(status=status),
        data=SimpleNamespace(params={"model_type": model_type}),
        outputs=SimpleNamespace(model_outputs=model_outputs),
    )


@pytest.mark.parametrize("model_type", [None, "xgboost"])
def test_register_initial_champion_registers_model_and_assigns_alias(monkeypatch, capsys, model_type):
    run = make_run(model_type=model_type)
    client = Mock()
    client.get_run.return_value = run

    champion = SimpleNamespace(version="3", run_id=registration.RUN_ID)
    client.get_model_version_by_alias.return_value = champion

    registered_model = SimpleNamespace(version="3")
    register_model = Mock(return_value=registered_model)

    monkeypatch.setattr(registration, "setup_mlflow", Mock())
    monkeypatch.setattr(
        registration,
        "MlflowClient",
        Mock(return_value=client),
    )
    monkeypatch.setattr(
        registration.mlflow,
        "register_model",
        register_model,
    )

    registration.register_initial_champion()

    client.get_run.assert_called_once_with(registration.RUN_ID)
    register_model.assert_called_once_with(
        model_uri="models:/model-id",
        name=registration.MODEL_NAME,
        await_registration_for=60,
    )
    client.set_registered_model_alias.assert_called_once_with(
        name=registration.MODEL_NAME,
        alias="champion",
        version="3",
    )
    client.get_model_version_by_alias.assert_called_once_with(
        name=registration.MODEL_NAME,
        alias="champion",
    )

    output = capsys.readouterr().out
    assert "Registered version: 3" in output
    assert "Champion version: 3" in output


@pytest.mark.parametrize(
    ("run", "expected_message"),
    [
        (
            make_run(status="FAILED"),
            "The selected training run did not finish successfully.",
        ),
        (
            make_run(model_type="linear_regression"),
            "The selected run is not an XGBoost experiment.",
        ),
        (
            make_run(model_outputs=[]),
            "No logged model found for this run.",
        ),
        (
            make_run(
                model_outputs=[
                    SimpleNamespace(model_id="first-model"),
                    SimpleNamespace(model_id="second-model"),
                ]
            ),
            "Expected exactly one logged model.",
        ),
    ],
)
def test_register_initial_champion_rejects_invalid_run(monkeypatch, run, expected_message):
    client = Mock()
    client.get_run.return_value = run
    register_model = Mock()

    monkeypatch.setattr(registration, "setup_mlflow", Mock())
    monkeypatch.setattr(
        registration,
        "MlflowClient",
        Mock(return_value=client),
    )
    monkeypatch.setattr(
        registration.mlflow,
        "register_model",
        register_model,
    )

    with pytest.raises(ValueError, match=expected_message):
        registration.register_initial_champion()

    register_model.assert_not_called()
    client.set_registered_model_alias.assert_not_called()
