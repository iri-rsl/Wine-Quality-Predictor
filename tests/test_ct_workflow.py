import pytest

from ml import continuous_training as ct


def test_invalid_data_stops_training(monkeypatch):
    def reject_dataset(version):
        raise ValueError("Invalid dataset")

    def unexpected_call(*args, **kwargs):
        pytest.fail("CT continued after data validation failed")

    monkeypatch.setattr(ct, "setup_mlflow", lambda: None)
    monkeypatch.setattr(ct, "validate_version", reject_dataset)

    monkeypatch.setattr(ct, "load_champion", unexpected_call)
    monkeypatch.setattr(ct, "train", unexpected_call)
    monkeypatch.setattr(ct, "register_candidate", unexpected_call)
    monkeypatch.setattr(ct, "promote_if_passing", unexpected_call)

    with pytest.raises(ValueError, match="Invalid dataset"):
        ct.run_continuous_training("v2")
