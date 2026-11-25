
from types import SimpleNamespace

import pytest

from ml import rollback as rollback_module


class FakeClient:
    def __init__(self):
        self.alias_version = "2"
        self.versions = {
            "1": {"approved": "true"},
            "2": {"approved": "true"},
            "3": {},
        }

    def get_model_version(self, name, version):
        version = str(version)

        if version not in self.versions:
            raise ValueError("Model version not found")

        return SimpleNamespace(
            version=version,
            tags=self.versions[version],
        )

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=self.alias_version)

    def set_registered_model_alias(self, name, alias, version):
        self.alias_version = str(version)


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()

    monkeypatch.setattr(
        rollback_module,
        "setup_mlflow",
        lambda: None,
    )

    monkeypatch.setattr(
        rollback_module,
        "MlflowClient",
        lambda: fake,
    )

    return fake


def test_rollback_to_approved_version(client):
    result = rollback_module.rollback("1")

    assert result.version == "1"
    assert client.alias_version == "1"


def test_reject_unapproved_version(client):
    with pytest.raises(ValueError, match="not marked as approved"):
        rollback_module.rollback("3")

    assert client.alias_version == "2"


def test_reject_nonexistent_version(client):
    with pytest.raises(ValueError, match="not found"):
        rollback_module.rollback("99")

    assert client.alias_version == "2"


def test_rollback_to_current_champion(client):
    result = rollback_module.rollback("2")

    assert result.version == "2"
    assert client.alias_version == "2"
