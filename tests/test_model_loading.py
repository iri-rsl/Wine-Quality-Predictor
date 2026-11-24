"""Test 4 - champion resolution, readiness and reload, using a fake registry (no network)."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_loader import MlflowChampionLoader


class FakeModel:
    def __init__(self, score, features=None, fail=False):
        self.score, self.fail = score, fail
        schema = SimpleNamespace(input_names=lambda: features) if features else None
        self.metadata = SimpleNamespace(get_input_schema=lambda: schema)

    def predict(self, df):
        if self.fail:
            raise RuntimeError("corrupted artifact")
        return np.full(len(df), self.score)


class FakeRegistry:
    """Versions 1..n; aliases map alias -> version."""

    def __init__(self, models, aliases):
        self.models, self.aliases = models, aliases

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise LookupError(f"alias {alias} not found for {name}")
        version = self.aliases[alias]
        return SimpleNamespace(version=version, run_id=f"run-{version}")

    def load_model(self, uri):
        version = int(uri.rsplit("/", 1)[1])
        return self.models[version]


@pytest.fixture
def mlflow_settings(tmp_path):
    return Settings(model_source="mlflow", database_path=str(tmp_path / "db.sqlite"), admin_token="secret")


def make_client(settings, registry):
    loader = MlflowChampionLoader(settings, client=registry, load_model=registry.load_model)
    return TestClient(create_app(settings, loader=loader))


def test_loads_champion_not_newest_candidate(mlflow_settings, wine):
    registry = FakeRegistry({1: FakeModel(6.0), 2: FakeModel(9.0)}, {"champion": 1, "candidate": 2})
    with make_client(mlflow_settings, registry) as client:
        info = client.get("/model-info").json()
        assert info["ready"] is True and info["version"] == "1" and info["run_id"] == "run-1"
        body = client.post("/predict", json=wine).json()
        assert body["model_version"] == "1" and body["quality"] == 6.0


def test_features_follow_model_signature(mlflow_settings, wine):
    features = ["alcohol", "volatile_acidity", "is_red"]
    registry = FakeRegistry({1: FakeModel(5.5, features=features)}, {"champion": 1})
    with make_client(mlflow_settings, registry) as client:
        assert client.get("/model-info").json()["features"] == features
        assert client.post("/predict", json=wine).status_code == 200


@pytest.mark.parametrize(
    "registry",
    [
        FakeRegistry({1: FakeModel(6.0)}, {}),  # no champion alias
        FakeRegistry({1: FakeModel(6.0, fail=True)}, {"champion": 1}),  # model cannot predict
        FakeRegistry({1: FakeModel(float("nan"))}, {"champion": 1}),  # non-finite output
        FakeRegistry({1: FakeModel(6.0, features=["colour"])}, {"champion": 1}),  # unknown feature
    ],
    ids=["missing-alias", "predict-error", "nan-output", "unknown-feature"],
)
def test_not_ready_when_champion_unusable(mlflow_settings, wine, registry):
    with make_client(mlflow_settings, registry) as client:
        assert client.get("/health").status_code == 200  # process alive
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=wine).status_code == 503
        info = client.get("/model-info").json()
        assert info["ready"] is False and info["version"] is None and info["error"]


def test_reload_picks_up_promoted_champion(mlflow_settings, wine):
    registry = FakeRegistry({1: FakeModel(6.0), 2: FakeModel(6.5)}, {"champion": 1})
    with make_client(mlflow_settings, registry) as client:
        registry.aliases["champion"] = 2  # promotion done by the CT workflow
        resp = client.post("/admin/reload-model", headers={"X-Admin-Token": "secret"})
        assert resp.status_code == 200 and resp.json()["version"] == "2"
        assert client.post("/predict", json=wine).json()["model_version"] == "2"


def test_failed_reload_keeps_previous_model(mlflow_settings, wine):
    registry = FakeRegistry({1: FakeModel(6.0), 2: FakeModel(0, fail=True)}, {"champion": 1})
    with make_client(mlflow_settings, registry) as client:
        registry.aliases["champion"] = 2
        resp = client.post("/admin/reload-model", headers={"X-Admin-Token": "secret"})
        assert resp.status_code == 502
        assert client.get("/ready").status_code == 200
        assert client.post("/predict", json=wine).json()["model_version"] == "1"


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Token": "wrong"}])
def test_reload_requires_admin_token(mlflow_settings, headers):
    registry = FakeRegistry({1: FakeModel(6.0)}, {"champion": 1})
    with make_client(mlflow_settings, registry) as client:
        assert client.post("/admin/reload-model", headers=headers).status_code == 403
