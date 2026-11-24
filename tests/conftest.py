import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

VALID_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "is_red": 1,
}


@pytest.fixture
def wine():
    return dict(VALID_WINE)


@pytest.fixture
def settings(tmp_path):
    return Settings(model_source="dummy", database_path=str(tmp_path / "test.db"), admin_token="secret")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def make_user(client):
    """Create a user and return the Authorization header."""

    def _make(username="alice", password="correct-horse"):
        resp = client.post("/auth/signup", json={"username": username, "password": password})
        assert resp.status_code == 201, resp.text
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _make
