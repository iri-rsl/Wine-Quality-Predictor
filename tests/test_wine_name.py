"""Optional bottle name: stored in the history, never used by the model."""

import sqlite3

import pytest

from app.db import Database


def test_name_is_returned_and_stored(client, make_user, wine):
    headers = make_user()
    body = client.post("/predict", json={**wine, "wine_name": "Château Test 2019"}, headers=headers).json()
    assert body["wine_name"] == "Château Test 2019"
    item = client.get("/history", headers=headers).json()[0]
    assert item["wine_name"] == "Château Test 2019"
    assert "wine_name" not in item["inputs"]  # kept apart from the model features


def test_name_does_not_change_prediction(client, wine):
    without = client.post("/predict", json=wine).json()
    named = client.post("/predict", json={**wine, "wine_name": "Any name"}).json()
    assert named["quality"] == without["quality"]


@pytest.mark.parametrize(
    "sent,stored",
    [(None, None), ("", None), ("   ", None), ("  Domaine   du\tTest \n", "Domaine du Test")],
)
def test_name_is_optional_and_normalised(client, make_user, wine, sent, stored):
    headers = make_user()
    payload = dict(wine) if sent is None else {**wine, "wine_name": sent}
    assert client.post("/predict", json=payload, headers=headers).json()["wine_name"] == stored
    assert client.get("/history", headers=headers).json()[0]["wine_name"] == stored


@pytest.mark.parametrize("value", ["x" * 81, 2019, ["Château"]])
def test_invalid_name_rejected(client, wine, value):
    resp = client.post("/predict", json={**wine, "wine_name": value})
    assert resp.status_code == 422
    assert "wine_name" in resp.text


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,"
        " created_at TEXT NOT NULL, inputs TEXT NOT NULL, quality REAL NOT NULL, label TEXT NOT NULL,"
        " model_version TEXT NOT NULL);"
    )
    conn.close()
    db = Database(str(path))
    with db.connect() as c:
        columns = {row["name"] for row in c.execute("PRAGMA table_info(predictions)")}
    assert "wine_name" in columns
