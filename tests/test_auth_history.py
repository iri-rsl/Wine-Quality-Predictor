"""Test 3 - accounts, login and per-user prediction history."""

import pytest


def test_signup_then_login(client, make_user):
    make_user("alice", "correct-horse")
    resp = client.post("/auth/login", json={"username": "alice", "password": "correct-horse"})
    assert resp.status_code == 200
    assert resp.json()["token_type"] == "bearer"
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {resp.json()['access_token']}"})
    assert me.json()["username"] == "alice"


@pytest.mark.parametrize(
    "username,password",
    [("alice", "wrong-password"), ("nobody", "correct-horse"), ("ALICE", "wrong-password")],
)
def test_login_rejects_bad_credentials(client, make_user, username, password):
    make_user("alice", "correct-horse")
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 401
    assert "access_token" not in resp.json()


def test_duplicate_username_is_case_insensitive(client, make_user):
    make_user("alice")
    resp = client.post("/auth/signup", json={"username": "Alice", "password": "another-pass"})
    assert resp.status_code == 409


@pytest.mark.parametrize(
    "username,password",
    [("ab", "long-enough"), ("bad name", "long-enough"), ("alice", "short"), ("a" * 33, "long-enough")],
)
def test_signup_validation(client, username, password):
    assert client.post("/auth/signup", json={"username": username, "password": password}).status_code == 422


def test_password_is_not_stored_in_clear(client, make_user):
    make_user("alice", "correct-horse")
    row = client.app.state.db.get_user("alice")
    assert "correct-horse" not in row["password_hash"]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-token"}, {"Authorization": "Basic abc"}])
def test_history_requires_valid_token(client, headers):
    assert client.get("/history", headers=headers).status_code in (401, 403)


def test_invalid_token_on_predict_is_rejected(client, wine):
    resp = client.post("/predict", json=wine, headers={"Authorization": "Bearer forged"})
    assert resp.status_code == 401


def test_history_records_authenticated_predictions_newest_first(client, make_user, wine):
    headers = make_user()
    ids = [
        client.post("/predict", json=dict(wine, alcohol=a), headers=headers).json()["prediction_id"]
        for a in (9, 11, 13)
    ]
    assert all(isinstance(i, int) for i in ids)
    items = client.get("/history", headers=headers).json()
    assert [i["id"] for i in items] == ids[::-1]
    assert items[0]["inputs"]["alcohol"] == 13
    assert items[0]["model_version"] == "dummy-0"


def test_history_is_isolated_between_users(client, make_user, wine):
    alice, bob = make_user("alice"), make_user("bob")
    client.post("/predict", json=wine, headers=alice)
    client.post("/predict", json=wine)  # anonymous: never stored
    assert len(client.get("/history", headers=alice).json()) == 1
    assert client.get("/history", headers=bob).json() == []


def test_history_limit_and_clear(client, make_user, wine):
    headers = make_user()
    for _ in range(3):
        client.post("/predict", json=wine, headers=headers)
    assert len(client.get("/history?limit=2", headers=headers).json()) == 2
    assert client.get("/history?limit=0", headers=headers).status_code == 422
    assert client.delete("/history", headers=headers).json() == {"deleted": 3}
    assert client.get("/history", headers=headers).json() == []


def test_logout_invalidates_token(client, make_user):
    headers = make_user()
    assert client.post("/auth/logout", headers=headers).status_code == 204
    assert client.get("/history", headers=headers).status_code == 401
