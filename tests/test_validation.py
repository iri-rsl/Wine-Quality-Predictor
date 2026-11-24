"""Test 2 - malformed, incomplete and out-of-range inputs are rejected with 422."""

import pytest


def _errors(resp):
    assert resp.status_code == 422, resp.text
    return {".".join(str(p) for p in e["loc"][1:]): e["msg"] for e in resp.json()["detail"]}


@pytest.mark.parametrize("missing", ["alcohol", "pH", "is_red"])
def test_missing_feature(client, wine, missing):
    del wine[missing]
    assert missing in _errors(client.post("/predict", json=wine))


@pytest.mark.parametrize(
    "field,value", [("alcohol", "12.5"), ("alcohol", None), ("pH", [3.2]), ("is_red", True), ("is_red", 0.5)]
)
def test_wrong_types(client, wine, field, value):
    wine[field] = value
    assert field in _errors(client.post("/predict", json=wine))


@pytest.mark.parametrize(
    "field,value",
    [("alcohol", 6.9), ("alcohol", 16.1), ("pH", 0.0), ("density", 1.2), ("volatile_acidity", -0.1), ("is_red", 2)],
)
def test_out_of_range(client, wine, field, value):
    wine[field] = value
    assert field in _errors(client.post("/predict", json=wine))


def test_unexpected_field_rejected(client, wine):
    wine["quality"] = 7  # leaking the target must not be silently ignored
    assert "quality" in _errors(client.post("/predict", json=wine))


def test_nan_rejected(client, wine):
    body = '{"alcohol": NaN, ' + ", ".join(f'"{k}": {v}' for k, v in wine.items() if k != "alcohol") + "}"
    resp = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_free_so2_cannot_exceed_total(client, wine):
    wine.update(free_sulfur_dioxide=50.0, total_sulfur_dioxide=40.0)
    resp = client.post("/predict", json=wine)
    assert resp.status_code == 422
    assert "cannot exceed" in resp.text


@pytest.mark.parametrize("payload", ["not json", "[]", "{}"])
def test_malformed_body(client, payload):
    resp = client.post("/predict", content=payload, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
