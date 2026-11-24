"""Test 1 - successful predictions and their output format."""

import pytest

from app.main import quality_label


def test_valid_prediction_has_expected_format(client, wine):
    resp = client.post("/predict", json=wine)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"quality", "quality_class", "label", "model_name", "model_version", "prediction_id"}
    assert isinstance(body["quality"], float)
    assert 0 <= body["quality"] <= 10
    assert body["quality_class"] == round(body["quality"])
    assert body["label"] in {"Poor", "Average", "Good", "Excellent"}
    assert body["model_version"] == "dummy-0"
    assert body["prediction_id"] is None  # anonymous calls are not stored


def test_prediction_is_deterministic(client, wine):
    first = client.post("/predict", json=wine).json()["quality"]
    second = client.post("/predict", json=wine).json()["quality"]
    assert first == second


def test_integer_values_are_accepted_for_float_fields(client, wine):
    wine["free_sulfur_dioxide"] = 11
    wine["alcohol"] = 10
    assert client.post("/predict", json=wine).status_code == 200


@pytest.mark.parametrize("field,value", [("alcohol", 7.0), ("alcohol", 16.0), ("pH", 2.5), ("density", 1.05)])
def test_boundary_values_are_accepted(client, wine, field, value):
    wine[field] = value
    resp = client.post("/predict", json=wine)
    assert resp.status_code == 200
    assert 0 <= resp.json()["quality"] <= 10


def test_extreme_inputs_are_clipped_to_scale(client, wine):
    # The dummy rule would exceed 10 for this wine; the API clips to the 0-10 scale.
    wine.update(alcohol=16.0, volatile_acidity=0.0, sulphates=2.5, total_sulfur_dioxide=0.0, free_sulfur_dioxide=0.0)
    body = client.post("/predict", json=wine).json()
    assert body["quality"] <= 10
    assert body["label"] == "Excellent"


def test_better_wine_scores_higher(client, wine):
    faulty = dict(wine, volatile_acidity=1.4, alcohol=8.5)
    good = dict(wine, volatile_acidity=0.25, alcohol=13.0)
    assert client.post("/predict", json=good).json()["quality"] > client.post("/predict", json=faulty).json()["quality"]


@pytest.mark.parametrize(
    "score,label", [(0, "Poor"), (4.99, "Poor"), (5.0, "Average"), (6.0, "Good"), (6.99, "Good"), (7.0, "Excellent")]
)
def test_quality_label_thresholds(score, label):
    assert quality_label(score) == label
