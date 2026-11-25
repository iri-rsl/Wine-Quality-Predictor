"""Promotion rule: quality floor, minimum improvement, champion unchanged after rejection."""

from types import SimpleNamespace

import pytest

from ml.promotion import (
    MAX_MAE,
    MAX_RMSE,
    MIN_MAE_IMPROVEMENT,
    promote_if_passing,
    should_promote,
)

CHAMPION = {"mae": 0.54, "rmse": 0.69}


def required_mae(champion_mae):
    return champion_mae * (1 - MIN_MAE_IMPROVEMENT)


@pytest.mark.parametrize(
    "candidate",
    [
        {"mae": 0.50, "rmse": 0.65},  # clear improvement
        {"mae": round(required_mae(0.54), 6), "rmse": 0.69},  # exactly the minimum improvement
    ],
)
def test_candidate_with_required_improvement_passes(candidate):
    promoted, reason = should_promote(CHAMPION, candidate)
    assert promoted is True
    assert reason == "Candidate passed all quality gates"


@pytest.mark.parametrize(
    "candidate,expected_reason",
    [
        ({"mae": MAX_MAE + 0.01, "rmse": 0.70}, "MAE exceeds"),  # worse than the floor
        ({"mae": 0.40, "rmse": MAX_RMSE + 0.01}, "RMSE exceeds"),  # good MAE but RMSE above floor
        ({"mae": 0.61, "rmse": 0.81}, "MAE exceeds"),  # fails both: MAE checked first
    ],
)
def test_candidate_below_quality_floor_is_rejected(candidate, expected_reason):
    promoted, reason = should_promote({"mae": 0.70, "rmse": 0.90}, candidate)
    assert promoted is False
    assert expected_reason in reason


@pytest.mark.parametrize("candidate_mae", [0.54, 0.539, 0.55])  # equal, < 1 % better, worse
def test_insufficient_improvement_is_rejected(candidate_mae):
    promoted, reason = should_promote(CHAMPION, {"mae": candidate_mae, "rmse": 0.69})
    assert promoted is False
    assert "insufficient" in reason


class FakeClient:
    def __init__(self, champion_version):
        self.aliases = {"champion": champion_version}
        self.calls = []

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=self.aliases[alias])

    def set_registered_model_alias(self, name, alias, version):
        self.calls.append((name, alias, version))
        self.aliases[alias] = version

    def set_model_version_tag(self, name, version, key, value):
        if not hasattr(self, "version_tags"):
            self.version_tags = {}

        self.version_tags.setdefault(str(version), {})[key] = value


def test_passing_candidate_moves_alias():
    client = FakeClient("1")
    promoted, _, version = promote_if_passing(client, "WineQualityPredictor", 2, CHAMPION, {"mae": 0.50, "rmse": 0.65})
    assert promoted and version == "2"
    assert client.aliases["champion"] == "2"
    assert client.calls == [("WineQualityPredictor", "champion", "2")]
    assert client.version_tags["2"]["approved"] == "true"


@pytest.mark.parametrize("candidate", [{"mae": 0.539, "rmse": 0.69}, {"mae": 0.70, "rmse": 0.90}])
def test_champion_unchanged_after_rejection(candidate):
    client = FakeClient("1")
    promoted, reason, version = promote_if_passing(client, "WineQualityPredictor", 2, CHAMPION, candidate)
    assert promoted is False and reason
    assert version == "1"
    assert client.aliases["champion"] == "1"
    assert client.calls == []
