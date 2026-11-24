import pytest

from ml.promotion import should_promote

@pytest.mark.parametrize(
    ("champion_MAE", "candidate_MAE", "candidate_rmse", "expected", "expected_reason"),
    [
        (0.60, 0.50, 0.70, True, "Candidate passed all quality gates"),        # Beats threshold and champion
        (0.60, 0.5999, 0.70, False, "Candidate improvement is insufficient"),  # Insufficient improvement
        (0.60, 0.65, 0.70, False, "Candidate MAE exceeds quality threshold"),  # Above MAE threshold
        (0.60, 0.50, 0.85, False, "Candidate RMSE exceeds quality threshold"), # Above RMSE threshold

    ]
)
def test_should_promote(champion_MAE, candidate_MAE, candidate_rmse, expected, expected_reason):
    result, reason = should_promote(
        champion_metrics={"mae": champion_MAE},
        candidate_metrics={"mae": candidate_MAE, "rmse": candidate_rmse},
    )
    assert result is expected
    assert reason == expected_reason

if __name__ == "__main__":
    pytest.main()