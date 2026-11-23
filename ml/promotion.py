MAX_MAE = 0.60
MAX_RMSE = 0.80
MIN_MAE_IMPROVEMENT = 0.01


def should_promote(champion_metrics, candidate_metrics):
    champion_mae = champion_metrics["mae"]
    candidate_mae = candidate_metrics["mae"]
    candidate_rmse = candidate_metrics["rmse"]

    if candidate_mae > MAX_MAE:
        return False, "Candidate MAE exceeds quality threshold"

    if candidate_rmse > MAX_RMSE:
        return False, "Candidate RMSE exceeds quality threshold"

    required_mae = champion_mae * (1 - MIN_MAE_IMPROVEMENT)

    if candidate_mae > required_mae:
        return False, "Candidate improvement is insufficient"

    return True, "Candidate passed all quality gates"