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


def promote_if_passing(client, model_name, candidate_version, champion_metrics, candidate_metrics, alias="champion"):
    """Move `alias` to the candidate only if it passes the quality gates.

    `client` is an MlflowClient (or a fake in tests). Returns (promoted, reason, champion_version)
    where champion_version is the version holding the alias after the decision.
    """
    previous = str(client.get_model_version_by_alias(model_name, alias).version)
    promoted, reason = should_promote(champion_metrics, candidate_metrics)
    if not promoted:
        return False, reason, previous
    client.set_registered_model_alias(model_name, alias, str(candidate_version))
    return True, reason, str(candidate_version)
