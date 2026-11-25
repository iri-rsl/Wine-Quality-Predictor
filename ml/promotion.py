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


def promote_if_passing(
    client,
    model_name,
    candidate_version,
    champion_metrics,
    candidate_metrics,
    alias="champion",
):
    """Promote a candidate only if it passes all quality gates."""

    previous = str(
        client.get_model_version_by_alias(model_name, alias).version
    )

    promoted, reason = should_promote(
        champion_metrics,
        candidate_metrics,
    )

    if not promoted:
        return False, reason, previous

    # Mark the candidate as approved
    client.set_model_version_tag(
        name=model_name,
        version=str(candidate_version),
        key="approved",
        value="true",
    )

    # Move the champion alias
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(candidate_version),
    )

    # Verify the new champion
    current = client.get_model_version_by_alias(model_name, alias)

    if str(current.version) != str(candidate_version):
        raise RuntimeError("Champion alias verification failed.")

    return True, reason, str(current.version)
