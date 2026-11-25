import argparse

import mlflow
from mlflow.tracking import MlflowClient

from ml.data_validation import (
    validate_cross_version_consistency,
    validate_version,
)
from ml.evaluate import evaluate_champion_and_candidate, load_champion
from ml.model_registry import register_candidate
from ml.promotion import promote_if_passing, should_promote
from ml.train import setup_mlflow, train

MODEL_NAME = "WineQualityPredictor"


def run_continuous_training(dataset_version, allow_promotion=False):
    """Validate, retrain, evaluate and optionally promote a candidate."""
    setup_mlflow()
    client = MlflowClient()

    # 1. Validate the selected dataset
    validate_version(dataset_version)
    validate_cross_version_consistency()

    # 2. Load the existing champion
    champion = load_champion(model_name=MODEL_NAME)

    previous_version = str(champion["registry_version"])
    champion_dataset = champion["dataset_version"]

    print(f"Current champion: version {previous_version}")
    print(f"Champion dataset: {champion_dataset}")

    # 3. Determine whether retraining is necessary
    retrain = dataset_version != champion_dataset

    candidate = None
    comparison = None
    candidate_registry_version = None
    promoted = False

    if not retrain:
        reason = "Dataset version unchanged; retraining not required."

    else:
        # 4. Train the candidate
        candidate = train(
            model_name="xgboost",
            version=dataset_version,
            comparison_group="continuous_training",
        )

        # 5. Evaluate both on the same held-out observations
        comparison = evaluate_champion_and_candidate(
            candidate_model=candidate["model"],
            candidate_version=dataset_version,
            champion=champion,
        )

        # 6. Apply the quality gates
        approved, reason = should_promote(
            comparison["champion"],
            comparison["candidate"],
        )

        print("Champion metrics:", comparison["champion"])
        print("Candidate metrics:", comparison["candidate"])

        if approved and allow_promotion:
            # 7. Register only approved candidates
            candidate_registry_version = register_candidate(
                run_id=candidate["run_id"],
                model_name=MODEL_NAME,
            )

            # 8. Move the champion alias if the gates pass
            promoted, reason, _ = promote_if_passing(
                client=client,
                model_name=MODEL_NAME,
                candidate_version=candidate_registry_version,
                champion_metrics=comparison["champion"],
                candidate_metrics=comparison["candidate"],
            )

        elif approved:
            reason = "Candidate passed quality gates; dry run, no promotion."

    # 9. Verify the registry state
    current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    current_version = str(current.version)

    if promoted and current_version != str(candidate_registry_version):
        raise RuntimeError("Champion promotion verification failed.")

    if not promoted and current_version != previous_version:
        raise RuntimeError("Champion changed unexpectedly.")

    # 10. Log the CT decision
    with mlflow.start_run(run_name=f"ct_{dataset_version}") as run:
        mlflow.log_params(
            {
                "dataset_version": dataset_version,
                "model_name": MODEL_NAME,
                "retrain_required": retrain,
                "allow_promotion": allow_promotion,
                "previous_champion_version": previous_version,
            }
        )

        if candidate is not None:
            mlflow.log_param("candidate_run_id", candidate["run_id"])

        if comparison is not None:
            for name in ("champion", "candidate"):
                mlflow.log_metrics({f"{name}_{metric}": value for metric, value in comparison[name].items()})

        mlflow.set_tags(
            {
                "workflow": "continuous_training",
                "promotion_decision": ("promoted" if promoted else "retained"),
                "decision_reason": reason,
                "candidate_registry_version": str(candidate_registry_version or "none"),
                "current_champion_version": current_version,
            }
        )

    print("\n--- CT Result ---")
    print(f"Retraining required: {retrain}")
    print(f"Promoted: {promoted}")
    print(f"Reason: {reason}")
    print(f"Champion version: {current_version}")
    print(f"CT run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        required=True,
        choices=["v1", "v2"],
    )

    parser.add_argument(
        "--allow-promotion",
        action="store_true",
        help="Allow approved candidates to replace the champion",
    )

    args = parser.parse_args()

    run_continuous_training(
        dataset_version=args.version,
        allow_promotion=args.allow_promotion,
    )
