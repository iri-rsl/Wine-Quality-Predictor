# Model Experiments — Wine Quality Predictor

## 1. Objective

The objective is to predict wine quality scores using regression and compare different algorithms and dataset versions.

Two algorithms were evaluated:

- **Linear Regression:** Simple, interpretable baseline.
- **XGBoost Regressor:** More advanced ensemble model capable of capturing nonlinear relationships.

Each algorithm was trained on both dataset versions (V1 and V2), producing four experiments.

## 2. Evaluation Metrics

Three metrics were used:

| Metric | Purpose | Direction |
|---|---|---|
| **MAE** | Average absolute prediction error; primary metric | Lower is better |
| RMSE | Penalizes larger errors | Lower is better |
| R² | Measures explained variance | Higher is better |

**MAE was selected as the primary metric** because it directly expresses the average prediction error in wine-quality points, making it interpretable for application users.

## 3. Experiment Results

All models were evaluated on the shared validation observations.

| Model | Dataset | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---:|---:|---:|
| Linear Regression | V1 | 0.5696 | 0.7358 | 0.2832 |
| XGBoost | V1 | **0.5390** | **0.6870** | **0.3751** |
| Linear Regression | V2 | 0.5658 | 0.7313 | 0.2921 |
| XGBoost | V2 | 0.5422 | 0.6909 | 0.3681 |

XGBoost V1 achieved the best results across all three metrics.

## 4. Quality Gates

The following criteria govern future candidate promotions:

| Criterion | Requirement |
|---|---|
| MAE quality floor | ≤ 0.60 |
| RMSE quality floor | ≤ 0.80 |
| Minimum MAE improvement | ≥ 1% over champion |

A candidate is promoted only if it passes all three requirements when evaluated against the champion on the **same held-out observations**, which must not have been used for training or tuning.

If a candidate fails, the existing champion remains unchanged.

## 5. Initial Champion Selection

**XGBoost V1** was selected as the initial champion because it achieved the lowest validation MAE and RMSE and the highest R².

The model was registered in the shared DagsHub MLflow Model Registry.

| Property | Value |
|---|---|
| Registered model | `WineQualityPredictor` |
| Initial champion | XGBoost V1 |
| Registry version | `1` |
| Alias | `champion` |
| MLflow run ID | `d1b5d7e2c48347aea30b0da19abaff19` |

The initial champion is established without requiring improvement over a previous model.

Future candidates must pass the quality gates before the `champion` alias can be reassigned.

## 6. Experiment Tracking and Reproducibility

Experiments are stored on the shared [DagsHub MLflow server](https://dagshub.com/iri-rsl/Wine-Quality-Predictor.mlflow).

Each training run records model parameters, evaluation metrics, Git commit SHA, dataset version, DVC identifier, trained model and validation prediction artifacts.

The held-out test set remains reserved for subsequent champion/candidate evaluation and is not used for model selection or hyperparameter tuning.

## Continuous Training (CT)

Continuous Training is implemented through `.github/workflows/ct.yml` and can be triggered manually from GitHub Actions.

### Workflow

1. **Data retrieval:** Pull versioned datasets from Azure Blob Storage using DVC.
2. **Validation:** Check dataset schema, values, and train/validation/test separation.
3. **Retraining decision:** Compare the requested dataset version with the current champion's training dataset.
4. **Training:** If the dataset version has changed, train an XGBoost candidate.
5. **Evaluation:** Compare the champion and candidate on the same held-out test observations.
6. **Promotion:** Approve candidates only if all quality gates pass.
7. **Tracking:** Log training runs, metrics, and decisions to DagsHub MLflow.

### Promotion criteria

A candidate must satisfy all three conditions:

- MAE ≤ 0.60
- RMSE ≤ 0.80
- MAE improves by at least 1% compared with the champion.

Rejected candidates do not replace the champion.

### Model registry and rollback

The approved production model is identified by the `champion` alias in the MLflow Model Registry. Approved versions receive the `approved=true` tag.

Rollback is implemented in `ml/rollback.py` and only permits restoring previously approved versions.

### Execution

Run locally:

```bash
python -m ml.continuous_training --version v2
```

Or trigger **Continuous Training** in GitHub Actions with the desired dataset version.

By default, execution is a dry run: training and evaluation occur, but no model is promoted. Promotion can be enabled using `--allow-promotion`.

Refreshing the deployed application and verifying the live model version are handled as part of deployment integration.