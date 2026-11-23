# Dataset Documentation — Wine Quality Predictor

## 1. Dataset Overview

The project uses the **Wine Quality dataset** from Hugging Face, originally published by the UCI Machine Learning Repository.

The objective is to predict wine quality from physicochemical measurements using regression.

| Property | Description |
|---|---|
| Source | [Hugging Face — mstz/wine](https://huggingface.co/datasets/mstz/wine) |
| Original dataset | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Observations | 6,497 |
| Features | 12 (including wine type) |
| Target | `quality` (scores 3–9) |
| Task | Regression |

## 2. License and Attribution

The Hugging Face repository declares a generic Creative Commons (`cc`) license. The original UCI dataset is distributed under **CC BY 4.0**, allowing reuse and modification with appropriate attribution.

**Original publication:**

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.

License: https://creativecommons.org/licenses/by/4.0/

This project uses the dataset for educational purposes and documents all preprocessing modifications.

## 3. Dataset Versions

Two versions were created to study the impact of data preparation on model performance.

| Property | V1 | V2 |
|---|---|---|
| Features | 11 | 12 |
| Wine type (`is_red`) | Excluded | Included |
| Training duplicates | Retained | Removed |
| Training samples | 3,961 | 3,479 |
| Validation samples | 893 | 893 |
| Test samples | 948 | 948 |

Both versions originate from the same dataset and use identical underlying validation and test observations.

## 4. Data Preparation and Splitting

The dataset was initially split into approximately **70% training, 15% validation and 15% test**, using `random_state=42`.

Duplicate observations were removed across split boundaries to avoid data leakage.

- **Training:** Used to fit models.
- **Validation:** Used for model comparison and selection.
- **Test:** Reserved for held-out evaluation and champion/candidate comparisons, without fitting or tuning.

## 5. Data Versioning and Storage

DVC is used to track dataset versions, with **Azure Blob Storage** as the remote storage backend.

| Setting | Value |
|---|---|
| Storage account | `winequalitydata` |
| Container | `dvc` |
| DVC remote | `myazure` |
| DVC pointer | `data/processed.dvc` |

The processed data is stored under `data/processed/v1/` and `data/processed/v2/`.

Raw datasets and credentials are excluded from Git. Only DVC metadata is committed.

To retrieve the datasets:

```bash
dvc pull
```

## 6. Reproducibility

| Identifier | Value |
|---|---|
| Source dataset | `mstz/wine` |
| Source file | `wine/train.csv` |
| Source revision | To be recorded |
| Random seed | `42` |
| Git commit SHA | To be recorded |
| DVC hash | To be recorded |

Each released dataset version must be identifiable through its corresponding Git commit/tag and DVC metadata.

## 7. Limitations

- Wine quality scores are subjective and unevenly distributed.
- The dataset contains relatively few very high- and low-quality wines.
- Deduplication reduces the available training observations.
- Adding wine type does not necessarily improve prediction accuracy.
- Predictions are estimates, not certified assessments of wine quality.