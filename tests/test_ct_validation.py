import pandas as pd
import pytest

from ml.data_validation import (
    CHEMICAL_FEATURES,
    validate_separation,
    validate_split,
)


@pytest.fixture
def valid_dataset():
    data = {feature: [1.0, 2.0] for feature in CHEMICAL_FEATURES}
    data["is_red"] = [0, 1]
    data["quality"] = [5, 7]
    return pd.DataFrame(data)


def test_valid_dataset(valid_dataset):
    validate_split(valid_dataset, "v2", "train")


def test_missing_feature(valid_dataset):
    df = valid_dataset.drop(columns="alcohol")

    with pytest.raises(ValueError, match="columns"):
        validate_split(df, "v2", "train")


def test_missing_values(valid_dataset):
    df = valid_dataset.copy()
    df.loc[0, "alcohol"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_split(df, "v2", "train")


@pytest.mark.parametrize("quality", [-1, 11, 5.5])
def test_invalid_quality(valid_dataset, quality):
    df = valid_dataset.copy()
    df["quality"] = df["quality"].astype(float)
    df.loc[0, "quality"] = quality

    with pytest.raises(ValueError, match="quality scores"):
        validate_split(df, "v2", "train")


def test_invalid_wine_type(valid_dataset):
    df = valid_dataset.copy()
    df.loc[0, "is_red"] = 3

    with pytest.raises(ValueError, match="wine type"):
        validate_split(df, "v2", "train")


def test_data_leakage(valid_dataset):
    splits = {
        "train": valid_dataset.copy(),
        "validation": valid_dataset.iloc[[0]].copy(),
        "test": valid_dataset.iloc[[1]].copy(),
    }

    with pytest.raises(ValueError, match="Data leakage"):
        validate_separation(splits)
