
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATASET_URL = "hf://datasets/mstz/wine/wine/train.csv"

OUTPUT_DIR = Path("data/processed")


def load_data():
    """Load and validate the original dataset."""
    df = pd.read_csv(DATASET_URL)

    if df.isna().any().any():
        raise ValueError("The dataset contains missing values.")

    if "is_red" not in df.columns or "quality" not in df.columns:
        raise ValueError("The dataset is missing required columns.")

    return df


def split_data(df):
    """Split the original dataset into train, validation and test sets."""
    train, remaining = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    return train.copy(), validation.copy(), test.copy()


def remove_cross_split_duplicates(train, validation, test):
    """
    Remove duplicate records across splits.

    Priority: test > validation > train.
    Duplicates within the training set are preserved.
    """
    columns = list(test.columns)

    test = test.drop_duplicates(subset=columns).copy()

    test_keys = set(
        pd.util.hash_pandas_object(test[columns], index=False)
    )

    validation_keys = pd.util.hash_pandas_object(
        validation[columns], index=False
    )

    validation = validation[
        ~validation_keys.isin(test_keys)
    ].drop_duplicates(subset=columns).copy()

    validation_keys = set(
        pd.util.hash_pandas_object(validation[columns], index=False)
    )

    train_keys = pd.util.hash_pandas_object(
        train[columns], index=False
    )

    train = train[
        ~train_keys.isin(test_keys | validation_keys)
    ].copy()

    return train, validation, test


def create_versions(train, validation, test):
    """Create V1 and V2 datasets from the same split."""

    # V1: retain training duplicates, exclude wine type.
    v1_splits = {
        "train": train.drop(columns=["is_red"]),
        "validation": validation.drop(columns=["is_red"]),
        "test": test.drop(columns=["is_red"]),
    }

    # V2: remove training duplicates and include wine type.
    v2_splits = {
        "train": train.drop_duplicates().copy(),
        "validation": validation.copy(),
        "test": test.copy(),
    }

    return {"v1": v1_splits, "v2": v2_splits}


def save_versions(versions):
    """Save each dataset version as CSV files."""

    for version, splits in versions.items():
        output_path = OUTPUT_DIR / version
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            split_df.to_csv(
                output_path / f"{split_name}.csv",
                index=False,
            )


def main():
    df = load_data()

    train, validation, test = split_data(df)

    train, validation, test = remove_cross_split_duplicates(
        train, validation, test
    )

    versions = create_versions(train, validation, test)

    save_versions(versions)

    print("Preprocessing completed.\n")

    for version, splits in versions.items():
        print(f"{version.upper()}")

        for split_name, split_df in splits.items():
            print(
                f"  {split_name}: {len(split_df)} rows "
                f"({split_df.duplicated().sum()} duplicates)"
            )

        print()


if __name__ == "__main__":
    main()
