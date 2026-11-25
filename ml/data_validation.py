from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CHEMICAL_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET = "quality"


def expected_columns(version):
    """Return the expected columns for a dataset version."""
    if version not in ("v1", "v2"):
        raise ValueError(f"Unsupported dataset version: {version}")

    columns = CHEMICAL_FEATURES.copy()

    if version == "v2":
        columns.append("is_red")

    return columns + [TARGET]


def load_splits(version):
    """Load the processed training, validation and test datasets."""
    version_dir = DATA_DIR / version

    splits = {}

    for split in ("train", "validation", "test"):
        path = version_dir / f"{split}.csv"

        if not path.is_file():
            raise ValueError(f"Missing dataset file: {path}")

        splits[split] = pd.read_csv(path)

    return splits


def validate_split(df, version, split_name):
    """Validate the structure and values of one split."""
    required = expected_columns(version)

    if set(df.columns) != set(required):
        raise ValueError(f"{version}/{split_name}: unexpected or missing columns")

    if df.empty:
        raise ValueError(f"{version}/{split_name}: dataset is empty")

    if df.isna().any().any():
        raise ValueError(f"{version}/{split_name}: missing values detected")

    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in required):
        raise ValueError(f"{version}/{split_name}: non-numeric values detected")

    if (
        not df[CHEMICAL_FEATURES]
        .apply(lambda column: column.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all())
        .all()
    ):
        raise ValueError(f"{version}/{split_name}: non-finite values detected")

    if not df[TARGET].between(0, 10).all():
        raise ValueError(f"{version}/{split_name}: invalid quality scores")

    if not (df[TARGET] % 1 == 0).all():
        raise ValueError(f"{version}/{split_name}: non-integer quality scores")

    if version == "v2" and not df["is_red"].isin([0, 1]).all():
        raise ValueError(f"{version}/{split_name}: invalid wine type")


def validate_separation(splits):
    """Check that observations do not overlap across splits."""
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    for name_a, name_b, df_a, df_b in [
        ("train", "validation", train, validation),
        ("train", "test", train, test),
        ("validation", "test", validation, test),
    ]:
        unique_a = df_a.drop_duplicates()
        unique_b = df_b.drop_duplicates()

        overlap = unique_a.merge(unique_b, how="inner")

        if not overlap.empty:
            raise ValueError(f"Data leakage detected between {name_a} and {name_b}")


def validate_version(version):
    """Run all checks for the selected dataset version."""
    splits = load_splits(version)

    for split_name, df in splits.items():
        validate_split(df, version, split_name)

    validate_separation(splits)

    print(f"{version.upper()}: validation passed")

    for split_name, df in splits.items():
        print(f"  {split_name}: {len(df)} observations")

    return splits


def validate_cross_version_consistency():
    """Verify that V1/V2 share the same validation and test observations."""
    v1 = load_splits("v1")
    v2 = load_splits("v2")

    for split in ("validation", "test"):
        v2_without_type = v2[split].drop(columns="is_red")

        if not v1[split].reset_index(drop=True).equals(v2_without_type.reset_index(drop=True)):
            raise ValueError(f"V1 and V2 have different {split} observations")

    print("Cross-version consistency: passed")


def main():
    for version in ("v1", "v2"):
        validate_version(version)

    validate_cross_version_consistency()

    print("\nAll dataset validations passed.")


if __name__ == "__main__":
    main()
