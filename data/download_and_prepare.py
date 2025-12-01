# TODO:
# 1. Load 'bprateek/amazon_product_description' via Hugging Face Datasets
# 2. Split into 80/20 stratified by category
# 3. Save 'data/index.csv' and 'data/test.csv'

from pathlib import Path

import pandas as pd
import numpy as np

from datasets import load_dataset


DATASET_NAME = "bprateek/amazon_product_description"
SPLIT_NAME = "train"
TEST_SIZE = 0.2
RANDOM_SEED = 42
STRATIFY_COL = "category"


def canonicalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    # Ensure we always return a Data Frame with the four columns
    lower_cols = {c.lower(): c for c in df.columns}
    def pick(col_names):
        for name in col_names:
            key = name.lower()
            if key in lower_cols:
                original_col = lower_cols[key]
                return df[original_col]
        return pd.Series([""] * len(df), index=df.index)

    return pd.DataFrame(
        {
            "product_title": pick(["product_title", "product_name", "product name", "title"]),
            "about_product": pick(
                ["about_product", "about product", "product_description", "product description", "description"]
            ),
            "product_specification": pick(
                ["product_specification", "product specification", "product_information", "product information"]
            ),
            "category": pick(
                ["category", "Category", "amazon_category_and_sub_category"]
            ),
        }
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    index_path = data_dir / "index.csv"
    test_path = data_dir / "test.csv"

    ds = load_dataset(DATASET_NAME, split=SPLIT_NAME)

    # Normalize the label column name
    if "Category" in ds.column_names and "category" not in ds.column_names:
        ds = ds.rename_column("Category", "category")

    # Convert entire dataset to pandas
    full_df = ds.to_pandas()
    full_df = canonicalize_columns(full_df)

    # Manual stratified split by category
    rng = np.random.default_rng(RANDOM_SEED)
    test_indices = []

    for _, group in full_df.groupby("category"):
        # Number of test samples for this category
        n_test = int(round(len(group) * TEST_SIZE))
        if n_test <= 0:
            # For very small categories, keep all examples in the index split
            continue
        sampled = group.sample(n=n_test, random_state=RANDOM_SEED)
        test_indices.extend(sampled.index.tolist())

    test_df = full_df.loc[test_indices]
    index_df = full_df.drop(index=test_indices)

    index_df.to_csv(index_path, index=False)

    test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    main()