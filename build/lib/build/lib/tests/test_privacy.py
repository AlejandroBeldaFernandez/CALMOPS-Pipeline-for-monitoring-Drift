import pytest
import pandas as pd
import numpy as np
from calmops.privacy.privacy import (
    pseudonymize_columns,
    add_laplace_noise,
    generalize_numeric_to_ranges,
    generalize_categorical_by_mapping,
    shuffle_columns,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": ["A1", "B2", "C3", "D4"],
            "age": [25, 30, 35, 40],
            "salary": [50000, 60000, 70000, 80000],
            "city": ["New York", "London", "Paris", "Tokyo"],
            "category": ["A", "B", "A", "C"],
        }
    )


def test_pseudonymize_columns(sample_df):
    columns = ["id", "city"]
    salt = "salty"
    df_pseudo = pseudonymize_columns(sample_df, columns, salt)

    # Check that columns are hashed
    assert not df_pseudo["id"].equals(sample_df["id"])
    assert not df_pseudo["city"].equals(sample_df["city"])

    # Check that other columns are unchanged
    assert df_pseudo["age"].equals(sample_df["age"])

    # Check consistency
    df_pseudo_2 = pseudonymize_columns(sample_df, columns, salt)
    assert df_pseudo.equals(df_pseudo_2)

    # Check salt effect
    df_pseudo_unsalted = pseudonymize_columns(sample_df, columns, "")
    assert not df_pseudo["id"].equals(df_pseudo_unsalted["id"])


def test_add_laplace_noise(sample_df):
    columns = ["age", "salary"]
    epsilon = 0.1
    df_noisy = add_laplace_noise(sample_df, columns, epsilon)

    # Check that noise is added (values are different)
    assert not df_noisy["age"].equals(sample_df["age"])
    assert not df_noisy["salary"].equals(sample_df["salary"])

    # Check that non-numeric columns are untouched
    assert df_noisy["id"].equals(sample_df["id"])

    # Check that types are still numeric (float)
    assert pd.api.types.is_float_dtype(df_noisy["age"])
    assert pd.api.types.is_float_dtype(df_noisy["salary"])


def test_generalize_numeric_to_ranges(sample_df):
    columns = ["age"]
    num_bins = 2
    df_gen = generalize_numeric_to_ranges(sample_df, columns, num_bins)

    # Check that column is converted to string (ranges)
    assert pd.api.types.is_string_dtype(df_gen["age"]) or pd.api.types.is_object_dtype(
        df_gen["age"]
    )

    # Check that we have at most num_bins unique values
    assert df_gen["age"].nunique() <= num_bins

    # Check that other columns are unchanged
    assert df_gen["salary"].equals(sample_df["salary"])


def test_generalize_categorical_by_mapping(sample_df):
    columns = ["city"]
    mapping = {"New York": "US", "London": "UK", "Paris": "EU", "Tokyo": "Asia"}
    df_gen = generalize_categorical_by_mapping(sample_df, columns, mapping)

    # Check mapping application
    expected = pd.Series(["US", "UK", "EU", "Asia"], name="city")
    assert df_gen["city"].equals(expected)

    # Check partial mapping
    mapping_partial = {"New York": "US"}
    df_gen_partial = generalize_categorical_by_mapping(
        sample_df, columns, mapping_partial
    )
    assert df_gen_partial["city"].iloc[0] == "US"
    assert df_gen_partial["city"].iloc[1] == "London"  # Unmapped remains same


def test_shuffle_columns(sample_df):
    columns = ["salary"]
    random_state = 42
    df_shuffled = shuffle_columns(sample_df, columns, random_state)

    # Check that values are shuffled (order changed) but set of values is same
    assert not df_shuffled["salary"].equals(sample_df["salary"])
    assert set(df_shuffled["salary"]) == set(sample_df["salary"])

    # Check reproducibility
    df_shuffled_2 = shuffle_columns(sample_df, columns, random_state)
    assert df_shuffled.equals(df_shuffled_2)

    # Check other columns unchanged
    assert df_shuffled["id"].equals(sample_df["id"])
