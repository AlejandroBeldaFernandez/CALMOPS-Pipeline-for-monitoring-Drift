"""
Test Suite for Privacy Module
"""

import os
import shutil
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
from calmops.privacy.PrivacyReporter import PrivacyReporter

TEST_OUTPUT_DIR = "tests_output/privacy"


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    yield


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "user_id": ["U1", "U2", "U3", "U4", "U5"],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [20, 30, 40, 50, 60],
            "salary": [20000, 30000, 40000, 50000, 60000],
            "city": ["New York", "Los Angeles", "Chicago", "New York", "Miami"],
        }
    )


# =============================================================================
# Core Privacy Functions Tests
# =============================================================================


def test_pseudonymize_columns(sample_df):
    """Test pseudonymization (SHA256 hashing)."""
    print("\n[TEST] Privacy - Pseudonymize...")

    # 1. Basic hashing
    df_hashed = pseudonymize_columns(sample_df, columns=["name"])
    assert df_hashed["name"].iloc[0] != "Alice"
    # Should be hex string
    assert len(df_hashed["name"].iloc[0]) == 64

    # 2. Salt effect
    df_salt1 = pseudonymize_columns(sample_df, columns=["name"], salt="salt1")
    df_salt2 = pseudonymize_columns(sample_df, columns=["name"], salt="salt2")
    assert df_salt1["name"].iloc[0] != df_salt2["name"].iloc[0]


def test_add_laplace_noise(sample_df):
    """Test Laplace noise injection."""
    print("\n[TEST] Privacy - Laplace Noise...")

    # Large epsilon (less noise) -> should be close
    # Small epsilon (more noise) -> should be far
    # But hard to test stochastically reliably on small sample.
    # We check that values CHANGED.

    df_noise = add_laplace_noise(sample_df, columns=["salary"], epsilon=0.1)

    # Values should be different (floats now)
    assert not df_noise["salary"].equals(sample_df["salary"])
    assert pd.api.types.is_float_dtype(df_noise["salary"])

    # Mean should be roughly preserved if N is large, but N=5 is small.
    # Just checking it runs without error and modifies data.


def test_generalize_numeric_to_ranges(sample_df):
    """Test numeric generalization (binning)."""
    print("\n[TEST] Privacy - Numeric Generalization...")

    df_gen = generalize_numeric_to_ranges(sample_df, columns=["age"], num_bins=2)

    # Should be converted to strings (intervals)
    assert df_gen["age"].dtype == object
    assert df_gen["age"].nunique() <= 2

    # Check interval logic roughly
    # 20 Should be in first bin, 60 in second (or same if 1 big bin?)
    # With 2 bins [20, 40], (40, 60]
    unique_vals = df_gen["age"].unique()
    assert len(unique_vals) == 2


def test_generalize_categorical_by_mapping(sample_df):
    """Test categorical generalization (mapping)."""
    print("\n[TEST] Privacy - Categorical Generalization...")

    mapping = {
        "New York": "East Coast",
        "Miami": "East Coast",
        "Los Angeles": "West Coast",
        # Chicago not mapped -> stays Chicago
    }

    df_gen = generalize_categorical_by_mapping(
        sample_df, columns=["city"], mapping=mapping
    )

    assert df_gen["city"].iloc[0] == "East Coast"
    assert df_gen["city"].iloc[4] == "East Coast"
    assert df_gen["city"].iloc[2] == "Chicago"  # Unmapped preserved


def test_shuffle_columns(sample_df):
    """Test column shuffling."""
    print("\n[TEST] Privacy - Shuffling...")

    df_shuffled = shuffle_columns(sample_df, columns=["salary"], random_state=42)

    # Set of values represents perfectly equal distribution (just permuted)
    assert set(df_shuffled["salary"]) == set(sample_df["salary"])

    # But order should likely change (though N=5 has small chance of collision, seed=42 ensures it differs)
    assert not df_shuffled["salary"].equals(sample_df["salary"])


# =============================================================================
# Privacy Reporter Tests
# =============================================================================


def test_privacy_reporter_generation(sample_df):
    """Test generation of privacy report HTML."""
    print("\n[TEST] Privacy - Reporter HTML...")

    # Create a "private" version
    df_priv = sample_df.copy()
    df_priv = add_laplace_noise(df_priv, columns=["salary", "age"], epsilon=0.5)

    output_dir = os.path.join(TEST_OUTPUT_DIR, "report_test")

    path = PrivacyReporter.generate_privacy_report(
        original_df=sample_df,
        private_df=df_priv,
        output_dir=output_dir,
    )

    assert path is not None
    assert os.path.exists(path)
    assert "privacy_report.html" in path

    # Verify content roughly (check file size > 0)
    assert os.path.getsize(path) > 100


if __name__ == "__main__":
    import sys

    # Manual run helper
    pytest.main([__file__, "-v"])
