import unittest
import pandas as pd
import numpy as np
import os
import shutil
from calmops.data_generators.Real.RealGenerator import RealGenerator


class TestRealGenerator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_output_real"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a dummy dataset
        np.random.seed(42)
        self.n_samples = 100
        self.data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, self.n_samples),
                "feature2": np.random.choice(["A", "B", "C"], self.n_samples),
                "target": np.random.choice([0, 1], self.n_samples),
            }
        )
        self.data["feature2"] = self.data["feature2"].astype("category")

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_resample_method(self):
        generator = RealGenerator(
            original_data=self.data,
            method="resample",
            target_column="target",
            auto_report=False,
            random_state=42,
        )
        synth_df = generator.synthesize(
            n_samples=50, output_dir=self.output_dir, save_dataset=False
        )
        self.assertEqual(len(synth_df), 50)
        self.assertTrue(set(synth_df.columns) == set(self.data.columns))

    def test_cart_method(self):
        # Use numeric data to avoid categorical encoding issues in sklearn
        numeric_data = self.data[["feature1", "target"]]
        generator = RealGenerator(
            original_data=numeric_data,
            method="cart",
            target_column="target",
            auto_report=False,
            random_state=42,
            model_params={"cart_iterations": 2},  # Reduce iterations for speed
        )
        synth_df = generator.synthesize(
            n_samples=50, output_dir=self.output_dir, save_dataset=False
        )
        self.assertEqual(len(synth_df), 50)

    def test_rf_method(self):
        # Use numeric data to avoid categorical encoding issues in sklearn
        numeric_data = self.data[["feature1", "target"]]
        generator = RealGenerator(
            original_data=numeric_data,
            method="rf",
            target_column="target",
            auto_report=False,
            random_state=42,
            model_params={"cart_iterations": 2, "rf_n_estimators": 5},
        )
        synth_df = generator.synthesize(
            n_samples=20, output_dir=self.output_dir, save_dataset=False
        )
        self.assertEqual(len(synth_df), 20)

    def test_gmm_method(self):
        # GMM only works on numeric data
        numeric_data = self.data[["feature1", "target"]]
        generator = RealGenerator(
            original_data=numeric_data,
            method="gmm",
            target_column="target",
            auto_report=False,
            random_state=42,
        )
        synth_df = generator.synthesize(
            n_samples=50, output_dir=self.output_dir, save_dataset=False
        )
        self.assertEqual(len(synth_df), 50)

    def test_custom_distributions(self):
        generator = RealGenerator(
            original_data=self.data,
            method="resample",
            target_column="target",
            auto_report=False,
            random_state=42,
        )
        custom_dist = {"feature2": {"A": 0.8, "B": 0.1, "C": 0.1}}
        synth_df = generator.synthesize(
            n_samples=100,
            output_dir=self.output_dir,
            custom_distributions=custom_dist,
            save_dataset=False,
        )
        # Check if distribution is roughly respected
        counts = synth_df["feature2"].value_counts(normalize=True)
        self.assertTrue(counts["A"] > 0.6)  # Allow some variance

    def test_date_injection(self):
        generator = RealGenerator(
            original_data=self.data,
            method="resample",
            target_column="target",
            auto_report=False,
            random_state=42,
        )
        synth_df = generator.synthesize(
            n_samples=50,
            output_dir=self.output_dir,
            date_start="2023-01-01",
            date_every=1,
            date_col="timestamp",
            save_dataset=False,
        )
        self.assertIn("timestamp", synth_df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(synth_df["timestamp"]))


if __name__ == "__main__":
    unittest.main()
