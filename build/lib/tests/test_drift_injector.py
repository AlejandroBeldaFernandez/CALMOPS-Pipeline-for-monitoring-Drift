import unittest
import pandas as pd
import numpy as np
import os
import shutil
from calmops.data_generators.DriftInjection.DriftInjector import DriftInjector


class TestDriftInjector(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_output_drift"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a dummy dataset
        np.random.seed(42)
        self.n_samples = 100
        self.data = pd.DataFrame(
            {
                "feature1": np.random.normal(10, 1, self.n_samples),
                "feature2": np.random.choice(["A", "B"], self.n_samples),
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=self.n_samples, freq="D"
                ),
            }
        )
        self.injector = DriftInjector(
            original_df=self.data,
            output_dir=self.output_dir,
            generator_name="test_injector",
            time_col="timestamp",
            random_state=42,
        )

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_inject_feature_drift_gaussian(self):
        drifted = self.injector.inject_feature_drift(
            df=self.data,
            feature_cols=["feature1"],
            drift_type="gaussian_noise",
            drift_magnitude=0.5,
            auto_report=False,
        )
        # Check that values have changed
        self.assertFalse(drifted["feature1"].equals(self.data["feature1"]))
        # Check that other columns are unchanged
        self.assertTrue(drifted["feature2"].equals(self.data["feature2"]))

    def test_inject_feature_drift_shift(self):
        drifted = self.injector.inject_feature_drift(
            df=self.data,
            feature_cols=["feature1"],
            drift_type="shift",
            drift_magnitude=0.1,  # 10% shift
            auto_report=False,
        )
        # Mean should be different
        self.assertNotAlmostEqual(
            drifted["feature1"].mean(), self.data["feature1"].mean()
        )

    def test_inject_feature_drift_categorical(self):
        drifted = self.injector.inject_feature_drift(
            df=self.data,
            feature_cols=["feature2"],
            drift_type="gaussian_noise",  # Used for categorical flip probability
            drift_magnitude=1.0,  # Force changes
            auto_report=False,
        )
        # Check that some values changed
        self.assertFalse(drifted["feature2"].equals(self.data["feature2"]))

    def test_targeting_by_index(self):
        drifted = self.injector.inject_feature_drift(
            df=self.data,
            feature_cols=["feature1"],
            drift_type="shift",
            drift_magnitude=1.0,
            start_index=50,
            auto_report=False,
        )
        # First 50 should be same
        self.assertTrue(
            drifted["feature1"].iloc[:50].equals(self.data["feature1"].iloc[:50])
        )
        # Rest should be different
        self.assertFalse(
            drifted["feature1"].iloc[50:].equals(self.data["feature1"].iloc[50:])
        )

    def test_targeting_by_time(self):
        drifted = self.injector.inject_feature_drift(
            df=self.data,
            feature_cols=["feature1"],
            drift_type="shift",
            drift_magnitude=1.0,
            time_start="2023-02-01",
            auto_report=False,
        )
        # Check based on date
        mask = self.data["timestamp"] >= "2023-02-01"
        # Unaffected rows
        self.assertTrue(
            drifted.loc[~mask, "feature1"].equals(self.data.loc[~mask, "feature1"])
        )
        # Affected rows
        self.assertFalse(
            drifted.loc[mask, "feature1"].equals(self.data.loc[mask, "feature1"])
        )

    def test_gradual_drift(self):
        drifted = self.injector.inject_feature_drift_gradual(
            df=self.data,
            feature_cols=["feature1"],
            drift_type="shift",
            drift_magnitude=1.0,
            center=50,
            width=20,
            auto_report=False,
        )
        self.assertFalse(drifted["feature1"].equals(self.data["feature1"]))


if __name__ == "__main__":
    unittest.main()
