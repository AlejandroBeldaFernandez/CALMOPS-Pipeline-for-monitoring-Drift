import unittest
import pandas as pd
import numpy as np
from calmops.data_generators.Scenario.DynamicsInjector import DynamicsInjector


class TestDynamicsInjector(unittest.TestCase):
    def setUp(self):
        self.modifier = DynamicsInjector(seed=42)
        self.df = pd.DataFrame(
            {
                "A": np.full(1000, 10.0),
                "B": np.arange(1000, dtype=float),
                "C": np.random.normal(0, 1, 1000),
            }
        )

    def test_evolve_features_linear(self):
        config = {"A": {"type": "linear", "slope": 1.0, "intercept": 0.0}}
        df_evolved = self.modifier.evolve_features(self.df, config)

        # A starts at 10. Slope 1.0 means A[i] = 10 + i
        expected = 10.0 + np.arange(1000)
        np.testing.assert_array_almost_equal(df_evolved["A"].values, expected)

    def test_evolve_features_cycle(self):
        config = {"A": {"type": "cycle", "period": 10, "amplitude": 5.0}}
        df_evolved = self.modifier.evolve_features(self.df, config)

        # Check period
        # A[0] = 10 + 0 = 10
        # A[10] = 10 + 0 = 10
        # A[2.5] (index 2 or 3) -> sin(pi/2) = 1 -> 10+5 = 15
        self.assertAlmostEqual(df_evolved["A"].iloc[0], 10.0)
        self.assertAlmostEqual(df_evolved["A"].iloc[10], 10.0)

    def test_construct_target_regression(self):
        # Y = 2*A + B
        # A is 10, B is 0..99
        # Y = 20 + B
        df_target = self.modifier.construct_target(
            self.df, target_col="Y", formula="2 * A + B", task_type="regression"
        )
        expected = 20.0 + self.df["B"].values
        np.testing.assert_array_almost_equal(df_target["Y"].values, expected)

    def test_construct_target_classification(self):
        # Y = 1 if B > 50 else 0
        df_target = self.modifier.construct_target(
            self.df,
            target_col="Y_class",
            formula="B",
            task_type="classification",
            threshold=50.0,
        )
        self.assertEqual(df_target["Y_class"].iloc[0], 0)  # B=0
        self.assertEqual(df_target["Y_class"].iloc[50], 0)  # B=50 (not > 50)
        self.assertEqual(df_target["Y_class"].iloc[51], 1)  # B=51

    def test_construct_target_noise(self):
        # Y = A (constant 10) + noise
        df_target = self.modifier.construct_target(
            self.df,
            target_col="Y_noise",
            formula="A",
            noise_std=1.0,
            task_type="regression",
        )
        # Check variance is close to 1
        std = df_target["Y_noise"].std()
        self.assertTrue(0.8 < std < 1.2, f"Std {std} not close to 1.0")

    def test_time_col(self):
        # Use B as time (0..99)
        # Evolve C linearly with slope 1 based on B
        # C_new = C_old + 1*B
        config = {"C": {"type": "linear", "slope": 1.0}}
        df_evolved = self.modifier.evolve_features(self.df, config, time_col="B")

        expected = self.df["C"] + self.df["B"]
        np.testing.assert_array_almost_equal(df_evolved["C"].values, expected.values)


if __name__ == "__main__":
    unittest.main()
