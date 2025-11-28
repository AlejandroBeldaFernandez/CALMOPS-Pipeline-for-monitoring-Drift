import unittest
import pandas as pd
import numpy as np
import os
import shutil
from river.datasets import synth
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator


class TestSyntheticGenerator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_output_synthetic"
        os.makedirs(self.output_dir, exist_ok=True)
        self.generator = SyntheticGenerator(random_state=42)
        # Use a simple generator for testing
        self.river_gen = synth.Agrawal(classification_function=0, seed=42)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_basic_generation(self):
        df = self.generator.generate(
            generator_instance=self.river_gen,
            output_path=self.output_dir,
            filename="basic.csv",
            n_samples=100,
            save_dataset=False,
            generate_report=False,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn("target", df.columns)

    def test_save_dataset(self):
        path = self.generator.generate(
            generator_instance=self.river_gen,
            output_path=self.output_dir,
            filename="saved.csv",
            n_samples=50,
            save_dataset=True,
            generate_report=False,
        )
        self.assertTrue(os.path.exists(path))
        df = pd.read_csv(path)
        self.assertEqual(len(df), 50)

    def test_date_injection(self):
        df = self.generator.generate(
            generator_instance=self.river_gen,
            output_path=self.output_dir,
            filename="dates.csv",
            n_samples=50,
            date_start="2023-01-01",
            date_every=1,
            date_step={"days": 1},
            save_dataset=False,
            generate_report=False,
        )
        self.assertIn("timestamp", df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertEqual(df["timestamp"].iloc[0], pd.Timestamp("2023-01-01"))
        self.assertEqual(df["timestamp"].iloc[1], pd.Timestamp("2023-01-02"))

    def test_virtual_drift(self):
        df = self.generator.generate(
            generator_instance=self.river_gen,
            output_path=self.output_dir,
            filename="virtual_drift.csv",
            n_samples=100,
            drift_type="virtual_drift",
            position_of_drift=50,
            drift_options={"missing_fraction": 0.5},
            save_dataset=False,
            generate_report=False,
        )
        # Check if there are NaNs after the drift point
        self.assertTrue(df.iloc[50:].isna().any().any())
        # Check if there are no NaNs before the drift point (Agrawal shouldn't produce NaNs normally)
        self.assertFalse(df.iloc[:50].isna().any().any())

    def test_abrupt_drift(self):
        gen1 = synth.Agrawal(classification_function=0, seed=42)
        gen2 = synth.Agrawal(classification_function=1, seed=42)

        df = self.generator.generate(
            generator_instance=gen1,
            drift_generator=gen2,
            output_path=self.output_dir,
            filename="abrupt_drift.csv",
            n_samples=100,
            drift_type="abrupt",
            position_of_drift=50,
            save_dataset=False,
            generate_report=False,
        )
        self.assertEqual(len(df), 100)
        # Hard to strictly verify concept drift without statistical tests,
        # but we verify the code path executes and produces data.

    def test_gradual_drift(self):
        gen1 = synth.Agrawal(classification_function=0, seed=42)
        gen2 = synth.Agrawal(classification_function=1, seed=42)

        df = self.generator.generate(
            generator_instance=gen1,
            drift_generator=gen2,
            output_path=self.output_dir,
            filename="gradual_drift.csv",
            n_samples=100,
            drift_type="gradual",
            position_of_drift=50,
            transition_width=20,
            save_dataset=False,
            generate_report=False,
        )
        self.assertEqual(len(df), 100)

    def test_balancing(self):
        # Agrawal usually produces imbalanced data
        df = self.generator.generate(
            generator_instance=self.river_gen,
            output_path=self.output_dir,
            filename="balanced.csv",
            n_samples=100,
            balance=True,
            save_dataset=False,
            generate_report=False,
        )
        self.assertEqual(len(df), 100)
        # Check if classes are somewhat balanced (not strictly equal due to random sampling but close)
        counts = df["target"].value_counts()
        # Just ensure we have multiple classes and code ran
        self.assertTrue(len(counts) > 1)


if __name__ == "__main__":
    unittest.main()
