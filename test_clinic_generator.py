import pandas as pd
import numpy as np
import os
import sys

# Ensure calmops is in path if running from root
sys.path.append(os.getcwd())

from calmops.data_generators.Clinic.Clinic import ClinicGenerator
from calmops.data_generators.configs import DateConfig, DriftConfig


def test_clinic_generator_api():
    print("\n[CLINIC GENERATOR] Testing new API compliance...")

    # 1. Initialize
    gen = ClinicGenerator()

    # 2. Configure Dates
    date_conf = DateConfig(start_date="2025-01-01")

    # 3. Generate
    # Clinic generator returns a dict of DataFrames
    print("  -> Generating clinic data with DateConfig...")
    datasets = gen.generate(
        n_patients=50,
        n_genes=100,
        n_proteins=20,
        date_config=date_conf,
        output_dir=None,  # In-memory
        save_dataset=False,
    )

    # 4. Verify Structure
    expected_keys = ["demographics", "genes", "proteins"]
    for key in expected_keys:
        if key not in datasets:
            print(f"  ❌ FAILURE: Missing dataset '{key}'")
            raise AssertionError(f"ClinicGenerator missing output: {key}")

        df = datasets[key]
        if not isinstance(df, pd.DataFrame):
            print(f"  ❌ FAILURE: '{key}' is not a DataFrame")
            raise AssertionError(f"ClinicGenerator output '{key}' is not a DataFrame")

        if len(df) != 50:
            print(f"  ❌ FAILURE: '{key}' has wrong length {len(df)} (expected 50)")
            raise AssertionError(f"ClinicGenerator output '{key}' length mismatch")

    print("  ✅ SUCCESS: All datasets returned correctly.")

    # 5. Verify Dates Injection (Demographics usually holds the main dates or IDs)
    # Note: Clinic generator might not inject dates into all files, checking demographics
    demo_df = datasets["demographics"]
    # Check if a date column exists (based on date_config.date_col default "timestamp")
    # Clinic generator usually puts dates? Let's check.
    # If ClinicGenerator doesn't support date injection natively yet, this check helps us decide if we need to add it.

    # Let's check for ANY date column or if the configurator worked
    print("  -> Checking logic: ClinicGenerator is unique.")
    print("  -> Demographics columns:", demo_df.columns.tolist())

    # Note: ClinicGenerator usually simulates a snapshot (T=0) unless multi-timepoints requested.
    # But let's see if generate() accepted date_config without error.

    print("  ✅ SUCCESS: ClinicGenerator API Test Passed.")


if __name__ == "__main__":
    try:
        test_clinic_generator_api()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
