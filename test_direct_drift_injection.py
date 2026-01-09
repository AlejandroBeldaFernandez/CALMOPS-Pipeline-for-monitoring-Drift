import pandas as pd
import numpy as np
import shutil
import os
import sys

# Ensure calmops is in path if running from root
sys.path.append(os.getcwd())

from calmops.data_generators.Real.RealGenerator import RealGenerator
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from calmops.data_generators.configs import DateConfig
from river.datasets import synth


def test_real_generator_drift_injection():
    print("\n[REAL GENERATOR] Testing internal drift injection...")

    # 1. Create Mock Data
    data = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40] * 20,  # Mean = 30
            "salary": [30000, 40000, 50000, 60000, 70000] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        }
    )

    gen = RealGenerator(data=data, method="cart", target_col="target", random_state=42)

    # 2. Configure Drift: Add constant value to 'age'
    # We expect age mean to shift from ~30 to ~50
    drift_conf = [
        {
            "method": "inject_feature_drift",
            "params": {
                "feature_cols": ["age"],
                "drift_type": "add_value",
                "drift_value": 200.0,  # Massive drift to be obvious
                "start_index": 0,  # Apply to all generated data for clear signal
            },
        }
    ]

    # 3. Generate (One call does it all)
    print("  -> Generating 100 samples with drift embedded...")
    df_gen = gen.generate(
        n_samples=100,
        drift_injection_config=drift_conf,
        output_dir=None,  # In-memory
        save_dataset=False,
    )

    # 4. Verify
    mean_age = df_gen["age"].mean()
    print(f"  -> Original Mean Age: ~30")
    print(f"  -> Generated Mean Age: {mean_age:.2f}")

    if mean_age > 100:
        print(
            "  ✅ SUCCESS: Age shifted significantly (RealGenerator Internal Injection worked)."
        )
    else:
        print("  ❌ FAILURE: Age did not shift as expected.")
        raise AssertionError("RealGenerator drift injection failed.")


def test_synthetic_generator_dual_drift():
    print("\n[SYNTHETIC GENERATOR] Testing dual drift (Concept + Injection)...")

    # Concept A: Agrawal Function 0 (Salary < 50k -> Loan Rejected)
    gen_A = synth.Agrawal(classification_function=0, seed=42)
    # Concept B: Agrawal Function 2 (Different rule)
    gen_B = synth.Agrawal(classification_function=2, seed=42)

    gen = SyntheticGenerator(random_state=42)
    date_conf = DateConfig(start_date="2024-01-01")

    # 1. Define Injection Drift (Post-processing)
    # Let's inject NULLs into 'salary' for the last 50 samples
    drift_injection_conf = [
        {
            "method": "inject_missing_values_drift",
            "params": {
                "feature_cols": ["salary"],
                "missing_fraction": 0.5,  # 50% missing
                "start_index": 50,  # Only applied to second half
            },
        }
    ]

    # 2. Generate
    # We combine Three things here:
    # A) Concept Drift (river): Gradual transition A -> B
    # B) Data Injection (injector): Nans in salary
    # C) Metadata: Dates

    print(
        "  -> Generating 100 samples with Concept Drift (Gradual) AND Data Injection (Missing Values)..."
    )
    df_gen = gen.generate(
        generator_instance=gen_A,
        generator_instance_drift=gen_B,
        # Concept Drift Params
        n_samples=100,
        drift_type="gradual",
        position_of_drift=50,
        transition_width=20,
        # Data Injection Params
        drift_config=drift_injection_conf,
        # Metadata
        date_config=date_conf,
        output_dir=None,
        save_dataset=False,
        generate_report=False,
    )

    # 3. Verify
    # Check 1: DataFrame Shape
    assert len(df_gen) == 100

    print("DEBUG: Salary column sample (tail):")
    print(df_gen["salary"].tail(10))
    print("DEBUG: Salary column sample (head):")
    print(df_gen["salary"].head(10))

    # Check 2: Missing values injection
    missing_count = df_gen["salary"].isnull().sum()
    print(f"  -> Missing values in 'salary': {missing_count}/100")

    # Expert missing ~ 25 (50 samples * 0.5 prob)
    if missing_count > 10:
        print("  ✅ SUCCESS: Data Injection applied (Missing Values found).")
    else:
        print("  ❌ FAILURE: No missing values injected.")
        raise AssertionError("SyntheticGenerator data injection failed.")

    # Check 3: Concept drift (Harder to verify deterministically with small sample, but we confirm flow ran)
    print("  ✅ SUCCESS: SyntheticGenerator flow completed without error.")


if __name__ == "__main__":
    try:
        test_real_generator_drift_injection()
        test_synthetic_generator_dual_drift()
        print("\n🎉 ALL TESTS PASSED: Direct Drift Injection is verified!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
