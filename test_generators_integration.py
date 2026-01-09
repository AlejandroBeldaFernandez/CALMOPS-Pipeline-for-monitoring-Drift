import pandas as pd
import numpy as np
import os
import shutil
import warnings

warnings.filterwarnings("ignore")

from calmops.data_generators.Real.RealGenerator import RealGenerator
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from calmops.data_generators.Clinic.Clinic import ClinicGenerator

# Mock data for RealGenerator
df_mock = pd.DataFrame(
    {"feature1": np.random.rand(100), "target": np.random.choice([0, 1], 100)}
)


def test_real_generator():
    print("Testing RealGenerator...")
    gen = RealGenerator(
        data=df_mock, method="cart", target_col="target", random_state=42
    )

    drift_conf = [
        {
            "method": "inject_feature_drift",
            "params": {
                "feature_cols": ["feature1"],
                "drift_type": "add_value",
                "drift_value": 10.0,
            },
        }
    ]

    df_synth = gen.generate(
        n_samples=50, drift_injection_config=drift_conf, output_dir="test_output_real"
    )

    assert df_synth is not None
    assert "feature1" in df_synth.columns
    mean_val = df_synth["feature1"].mean()
    print(f"RealGenerator feature1 original mean ~0.5. Shifted mean: {mean_val}")
    assert mean_val > 5.0
    print("RealGenerator passed.")


def test_synthetic_generator():
    print("\nTesting SyntheticGenerator...")
    gen = SyntheticGenerator()
    from river.datasets import synth

    river_gen = synth.Agrawal(seed=42)

    drift_conf = [
        {
            "method": "inject_feature_drift",
            "params": {
                "feature_cols": ["age"],
                "drift_type": "add_value",
                "drift_value": 20.0,
            },
        }
    ]

    df = gen.generate(
        generator_instance=river_gen,
        output_path="test_output_synth",
        filename="synth_gen.csv",
        n_samples=50,
        drift_injection_config=drift_conf,
        save_dataset=False,
        generate_report=False,
    )

    mean_val = df["age"].mean()
    print(f"SyntheticGenerator age mean (expected > 60): {mean_val}")
    assert mean_val > 55.0
    print("SyntheticGenerator passed.")


def test_clinic_generator():
    print("\nTesting ClinicGenerator...")
    gen = ClinicGenerator(seed=42)

    # Test n_samples presence and basic generation
    df_genes = gen.generate_gene_data(n_genes=10, gene_type="Microarray", n_samples=50)
    assert len(df_genes) == 50
    print("ClinicGenerator generate_gene_data n_samples check passed.")

    # Test Unified API (Dashboard Usage)
    print("Testing ClinicGenerator Unified API...")
    results = gen.generate(
        n_samples=20,
        control_disease_ratio=0.5,
        gene_config={"n_genes": 10, "gene_type": "Microarray"},
        protein_config={"n_proteins": 5},
    )
    assert isinstance(results, dict)
    assert "demographics" in results
    assert "genes" in results
    assert "proteins" in results
    assert len(results["demographics"]) == 20
    print("ClinicGenerator Unified API passed.")


if __name__ == "__main__":
    if not os.path.exists("test_output_real"):
        os.makedirs("test_output_real")
    if not os.path.exists("test_output_synth"):
        os.makedirs("test_output_synth")

    try:
        test_real_generator()
        test_synthetic_generator()
        test_clinic_generator()
        print("\nAll tests passed successfully.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists("test_output_real"):
            shutil.rmtree("test_output_real")
        if os.path.exists("test_output_synth"):
            shutil.rmtree("test_output_synth")
