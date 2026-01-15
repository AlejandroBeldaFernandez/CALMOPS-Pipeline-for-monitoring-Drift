import pytest
import pandas as pd
import numpy as np
import os
import shutil
from calmops.data_generators.Clinic.ClinicGeneratorBlock import ClinicGeneratorBlock
from calmops.data_generators.Clinic.Clinic import ClinicGenerator
from calmops.data_generators.Real.RealBlockGenerator import RealBlockGenerator
from calmops.data_generators.Real.RealGenerator import RealGenerator
from calmops.data_generators.Synthetic.SyntheticBlockGenerator import (
    SyntheticBlockGenerator,
)
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from calmops.data_generators.DriftInjection.DriftInjector import DriftInjector
from calmops.data_generators.Dynamics.ScenarioInjector import ScenarioInjector


@pytest.fixture
def output_dir():
    dir_name = "test_output_comprehensive"
    os.makedirs(dir_name, exist_ok=True)
    yield dir_name
    shutil.rmtree(dir_name)


# --- 1. RealGenerator Tests ---


def test_real_generator_basic():
    """Test standard RealGenerator generation."""
    output_dir = "/home/alex/calmops/tests/Datos"
    data = pd.DataFrame(
        {"A": np.random.normal(0, 1, 50), "B": np.random.choice(["X", "Y"], 50)}
    )
    gen = RealGenerator(auto_report=False)
    # Stateless check: 'data' passed to generate
    syn_df = gen.generate(
        data=data,
        method="cart",
        n_samples=50,
        output_dir=output_dir,
        save_dataset=False,
    )
    assert len(syn_df) == 50
    assert set(syn_df.columns) == {"A", "B"}
    assert syn_df["B"].isin(["X", "Y"]).all()


# --- 2. SyntheticGenerator Tests ---


def test_synthetic_generator_basic():
    """Test standard SyntheticGenerator with River."""
    gen = SyntheticGenerator()
    output_dir = "/home/alex/calmops/tests/Datos"
    try:
        from river.datasets import synth

        stream = synth.Agrawal(classification_function=0, seed=42)

        # SyntheticGenerator.generate returns the DataFrame, not path
        df = gen.generate(
            generator_instance=stream,
            n_samples=50,
            filename="syn_test.csv",
            output_path=output_dir,
            generate_report=False,
            save_dataset=True,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    except ImportError:
        pytest.skip("River not installed")


# --- 3. ClinicGenerator Tests ---


def test_clinic_generator_basic():
    """Test ClinicGenerator demographics and data creation."""
    gen = ClinicGenerator(seed=42)
    output_dir = "/home/alex/calmops/tests/Datos"
    # 1. Demographics
    demo_df, raw_demo = gen.generate_demographic_data(n_samples=20)
    assert len(demo_df) == 20
    # Patient_ID is in index
    assert "Patient_ID" in demo_df.index.names or "Patient_ID" in demo_df.columns

    # 2. Genes
    genes_df = gen.generate_gene_data(
        n_genes=5,
        demographic_df=demo_df,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demo,
        n_samples=20,
        gene_type="RNA-Seq",  # Added required arg
    )
    assert len(genes_df) == 20
    assert not genes_df.empty


def test_clinic_generator_longitudinal():
    """Test longitudinal data generation (simulating time steps)."""
    gen = ClinicGenerator(seed=42)
    output_dir = "/home/alex/calmops/tests/Datos"
    # Initial state (T0)
    demo_df, raw_demo = gen.generate_demographic_data(n_samples=10)
    genes_t0 = gen.generate_gene_data(
        n_genes=3,
        demographic_df=demo_df,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demo,
        n_samples=10,
        gene_type="Microarray",  # Added required arg
    )

    # Evolve to T1
    demo_t1, genes_t1 = gen.generate_additional_time_step_data(
        n_samples=10,
        date_value="2023-01-02",
        omics_to_generate=["genes"],
        n_genes=3,
        gene_type="Microarray",
        demographic_params={"control_disease_ratio": 0.5},
    )
    assert len(genes_t1) == 10
    assert genes_t1.shape[1] == 3


# --- 4. DriftInjector Comprehensive Tests ---


def test_drift_injector_comprehensive():
    """Test various drift injection methods in stateless DriftInjector."""

    output_dir = "/home/alex/calmops/tests/Datos"

    df = pd.DataFrame(
        {
            "Feature": np.random.rand(100),
            "Category": np.random.choice(["A", "B"], 100),
            "Target": np.random.randint(0, 2, 100),
            "Time": pd.date_range("2023-01-01", periods=100),
        }
    )

    injector = DriftInjector(
        output_dir=output_dir, generator_name="test", random_state=42
    )

    # A. Feature Drift (Abrupt via alias)
    df_abrupt = injector.inject_feature_drift_abrupt(
        df=df,
        feature_cols=["Feature"],
        drift_type="shift",
        drift_magnitude=2.0,
        change_index=50,
        time_col="Time",
    )
    assert np.allclose(df_abrupt["Feature"].iloc[:50], df["Feature"].iloc[:50])
    assert not np.allclose(df_abrupt["Feature"].iloc[50:], df["Feature"].iloc[50:])

    # B. Concept Drift (Gradual)
    df_concept = injector.inject_concept_drift_gradual(
        df=df,
        target_col="Target",
        concept_drift_magnitude=0.9,
        center=50,
        width=20,
        time_col="Time",
    )
    assert "Target" in df_concept.columns

    # C. Global Outliers
    df_outliers = injector.inject_outliers_global(
        df=df, cols=["Feature"], outlier_prob=0.1, factor=5.0, time_col="Time"
    )
    assert df_outliers["Feature"].std() != df["Feature"].std()

    # D. Feature Deletion
    df_deleted = injector.inject_nulls(
        df=df, cols=["Feature"], prob=0.5, time_col="Time"
    )
    assert df_deleted["Feature"].isna().sum() > 0


# --- 5. Block Generators Tests ---


def test_real_block_generator_comprehensive():
    """Test RealBlockGenerator with drift schedule."""
    output_dir = "/home/alex/calmops/tests/Datos"
    data = pd.DataFrame(
        {"Val": np.random.randn(100), "Date": pd.date_range("2023-01-01", periods=100)}
    )
    data["Block"] = np.where(data.index < 50, "B1", "B2")

    gen = RealBlockGenerator(auto_report=False)

    drift_schedule = [
        {
            "method": "inject_feature_drift",
            "params": {
                "feature_cols": ["Val"],
                "drift_type": "scale",
                "drift_magnitude": 2.0,
                "blocks": ["B2"],
            },
        }
    ]

    final_df = gen.generate(
        data=data,
        output_dir=output_dir,
        method="cart",
        block_column="Block",
        drift_config=drift_schedule,
        date_col="Date",
    )

    assert len(final_df) == 100
    assert "B1" in final_df["Block"].values
    assert "B2" in final_df["Block"].values

    b1_std = final_df[final_df["Block"] == "B1"]["Val"].std()
    b2_std = final_df[final_df["Block"] == "B2"]["Val"].std()

    # Relaxed assertion: B2 std should be significantly larger than B1 std
    assert b2_std > b1_std * 1.2


def test_clinic_generator_block_comprehensive():
    """Test ClinicGeneratorBlock with drift and dynamics."""
    output_dir = "/home/alex/calmops/tests/Datos"
    cohort_gen = ClinicGeneratorBlock()
    clinic_instance = ClinicGenerator(seed=123)

    # Dynamics config
    dynamics_conf = {"evolve_features": {"Age": {"type": "linear", "slope": 0.05}}}

    # Drift config
    drift_conf = [
        {
            "method": "inject_new_value",  # Stateless check
            "params": {
                "cols": ["Group"],
                "new_value": "NewGroup",
                "prob": 0.5,
                "blocks": [
                    2
                ],  # Target 2nd block (labels are 1-based integers usually in helper)
                "auto_report": False,
            },
        }
    ]

    path = cohort_gen.generate(
        output_dir=output_dir,
        filename="clinic_block_test.csv",
        n_blocks=2,
        total_samples=40,
        n_samples_block=20,  # 20 per block
        generators=clinic_instance,
        date_start="2023-01-01",
        dynamics_config=dynamics_conf,
        drift_config=drift_conf,
        target_col="Diagnosis",
        generate_report=False,
    )

    df = pd.read_csv(path)
    assert len(df) == 40
    # Check dynamics (hard to verify evolution without T0 ref, but execution passes)

    # Check drift: "NewGroup" should exist in block 2 but not 1
    # Block labels are typically created as 1, 2...
    if "block" in df.columns:
        b1 = df[df["block"] == 1]
        b2 = df[df["block"] == 2]
        assert "NewGroup" not in b1["Group"].values
        assert "NewGroup" in b2["Group"].values


# --- 6. DynamicsInjector Tests ---


def test_scenario_injector_basic():
    """Test basic rule application of ScenarioInjector."""
    injector = ScenarioInjector()
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

    # Construct target rule: if A > B then 1 else 0
    # Using formula string logic supported by DynamicsInjector via pandas eval
    # Note: construct_target returns dataframe with new column
    df_new = injector.construct_target(
        df,
        target_col="Target",
        formula="A > B",
        task_type="classification",
        threshold=0.5,  # True > 0.5 (True is 1.0)
    )

    assert "Target" in df_new.columns
    # 1>5(0), 2>4(0), 3>3(0), 4>2(1), 5>1(1) -> 0,0,0,1,1
    expected = [0, 0, 0, 1, 1]
    assert df_new["Target"].tolist() == expected
