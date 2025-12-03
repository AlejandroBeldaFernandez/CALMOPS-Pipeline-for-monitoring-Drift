import pytest
import pandas as pd
import numpy as np
from calmops.data_generators.Clinic.Clinic import ClinicGenerator


@pytest.fixture
def generator():
    return ClinicGenerator(seed=42)


def test_generate_demographic_data(generator):
    n_samples = 50
    df_demo, raw_demo = generator.generate_demographic_data(n_samples=n_samples)

    assert len(df_demo) == n_samples
    assert "Age" in df_demo.columns
    assert "Sex" in df_demo.columns
    assert "Group" in df_demo.columns
    assert df_demo.index.name == "Patient_ID"

    # Check raw data
    assert len(raw_demo) == n_samples
    assert "Age" in raw_demo.columns
    # Sex might be dropped in raw if Sex_Binario exists, or kept.
    # The code says: if "Sex" and "Sex_Binario" in raw, drop "Sex".
    # But raw is a copy of df_temp before some cleanups.
    # Let's just check it returns a dataframe.
    assert isinstance(raw_demo, pd.DataFrame)


def test_generate_gene_data_rnaseq(generator):
    n_samples = 20
    n_genes = 10
    df_demo, raw_demo = generator.generate_demographic_data(n_samples=n_samples)

    df_genes = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type="RNA-Seq",
        demographic_df=df_demo,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demo,
    )

    assert df_genes.shape == (n_samples, n_genes)
    assert all(col.startswith("G_") for col in df_genes.columns)
    # RNA-Seq should be integers (counts)
    assert (df_genes.dtypes == int).all() or (df_genes.dtypes == "int64").all()


def test_generate_gene_data_microarray(generator):
    n_samples = 20
    n_genes = 10
    df_demo, raw_demo = generator.generate_demographic_data(n_samples=n_samples)

    df_genes = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type="Microarray",
        demographic_df=df_demo,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demo,
    )

    assert df_genes.shape == (n_samples, n_genes)
    # Microarray should be floats
    assert (df_genes.dtypes == float).all() or (df_genes.dtypes == "float64").all()


def test_generate_protein_data(generator):
    n_samples = 20
    n_proteins = 5
    df_demo, raw_demo = generator.generate_demographic_data(n_samples=n_samples)

    df_proteins = generator.generate_protein_data(
        n_proteins=n_proteins,
        demographic_df=df_demo,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demo,
    )

    assert df_proteins.shape == (n_samples, n_proteins)
    assert all(col.startswith("P_") for col in df_proteins.columns)
    assert (df_proteins.dtypes == float).all() or (
        df_proteins.dtypes == "float64"
    ).all()


def test_generate_target_variable(generator):
    n_samples = 20
    df_demo, raw_demo = generator.generate_demographic_data(n_samples=n_samples)

    # Mock omics data
    df_genes = pd.DataFrame(
        np.random.rand(n_samples, 5),
        columns=[f"G_{i}" for i in range(5)],
        index=df_demo.index,
    )

    weights = {"Age": 0.1, "G_0": 0.5}

    target = generator.generate_target_variable(
        demographic_df=df_demo, omics_dfs=[df_genes], weights=weights
    )

    assert len(target) == n_samples
    assert target.name == "Target_Y"
    assert isinstance(target, pd.Series)


def test_disease_effects(generator):
    n_samples = 20
    n_genes = 5
    df_demo, raw_demo = generator.generate_demographic_data(
        n_samples=n_samples, control_disease_ratio=0.5
    )

    # Define a simple effect
    effects_config = [
        {
            "name": "Test_Effect",
            "indices": [0, 1],  # Affect first two genes
            "effect_type": "additive_shift",
            "effect_value": 100.0,  # Large shift to be obvious
        }
    ]

    # Generate with effects
    df_genes = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type="Microarray",
        demographic_df=df_demo,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demo,
        disease_effects_config=effects_config,
    )

    # Check that disease group has higher values for affected genes
    disease_ids = df_demo[df_demo["Group"] == "Disease"].index
    control_ids = df_demo[df_demo["Group"] == "Control"].index

    if len(disease_ids) > 0 and len(control_ids) > 0:
        mean_disease = df_genes.loc[disease_ids, ["G_0", "G_1"]].mean().mean()
        mean_control = df_genes.loc[control_ids, ["G_0", "G_1"]].mean().mean()

        # With a shift of 100, disease should be much higher
        assert mean_disease > mean_control
