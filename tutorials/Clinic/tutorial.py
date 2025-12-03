import numpy as np
import pandas as pd
import scipy.stats as stats
from calmops.data_generators.Clinic.Clinic import ClinicGenerator
import os


def build_correlation_matrix(n_demo, group_sizes, correlations):
    """
    Helper to build a block correlation matrix.
    Supports fixed values or ranges (min, max) for internal correlations.
    """
    n_omics = sum(group_sizes)
    n_total = n_demo + n_omics
    matrix = np.eye(n_total)

    current_idx = n_demo
    for i, size in enumerate(group_sizes):
        end_idx = current_idx + size
        config = correlations[i]

        # Internal correlation
        if "internal" in config:
            internal_val = config["internal"]
            # Handle range (tuple or list)
            if isinstance(internal_val, (tuple, list)) and len(internal_val) == 2:
                val = np.random.uniform(internal_val[0], internal_val[1])
            else:
                val = float(internal_val)

            if val > 0:
                block = matrix[current_idx:end_idx, current_idx:end_idx]
                block[:] = val
                np.fill_diagonal(block, 1.0)
                matrix[current_idx:end_idx, current_idx:end_idx] = block

        # Demographic correlation
        if "demo_idx" in config and "demo_corr" in config:
            demo_idx = config["demo_idx"]
            corr = config["demo_corr"]
            if demo_idx is not None:
                matrix[demo_idx, current_idx:end_idx] = corr
                matrix[current_idx:end_idx, demo_idx] = corr

        current_idx = end_idx

    return matrix


def rename_features(df, prefix, group_sizes, group_names):
    """Renames features with group prefixes."""
    new_columns = []
    current_idx = 0
    for size, name in zip(group_sizes, group_names):
        for i in range(size):
            new_columns.append(f"{prefix}_{name}_{current_idx + i}")
        current_idx += size
    df.columns = new_columns
    return df


def run_tutorial():
    print("=== ClinicGenerator Tutorial ===")

    # 1. Initialize
    generator = ClinicGenerator(seed=42)
    n_samples = 1000

    # 2. Generate Demographics
    print("\nGenerating Demographics...")
    custom_demo_cols = {
        "Age": {
            "distribution": "truncnorm",
            "a": -2.0,
            "b": 2.5,
            "loc": 60,
            "scale": 10,
        },
        "Sex": {"distribution": "binom", "n": 1, "p": 0.5},
    }
    demographic_df, raw_demographic_data = generator.generate_demographic_data(
        n_samples=n_samples,
        custom_demographic_columns=custom_demo_cols,
    )

    # Identify conditioning columns (excluding ID)
    cond_cols = [
        c for c in raw_demographic_data.columns if c != "Patient_ID" and c != "Group"
    ]
    col_to_idx = {col: i for i, col in enumerate(cond_cols)}
    n_demo = len(cond_cols)

    # 3. Define Gene Scenario
    print("\nDefining Gene Scenario...")
    gene_group_sizes = [100, 200, 500]  # Group A, Group B, Noise
    gene_group_names = ["GroupA", "GroupB", "Noise"]
    n_genes = sum(gene_group_sizes)

    # Define correlations
    # Group A: Correlated with Age, Internal correlation random between 0.3 and 0.5
    # Group B: Correlated with Sex, Fixed internal correlation 0.3
    # Noise: No correlation
    gene_correlations_config = [
        {"internal": (0.3, 0.5), "demo_idx": col_to_idx.get("Age"), "demo_corr": 0.4},
        {"internal": 0.3, "demo_idx": col_to_idx.get("Sex_Binario"), "demo_corr": 0.4},
        {"internal": 0.0},
    ]

    gene_corr_matrix = build_correlation_matrix(
        n_demo, gene_group_sizes, gene_correlations_config
    )

    # 4. Generate Genes
    print("Generating Genes (Microarray)...")
    genes_df = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type="Microarray",
        demographic_df=demographic_df,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demographic_data,
        demographic_gene_correlations=gene_corr_matrix,
        n_samples=n_samples,
    )
    genes_df = rename_features(genes_df, "Gene", gene_group_sizes, gene_group_names)

    # 5. Generate Target Variable (Diagnosis)
    print("\nGenerating Target Variable (Diagnosis)...")
    # Define weights for linear combination
    weights = {
        "Age": 0.3,
        "Sex_Binario": 0.1,
    }
    # Add weights for first 10 genes of Group A
    for col in genes_df.columns[:10]:
        weights[col] = 0.05

    diagnosis = generator.generate_target_variable(
        demographic_df=raw_demographic_data,
        omics_dfs=genes_df,
        weights=weights,
        binary_threshold=0.0,  # Binarize at mean
    )
    diagnosis.name = "diagnosis"

    # 6. Save Data
    output_dir = "tutorial_output"
    os.makedirs(output_dir, exist_ok=True)

    # Add diagnosis to dataframes
    raw_demographic_data["diagnosis"] = diagnosis
    genes_df["diagnosis"] = diagnosis

    raw_demographic_data.to_csv(f"{output_dir}/demographics.csv")
    genes_df.to_csv(f"{output_dir}/genes.csv")

    print(f"\nData saved to {output_dir}/")
    print("Demographics shape:", raw_demographic_data.shape)
    print("Genes shape:", genes_df.shape)
    print("Diagnosis balance:\n", diagnosis.value_counts(normalize=True))


if __name__ == "__main__":
    run_tutorial()
