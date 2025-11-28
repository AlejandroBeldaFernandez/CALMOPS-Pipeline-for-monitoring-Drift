import numpy as np
import pandas as pd
import scipy.stats as stats
from calmops.data_generators.Clinic.Clinic import ClinicGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import os


def build_correlation_matrix(n_demo, group_sizes, correlations):
    """
    Builds a block correlation matrix.

    Args:
        n_demo (int): Number of demographic variables.
        group_sizes (list): List of sizes for each omics group.
        correlations (list): List of dicts defining correlations for each group.
                             Each dict: {'internal': float, 'demo_idx': int, 'demo_corr': float}
    """
    n_omics = sum(group_sizes)
    n_total = n_demo + n_omics
    matrix = np.eye(n_total)

    current_idx = n_demo
    for i, size in enumerate(group_sizes):
        end_idx = current_idx + size
        config = correlations[i]

        # Internal correlation
        if "internal" in config and config["internal"] > 0:
            block = matrix[current_idx:end_idx, current_idx:end_idx]
            # Set off-diagonal elements to internal correlation
            block[:] = config["internal"]
            np.fill_diagonal(block, 1.0)
            matrix[current_idx:end_idx, current_idx:end_idx] = block

        # Demographic correlation
        if "demo_idx" in config and "demo_corr" in config:
            demo_idx = config["demo_idx"]
            corr = config["demo_corr"]
            if demo_idx is not None:
                # Set correlation between demo var and group vars
                matrix[demo_idx, current_idx:end_idx] = corr
                matrix[current_idx:end_idx, demo_idx] = corr

        current_idx = end_idx

    return matrix


def verify_scenario():
    print("Initializing ClinicGenerator...")
    generator = ClinicGenerator(seed=42)

    # 1. Generate Demographics with RIN and Lote
    print("Generating Demographics...")
    n_samples = 4000

    # Define custom columns for RIN and Lote
    # Age: Mean 60, Range [40, 85]. Assuming sigma=10.
    # a = (40-60)/10 = -2.0, b = (85-60)/10 = 2.5

    # RIN: Mean 6, Range [0, 10]. Assuming sigma=2.
    # a = (0-6)/2 = -3.0, b = (10-6)/2 = 2.0

    # Sex: "Normal umbral 0" -> Binomial p=0.5 (Latent normal > 0)

    custom_demo_cols = {
        "Age": {
            "distribution": "truncnorm",
            "a": -2.0,
            "b": 2.5,
            "loc": 60,
            "scale": 10,
        },
        "RIN": {"distribution": "truncnorm", "a": -3.0, "b": 2.0, "loc": 6, "scale": 2},
        "Lote": {"distribution": "randint", "low": 1, "high": 5},
        "Sex": {"distribution": "binom", "n": 1, "p": 0.5},
    }

    demographic_df, raw_demographic_data = generator.generate_demographic_data(
        n_samples=n_samples,
        control_disease_ratio=0.5,
        custom_demographic_columns=custom_demo_cols,
    )
    print("Demographic columns:", raw_demographic_data.columns)

    # Identify conditioning columns (same logic as in Clinic.py)
    # Clinic.py excludes 'Patient_ID' and 'Group'
    cond_cols = [
        c for c in raw_demographic_data.columns if c != "Patient_ID" and c != "Group"
    ]
    print(f"Conditioning columns used: {cond_cols}")
    n_demo = len(cond_cols)

    # Map column names to indices for easier config
    col_to_idx = {col: i for i, col in enumerate(cond_cols)}

    # 2. Define Gene Scenario
    # Groups: A (450), B (670), D (1000), Ruido (2000)
    gene_group_sizes = [450, 670, 1000, 2000]
    n_genes = sum(gene_group_sizes)

    # Correlations
    # We need to use the indices from col_to_idx
    gene_correlations_config = [
        {
            "internal": 0.45,
            "demo_idx": col_to_idx.get("Age"),
            "demo_corr": 0.4,
        },  # Group A: Age
        {
            "internal": 0.3,
            "demo_idx": col_to_idx.get("Sex_Binario"),
            "demo_corr": 0.4,
        },  # Group B: Sex
        {"internal": 0.3},  # Group D
        {"internal": 0.0},  # Ruido
    ]

    print(
        f"Building Gene Correlation Matrix for {n_demo} demo vars and {n_genes} genes..."
    )
    gene_corr_matrix = build_correlation_matrix(
        n_demo=n_demo,
        group_sizes=gene_group_sizes,
        correlations=gene_correlations_config,
    )

    # 3. Generate Genes
    print("Generating Genes...")
    genes_df = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type="Microarray",
        demographic_df=demographic_df,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demographic_data,
        demographic_gene_correlations=gene_corr_matrix,
        n_samples=n_samples,
    )

    # 4. Verify Gene Correlations
    print("Verifying Gene Correlations...")
    # Create combined DF for check
    check_df = pd.concat([raw_demographic_data, genes_df], axis=1)

    # Check Group A vs Age
    group_a_cols = genes_df.columns[:450]
    if "Age" in check_df.columns:
        corr_a_age = check_df[group_a_cols].corrwith(check_df["Age"]).mean()
        print(f"Avg Correlation Group A vs Age (Expected ~0.4): {corr_a_age:.3f}")

    # Check Group B vs Sex
    group_b_cols = genes_df.columns[450 : 450 + 670]
    if "Sex_Binario" in check_df.columns:
        corr_b_sex = check_df[group_b_cols].corrwith(check_df["Sex_Binario"]).mean()
        print(f"Avg Correlation Group B vs Sex (Expected ~0.4): {corr_b_sex:.3f}")

    # 5. Generate Target Y for Genes
    # Y = 0.2*A + 0.5*B + 0.3*Age + 0.1*Sex
    print("Generating Target Y (Genes)...")

    weights_genes = {}
    if "Age" in raw_demographic_data.columns:
        weights_genes["Age"] = 0.3
    if "Sex_Binario" in raw_demographic_data.columns:
        weights_genes["Sex_Binario"] = 0.1

    # Add Group A weights
    for col in group_a_cols:
        weights_genes[col] = 0.2 / len(group_a_cols)
    # Add Group B weights
    for col in group_b_cols:
        weights_genes[col] = 0.5 / len(group_b_cols)

    y_genes = generator.generate_target_variable(
        demographic_df=raw_demographic_data, omics_dfs=genes_df, weights=weights_genes
    )
    print("Target Y (Genes) generated. Mean:", y_genes.mean(), "Std:", y_genes.std())

    # 6. Define Protein Scenario
    # Groups: A (1500), B (100), D (2000), Ruido (1250)
    prot_group_sizes = [1500, 100, 2000, 1250]
    n_proteins = sum(prot_group_sizes)

    # Correlations: Match Genes structure
    # A: Age, B: Sex, D: Internal, Noise: 0
    prot_correlations_config = [
        {
            "internal": 0.45,
            "demo_idx": col_to_idx.get("Age"),
            "demo_corr": 0.4,
        },  # Group A: Age
        {
            "internal": 0.3,
            "demo_idx": col_to_idx.get("Sex_Binario"),
            "demo_corr": 0.4,
        },  # Group B: Sex
        {"internal": 0.3},  # Group D
        {"internal": 0.0},  # Ruido
    ]

    print(
        f"Building Protein Correlation Matrix for {n_demo} demo vars and {n_proteins} proteins..."
    )
    prot_corr_matrix = build_correlation_matrix(
        n_demo=n_demo,
        group_sizes=prot_group_sizes,
        correlations=prot_correlations_config,
    )

    # 7. Generate Proteins
    print("Generating Proteins...")
    prot_df = generator.generate_protein_data(
        n_proteins=n_proteins,
        demographic_df=demographic_df,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demographic_data,
        demographic_protein_correlations=prot_corr_matrix,
        n_samples=n_samples,
    )

    # 8. Verify Protein Correlations
    print("Verifying Protein Correlations...")
    check_df_prot = pd.concat([raw_demographic_data, prot_df], axis=1)

    # Check Group B vs Sex
    # Group A is 0-1499. Group B is 1500-1599.
    group_b_prot_cols = prot_df.columns[1500:1600]
    if "Sex_Binario" in check_df_prot.columns:
        corr_b_sex_prot = (
            check_df_prot[group_b_prot_cols]
            .corrwith(check_df_prot["Sex_Binario"])
            .mean()
        )
        print(
            f"Avg Correlation Protein Group B vs Sex (Expected ~0.4): {corr_b_sex_prot:.3f}"
        )

    # Save Datasets
    output_dir = "tests/output_verification"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving datasets to {output_dir}...")
    raw_demographic_data.to_csv(os.path.join(output_dir, "demographics.csv"))
    genes_df.to_csv(os.path.join(output_dir, "genes.csv"))
    prot_df.to_csv(os.path.join(output_dir, "proteins.csv"))

    # Save Target Y
    pd.DataFrame(y_genes, columns=["Target_Y"]).to_csv(
        os.path.join(output_dir, "target_y_genes.csv")
    )

    print("Verification Complete. Files saved.")


if __name__ == "__main__":
    verify_scenario()
