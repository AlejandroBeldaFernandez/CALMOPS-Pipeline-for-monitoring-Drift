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
        if "internal" in config:
            internal_val = config["internal"]

            # Handle range (tuple or list)
            if isinstance(internal_val, (tuple, list)) and len(internal_val) == 2:
                # Sample a random value from the range [min, max]
                val = np.random.uniform(internal_val[0], internal_val[1])
                print(
                    f"  Group {i} internal correlation sampled from {internal_val}: {val:.3f}"
                )
            else:
                val = float(internal_val)

            if val > 0:
                block = matrix[current_idx:end_idx, current_idx:end_idx]
                # Set off-diagonal elements to internal correlation
                block[:] = val
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


def rename_features(df, prefix, group_sizes, group_names, relevant_groups):
    """
    Renames features in the dataframe based on group membership and relevance.

    Args:
        df (pd.DataFrame): The dataframe with original feature names.
        prefix (str): Prefix for the new names (e.g., "Gene", "Prot").
        group_sizes (list): List of sizes for each group.
        group_names (list): List of names for each group.
        relevant_groups (list): List of group names that are relevant (will get _Rel suffix).

    Returns:
        pd.DataFrame: Dataframe with renamed columns.
        dict: Dictionary mapping group name to list of new column names.
    """
    new_columns = []
    group_col_map = {name: [] for name in group_names}

    current_idx = 0
    for size, name in zip(group_sizes, group_names):
        is_relevant = name in relevant_groups
        rel_suffix = "_Rel" if is_relevant else ""

        for i in range(size):
            new_name = f"{prefix}_{name}_{current_idx + i}{rel_suffix}"
            new_columns.append(new_name)
            group_col_map[name].append(new_name)

        current_idx += size

    if len(new_columns) != len(df.columns):
        raise ValueError(
            f"Generated {len(new_columns)} names but dataframe has {len(df.columns)} columns."
        )

    df.columns = new_columns
    return df, group_col_map


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
    gene_group_names = ["GroupA", "GroupB", "GroupD", "Ruido"]
    gene_relevant_groups = ["GroupA", "GroupB"]
    n_genes = sum(gene_group_sizes)

    # Correlations
    # We need to use the indices from col_to_idx
    gene_correlations_config = [
        {
            "internal": (
                0.3,
                0.6,
            ),  # Group A: Random internal correlation between 0.3 and 0.6
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

    # Rename Genes
    print("Renaming Genes...")
    genes_df, gene_group_cols = rename_features(
        genes_df,
        prefix="Gene",
        group_sizes=gene_group_sizes,
        group_names=gene_group_names,
        relevant_groups=gene_relevant_groups,
    )
    print("Example Gene Names:")
    for group in gene_group_names:
        print(f"  {group}: {gene_group_cols[group][:3]} ...")

    # 4. Verify Gene Correlations
    print("Verifying Gene Correlations...")
    # Create combined DF for check
    check_df = pd.concat([raw_demographic_data, genes_df], axis=1)

    # Check Group A vs Age
    group_a_cols = gene_group_cols["GroupA"]
    if "Age" in check_df.columns:
        corr_a_age = check_df[group_a_cols].corrwith(check_df["Age"]).mean()
        print(f"Avg Correlation Group A vs Age (Expected ~0.4): {corr_a_age:.3f}")

    # Check Group B vs Sex
    group_b_cols = gene_group_cols["GroupB"]
    if "Sex_Binario" in check_df.columns:
        corr_b_sex = check_df[group_b_cols].corrwith(check_df["Sex_Binario"]).mean()
        print(f"Avg Correlation Group B vs Sex (Expected ~0.4): {corr_b_sex:.3f}")

    # 5. Generate Target Y (Diagnosis)
    # Y = 0.2*A + 0.5*B + 0.3*Age + 0.1*Sex
    print("Generating Target Y (Diagnosis)...")

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

    # Generate continuous score first, then binarize
    y_score = generator.generate_target_variable(
        demographic_df=raw_demographic_data,
        omics_dfs=genes_df,
        weights=weights_genes,
        binary_threshold=0.0,  # Binarize at mean 0
    )
    y_score.name = "diagnosis"
    print("Target Y (Diagnosis) generated. Mean:", y_score.mean())
    print("Class Proportions:")
    print(y_score.value_counts(normalize=True))

    # Add diagnosis to all dataframes
    raw_demographic_data["diagnosis"] = y_score
    genes_df["diagnosis"] = y_score
    # prot_df will be created later, so we add diagnosis to it after its creation
    # For now, we'll just ensure it's available for the protein generation step if needed.

    # Remove old random groups if present
    if "Binary_Group" in raw_demographic_data.columns:
        raw_demographic_data.drop(columns=["Binary_Group"], inplace=True)
    if "Group" in raw_demographic_data.columns:
        raw_demographic_data.drop(columns=["Group"], inplace=True)

    # 6. Define Protein Scenario
    # Groups: A (1500), B (100), D (2000), Ruido (1250)
    prot_group_sizes = [1500, 100, 2000, 1250]
    prot_group_names = ["GroupA", "GroupB", "GroupD", "Ruido"]
    prot_relevant_groups = ["GroupA", "GroupB"]
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

    # Rename Proteins
    print("Renaming Proteins...")
    prot_df, prot_group_cols = rename_features(
        prot_df,
        prefix="Prot",
        group_sizes=prot_group_sizes,
        group_names=prot_group_names,
        relevant_groups=prot_relevant_groups,
    )

    # Add diagnosis to proteins (it was generated separately)
    prot_df["diagnosis"] = y_score

    print("Example Protein Names:")
    for group in prot_group_names:
        print(f"  {group}: {prot_group_cols[group][:3]} ...")

    # 8. Verify Protein Correlations
    print("Verifying Protein Correlations...")
    check_df_prot = pd.concat([raw_demographic_data, prot_df], axis=1)

    # Check Group A vs Age
    group_a_prot_cols = prot_group_cols["GroupA"]
    if "Age" in check_df_prot.columns:
        corr_a_age_prot = (
            check_df_prot[group_a_prot_cols].corrwith(check_df_prot["Age"]).mean()
        )
        print(
            f"Avg Correlation Protein Group A vs Age (Expected ~0.4): {corr_a_age_prot:.3f}"
        )

    # Check Group B vs Sex
    # Group A is 0-1499. Group B is 1500-1599.
    group_b_prot_cols = prot_group_cols["GroupB"]
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

    # Ensure index name is set for CSV saving
    genes_df.index.name = "Patient_ID"
    prot_df.index.name = "Patient_ID"

    print(f"Saving datasets to {output_dir}...")
    raw_demographic_data.to_csv(os.path.join(output_dir, "demographics.csv"))
    genes_df.to_csv(os.path.join(output_dir, "genes.csv"))
    prot_df.to_csv(os.path.join(output_dir, "proteins.csv"))

    # 9. Verify Diagnosis Dependencies
    print("Verifying Diagnosis Dependencies...")
    # Combine Y with features for correlation check
    # Drop diagnosis from genes_df to avoid duplicate columns (it's already in raw_demographic_data)
    analysis_df = pd.concat(
        [raw_demographic_data, genes_df.drop(columns=["diagnosis"], errors="ignore")],
        axis=1,
    )

    # Correlation with Age
    if "Age" in analysis_df.columns:
        corr_y_age = analysis_df["diagnosis"].corr(analysis_df["Age"])
        print(f"Correlation Diagnosis vs Age (Expected positive): {corr_y_age:.3f}")

    # Correlation with Sex
    if "Sex_Binario" in analysis_df.columns:
        corr_y_sex = analysis_df["diagnosis"].corr(analysis_df["Sex_Binario"])
        print(f"Correlation Diagnosis vs Sex (Expected positive): {corr_y_sex:.3f}")

    # Correlation with Group A Mean (using renamed columns)
    group_a_cols = gene_group_cols["GroupA"]
    mean_group_a = analysis_df[group_a_cols].mean(axis=1)
    corr_y_group_a = analysis_df["diagnosis"].corr(mean_group_a)
    print(
        f"Correlation Diagnosis vs Mean(Group A Genes) (Expected positive): {corr_y_group_a:.3f}"
    )

    # Correlation with Group B Mean (using renamed columns)
    group_b_cols = gene_group_cols["GroupB"]
    mean_group_b = analysis_df[group_b_cols].mean(axis=1)
    corr_y_group_b = analysis_df["diagnosis"].corr(mean_group_b)
    print(
        f"Correlation Diagnosis vs Mean(Group B Genes) (Expected positive): {corr_y_group_b:.3f}"
    )

    # Correlation with Noise (Control check)
    if "Ruido" in gene_group_cols:
        group_noise_cols = gene_group_cols["Ruido"]
        mean_group_noise = analysis_df[group_noise_cols].mean(axis=1)
        corr_y_noise = analysis_df["diagnosis"].corr(mean_group_noise)
        print(
            f"Correlation Diagnosis vs Mean(Noise Genes) (Expected ~0): {corr_y_noise:.3f}"
        )

    print("Verification Complete. Files saved.")


if __name__ == "__main__":
    verify_scenario()
