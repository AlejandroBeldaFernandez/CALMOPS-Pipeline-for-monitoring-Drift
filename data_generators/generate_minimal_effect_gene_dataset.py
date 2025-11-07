import numpy as np
import pandas as pd
import os
from Clinic.Clinic import ClinicGenerator

def create_minimal_effect_gene_dataset(
    n_samples: int = 100,
    n_genes: int = 15000,
    num_disease_genes: int = 10,
    num_correlated_genes: int = 10,
    correlation_value: float = 0.8,
    gene_type: str = "Microarray",
    disease_effect_type: str = "additive_shift",
    disease_effect_value: float = 0.0, # Minimal effect
    output_dir: str = "generated_datasets",
    control_disease_ratio: float = 0.5
):
    """
    Generates a gene expression dataset where a subset of genes are correlated with each other,
    and a subset of genes have a minimal correlation with a disease state.

    Args:
        n_samples (int): Number of patient samples to generate.
        n_genes (int): Total number of genes in the dataset.
        num_disease_genes (int): Number of genes that will show a disease effect.
        num_correlated_genes (int): Number of genes in the correlated group.
        correlation_value (float): Correlation value for genes within the group.
        gene_type (str): Type of gene data to generate ("Microarray" or "RNA-Seq").
        disease_effect_type (str): Type of effect for disease genes
                                   ("additive_shift" for Microarray, "fold_change" for RNA-Seq).
        disease_effect_value (float or list): Magnitude of the disease effect. Can be a float or a [min, max] range.
        output_dir (str): Directory to save the generated dataset.
    """
    print(f"Starting gene dataset generation with {n_genes} genes...")

    # Initialize the ClinicGenerator
    generator = ClinicGenerator(seed=42)

    # Generate demographic data to get the groups
    demographic_df, raw_demographic_data = generator.generate_demographic_data(
        n_samples=n_samples,
        control_disease_ratio=control_disease_ratio
    )

    # --- 1. Define correlation matrix ---
    correlation_matrix = np.identity(n_genes)
    if num_correlated_genes > 0 and correlation_value != 0:
        print(f"Introducing correlation for a group of {num_correlated_genes} genes.")
        corr_block = np.full((num_correlated_genes, num_correlated_genes), correlation_value)
        np.fill_diagonal(corr_block, 1.0)
        # Place the block at the beginning of the matrix
        correlation_matrix[0:num_correlated_genes, 0:num_correlated_genes] = corr_block

    # Define the disease effect using the new configuration object
    disease_effects_config = [
        {
            'name': 'disease_effect',
            'indices': list(range(num_disease_genes)),
            'effect_type': disease_effect_type,
            'effect_value': disease_effect_value
        }
    ]

    # Generate gene data
    df_genes = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type=gene_type,
        demographic_df=demographic_df,
        demographic_id_col=demographic_df.index.name,
        raw_demographic_data=raw_demographic_data,
        gene_correlations=correlation_matrix,
        disease_effects_config=disease_effects_config
    )

    # Add the 'Grupo' column to the dataframe
    df_genes['Grupo'] = demographic_df['Grupo']

    # Convert 'Grupo' column to binary (0 for Control, 1 for Enfermedad)
    df_genes['Grupo'] = df_genes['Grupo'].map({'Control': 0, 'Enfermedad': 1})

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the generated gene data to a CSV file
    output_filepath = os.path.join(output_dir, f"gene_dataset_{n_genes}_genes_minimal_disease_evident_numerical.csv")
    df_genes.to_csv(output_filepath, index=True)

    print(f"Gene dataset saved to: {output_filepath}")
    print("Generation complete.")

if __name__ == "__main__":
    # Example usage:
    create_minimal_effect_gene_dataset(
        n_samples=100,
        n_genes=50,
        num_disease_genes=10,
        num_correlated_genes=10,
        correlation_value=0.8,
        gene_type="Microarray",
        disease_effect_type="additive_shift",
        disease_effect_value=[0.05, 0.15], # Use a range for a minimal, uniform stochastic effect
        output_dir="generated_datasets_50"
    )
