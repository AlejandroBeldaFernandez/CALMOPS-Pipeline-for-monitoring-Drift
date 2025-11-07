import numpy as np
import pandas as pd
import os
from Clinic.Clinic import ClinicGenerator

def create_gene_dataset_with_disease_effect(
    n_samples: int = 100,
    n_genes: int = 15000,
    num_disease_genes: int = 15000,
    gene_type: str = "Microarray",
    disease_effect_type: str = "additive_shift",
    disease_effect_value: float = 1.5,
    output_dir: str = "generated_datasets",
    control_disease_ratio: float = 0.5
):
    """
    Generates a gene expression dataset where a specified number of genes
    are correlated with a disease state.

    Args:
        n_samples (int): Number of patient samples to generate.
        n_genes (int): Total number of genes in the dataset.
        num_disease_genes (int): Number of genes that will show a disease effect.
        gene_type (str): Type of gene data to generate ("Microarray" or "RNA-Seq").
        disease_effect_type (str): Type of effect for disease genes
                                   ("additive_shift" for Microarray, "fold_change" for RNA-Seq).
        disease_effect_value (float or list): Magnitude of the disease effect. Can be a float or a [min, max] range.
        output_dir (str): Directory to save the generated dataset.
        control_disease_ratio (float): The ratio of control to disease patients.
    """
    print(f"Starting gene dataset generation with {n_genes} genes, {num_disease_genes} disease-correlated.")

    # Initialize the ClinicGenerator
    generator = ClinicGenerator(seed=42)

    # Generate demographic data to get the groups
    demographic_df, raw_demographic_data = generator.generate_demographic_data(
        n_samples=n_samples,
        control_disease_ratio=control_disease_ratio
    )

    # Define the disease effect using the new configuration object
    disease_effects_config = [
        {
            'name': 'disease_effect',
            'indices': list(range(num_disease_genes)),
            'effect_type': disease_effect_type,
            'effect_value': disease_effect_value
        }
    ]

    # Generate gene data using the new config
    df_genes = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type=gene_type,
        demographic_df=demographic_df,
        demographic_id_col=demographic_df.index.name,
        raw_demographic_data=raw_demographic_data,
        disease_effects_config=disease_effects_config
    )

    # Add the 'Grupo' column to the dataframe
    df_genes['Grupo'] = demographic_df['Grupo']

    # Convert 'Grupo' column to binary (0 for Control, 1 for Enfermedad)
    df_genes['Grupo'] = df_genes['Grupo'].map({'Control': 0, 'Enfermedad': 1})

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the generated gene data to a CSV file
    output_filepath = os.path.join(output_dir, f"gene_dataset_{n_genes}_genes_half_disease_evident_numerical.csv")
    df_genes.to_csv(output_filepath, index=True)

    print(f"Gene dataset saved to: {output_filepath}")
    print("Generation complete.")

if __name__ == "__main__":
    # Example usage:
    create_gene_dataset_with_disease_effect(
        n_samples=100, # Number of patients
        n_genes=50,
        num_disease_genes=50, # Half of the genes correlated with disease
        gene_type="Microarray", # Or "RNA-Seq"
        disease_effect_type="additive_shift", # "additive_shift" for Microarray, "fold_change" for RNA-Seq
        disease_effect_value=[20, 30], # Use a range for a uniform stochastic effect
        output_dir="generated_datasets_50"
    )
