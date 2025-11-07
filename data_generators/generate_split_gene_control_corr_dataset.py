import numpy as np
import pandas as pd
import os
from Clinic.Clinic import ClinicGenerator

def create_split_gene_correlated_dataset(
    n_samples: int = 100,
    n_genes: int = 50,
    num_disease_genes_1: int = 20,
    num_disease_genes_2: int = 10,
    corr_group_1: float = 0.6,
    corr_group_2: float = 0.7,
    gene_type: str = "Microarray",
    disease_effect_type: str = "additive_shift",
    disease_effect_type_2: str = None, 
    disease_effect_value: float = 1.5,
    disease_effect_value_2: float = None, 
    num_patients_subtype_1: int = 20, # New: Number of patients in the first disease subtype
    output_dir: str = "generated_datasets",
    control_disease_ratio: float = 0.5
):
    """
    Generates a gene expression dataset with two independent, correlated gene modules.
    It now supports creating disease subtypes, where different subsets of disease patients
    have different molecular effects.

    Args:
        n_samples (int): Number of patient samples to generate.
        n_genes (int): Total number of genes in the dataset.
        num_disease_genes_1 (int): Number of genes in the first group.
        num_disease_genes_2 (int): Number of genes in the second group.
        corr_group_1 (float): Correlation value for genes within the first group.
        corr_group_2 (float): Correlation value for genes within the second group.
        gene_type (str): Type of gene data to generate ("Microarray" or "RNA-Seq").
        disease_effect_type (str): Type of effect for the first group of disease genes.
        disease_effect_type_2 (str, optional): Type of effect for the second group. Defaults to `disease_effect_type`.
        disease_effect_value (float or list): Magnitude of the disease effect for the first group.
        disease_effect_value_2 (float or list, optional): Magnitude of the disease effect for the second group.
        num_patients_subtype_1 (int): Number of disease patients to assign to the first subtype (affecting module 1).
        output_dir (str): Directory to save the generated dataset.
    """
    total_affected_genes = num_disease_genes_1 + num_disease_genes_2
    if total_affected_genes > n_genes:
        raise ValueError("The sum of disease genes cannot be greater than the total number of genes.")

    # Default for second effect type is the same as the first
    if disease_effect_type_2 is None:
        disease_effect_type_2 = disease_effect_type

    if disease_effect_value_2 is None:
        if isinstance(disease_effect_value, list):
            if disease_effect_type_2 == 'fold_change':
                 disease_effect_value_2 = [1/x for x in reversed(disease_effect_value)]
            else:
                 disease_effect_value_2 = [-x for x in reversed(disease_effect_value)]
        else:
            if disease_effect_type_2 == 'fold_change':
                disease_effect_value_2 = 1 / disease_effect_value if disease_effect_value != 0 else 1
            else:
                disease_effect_value_2 = -disease_effect_value

    print(f"Starting gene dataset generation with {n_genes} genes...")

    generator = ClinicGenerator(seed=42)

    demographic_df, raw_demographic_data = generator.generate_demographic_data(
        n_samples=n_samples,
        control_disease_ratio=control_disease_ratio
    )

    # --- 1. Define Gene Modules and Block-Diagonal Correlation Matrix ---
    module_1_indices = list(range(num_disease_genes_1))
    module_2_start_index = num_disease_genes_1
    module_2_indices = list(range(module_2_start_index, module_2_start_index + num_disease_genes_2))

    correlation_matrix = np.identity(n_genes)

    def fill_block(matrix, indices, corr_value):
        for i in indices:
            for j in indices:
                if i != j:
                    matrix[i, j] = corr_value

    if num_disease_genes_1 > 0:
        fill_block(correlation_matrix, module_1_indices, corr_group_1)

    if num_disease_genes_2 > 0:
        fill_block(correlation_matrix, module_2_indices, corr_group_2)

    # --- 2. Define Disease Heterogeneity (Subtypes and Effects) ---
    print(f"Defining disease subtypes: {num_patients_subtype_1} patients for module 1, remainder for module 2.")
    disease_effects_config = {
        'patient_subgroups': [
            {
                'name': 'Subtype_A',
                'count': num_patients_subtype_1,
                'apply_effects': ['module_1_effect']
            },
            {
                'name': 'Subtype_B',
                'remainder': True,
                'apply_effects': ['module_2_effect']
            }
        ],
        'effects': {
            'module_1_effect': {
                'indices': module_1_indices,
                'effect_type': disease_effect_type,
                'effect_value': disease_effect_value
            },
            'module_2_effect': {
                'indices': module_2_indices,
                'effect_type': disease_effect_type_2,
                'effect_value': disease_effect_value_2
            }
        }
    }

    # --- 3. Generate all gene data in a single call ---
    print("Generating gene data with disease subtypes...")
    df_genes_final = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type=gene_type,
        demographic_df=demographic_df,
        demographic_id_col=demographic_df.index.name,
        raw_demographic_data=raw_demographic_data,
        gene_correlations=correlation_matrix,
        disease_effects_config=disease_effects_config,
        n_samples=n_samples
    )

    df_genes_final['Grupo'] = demographic_df['Grupo']
    df_genes_final['Grupo'] = df_genes_final['Grupo'].map({'Control': 0, 'Enfermedad': 1})

    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, f"gene_dataset_split_correlated_{n_genes}_genes.csv")
    df_genes_final.to_csv(output_filepath, index=True)

    print(f"Gene dataset saved to: {output_filepath}")
    print("Generation complete.")

if __name__ == "__main__":
    create_split_gene_correlated_dataset(
        n_samples=10000,
        n_genes=50,
        num_disease_genes_1=20,
        num_disease_genes_2=10,
        corr_group_1=0.6,
        corr_group_2=0.7,
        gene_type="Microarray",
        disease_effect_type="additive_shift",
        disease_effect_type_2="additive_shift",
        disease_effect_value=[20, 50],
        disease_effect_value_2=[10, 15],
        num_patients_subtype_1 = 3000,
        output_dir="generated_datasets_50"
    )