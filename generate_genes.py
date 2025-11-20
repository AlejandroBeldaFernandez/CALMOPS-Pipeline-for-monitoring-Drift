"""
This script provides a comprehensive example of how to use the ClinicGenerator to create synthetic clinical datasets.
It covers the generation of demographic, gene, and protein data, as well as the injection of various types of drift.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

from calmops.data_generators.Clinic.Clinic import ClinicGenerator

def run_clinic_generator_example():
    """
    Runs a full example of the ClinicGenerator, showcasing its main features.
    """
    print("--- Running ClinicGenerator Example ---")

    # Initialize the generator
    generator = ClinicGenerator(seed=42)
    output_dir = Path("clinic_generator_output")
    output_dir.mkdir(exist_ok=True)

    # --- 1. Generate Demographic Data ---
    print("\n--- 1. Generating Demographic Data ---")
    demographic_df, raw_demographic_data = generator.generate_demographic_data(
        n_samples=150,
        control_disease_ratio=0.6,
        custom_demographic_columns={
            'Biomarker_A': stats.norm(loc=10, scale=2),
            'Treatment_Group': stats.binom(n=1, p=0.5)
        },
        date_column_name="Recruitment_Date",
        date_value="2024-01-01"
    )
    print("Demographic data generated:")
    print(demographic_df.head())
    demographic_df.to_csv(output_dir / "demographic_data.csv")

    # --- 2. Generate Gene Expression Data (RNA-Seq) ---
    print("\n--- 2. Generating RNA-Seq Gene Data ---")
    gene_effects_config = {
        'effects': {
            'up_regulated': {'indices': list(range(10)), 'effect_type': 'fold_change', 'effect_value': 2.5},
            'down_regulated': {'indices': list(range(10, 20)), 'effect_type': 'fold_change', 'effect_value': 0.5}
        },
        'patient_subgroups': [
            {'name': 'Subgroup1', 'percentage': 0.5, 'apply_effects': ['up_regulated']},
            {'name': 'Subgroup2', 'remainder': True, 'apply_effects': ['down_regulated']}
        ]
    }
    rna_seq_df = generator.generate_gene_data(
        n_genes=50,
        gene_type="RNA-Seq",
        demographic_df=demographic_df,
        demographic_id_col=demographic_df.index.name,
        disease_effects_config=gene_effects_config
    )
    print("RNA-Seq data generated:")
    print(rna_seq_df.head())
    rna_seq_df.to_csv(output_dir / "rna_seq_data.csv")

    # --- 3. Generate Gene Expression Data (Microarray) ---
    print("\n--- 3. Generating Microarray Gene Data ---")
    microarray_df = generator.generate_gene_data(
        n_genes=50,
        gene_type="Microarray",
        demographic_df=demographic_df,
        demographic_id_col=demographic_df.index.name
    )
    print("Microarray data generated:")
    print(microarray_df.head())
    microarray_df.to_csv(output_dir / "microarray_data.csv")

    # --- 4. Generate Protein Data ---
    print("\n--- 4. Generating Protein Data ---")
    protein_effects_config = [
        {'name': 'Protein_Effect_1', 'indices': list(range(5)), 'effect_type': 'additive_shift', 'effect_value': 1.5}
    ]
    protein_df = generator.generate_protein_data(
        n_proteins=30,
        demographic_df=demographic_df,
        demographic_id_col=demographic_df.index.name,
        disease_effects_config=protein_effects_config
    )
    print("Protein data generated:")
    print(protein_df.head())
    protein_df.to_csv(output_dir / "protein_data.csv")

    # --- 5. Inject Drift (Group Transition) ---
    print("\n--- 5. Injecting Drift by Group Transition ---")
    omics_df = pd.concat([rna_seq_df, protein_df], axis=1)
    
    transition_drift_config = {
        'transition_type': 'control_to_disease',
        'selection_criteria': {'percentage': 0.2},
        'omics_type': 'both',
        'disease_gene_indices': list(range(10)),
        'disease_protein_indices': list(range(5)),
        'disease_effect_type': 'fold_change',
        'disease_effect_value': 2.0
    }

    drifted_demographic_df, drifted_omics_df = generator.inject_drift_group_transition(
        demographic_df=demographic_df,
        omics_data_df=omics_df,
        n_genes_total=50,
        n_proteins_total=30,
        gene_type="RNA-Seq",
        **transition_drift_config
    )
    print("Drift injected. Number of transitions:")
    print((drifted_demographic_df['Grupo'] != demographic_df['Grupo']).sum())
    drifted_demographic_df.to_csv(output_dir / "drifted_demographic_data.csv")
    drifted_omics_df.to_csv(output_dir / "drifted_omics_data.csv")

    print(f"\n--- Example run complete. All generated files are in the '{output_dir}' directory. ---")

if __name__ == "__main__":
    run_clinic_generator_example()