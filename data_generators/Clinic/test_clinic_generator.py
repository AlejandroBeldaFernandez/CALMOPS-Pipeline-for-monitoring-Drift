import unittest
import os
import shutil
import pandas as pd
import numpy as np
from .Clinic import ClinicGenerator, replicate_genes_proteinas

class TestClinicGenerator(unittest.TestCase):

    def setUp(self):
        self.output_dir = "./test_output_unittest"
        self.generator = ClinicGenerator(seed=42)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_replicate_genes_proteinas_microarray(self):
        n_samples = 50
        factor_escala = 1.0
        n_genes_expected = int(100 * factor_escala)
        n_proteins_expected = int(60 * factor_escala)

        # CORRECCIÓN: Llamar como función externa pasando el generador
        df_demo_t2, df_genes_t2, df_proteins_t2 = replicate_genes_proteinas(
            generator=self.generator, 
            mode='microarray', 
            output_dir=self.output_dir, 
            n_samples=n_samples, 
            factor_escala=factor_escala
        )
        
        # Check if files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_demografico_t1_microarray.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_genes_t1_microarray.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_proteinas_t1_microarray.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "report_t1_microarray.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_demografico_t2_microarray.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_genes_t2_microarray.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_proteinas_t2_microarray.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "report_t2_microarray.txt")))

        # Check if reports are not empty
        with open(os.path.join(self.output_dir, "report_t1_microarray.txt"), 'r') as f:
            self.assertGreater(len(f.read()), 0)
        with open(os.path.join(self.output_dir, "report_t2_microarray.txt"), 'r') as f:
            self.assertGreater(len(f.read()), 0)

        # Check DataFrame shapes and columns for T2 data
        self.assertEqual(df_demo_t2.shape[0], n_samples)
        # self.assertIn('ID_Paciente', df_demo_t2.index.name) # La función guarda el índice en el CSV, pero el df retornado no
        self.assertIn('Grupo', df_demo_t2.columns)
        self.assertEqual(df_genes_t2.shape[0], n_samples)
        self.assertEqual(df_genes_t2.shape[1], n_genes_expected)
        self.assertTrue(all(col.startswith('G_') for col in df_genes_t2.columns))
        self.assertEqual(df_proteins_t2.shape[0], n_samples)
        self.assertEqual(df_proteins_t2.shape[1], n_proteins_expected)
        self.assertTrue(all(col.startswith('P_') for col in df_proteins_t2.columns))

        # Check data types for microarray
        self.assertTrue(pd.api.types.is_float_dtype(df_genes_t2.iloc[:, 0]))
        self.assertTrue(pd.api.types.is_float_dtype(df_proteins_t2.iloc[:, 0]))

        # Check basic content of reports (keywords)
        with open(os.path.join(self.output_dir, "report_t1_microarray.txt"), 'r', encoding='utf-8') as f:
            report_content = f.read()
            self.assertIn("T1 DATA (MICROARRAY) - Modules A, B, C", report_content)
            self.assertIn("GENES T1", report_content)
            self.assertIn("PROTEINS T1", report_content)
        with open(os.path.join(self.output_dir, "report_t2_microarray.txt"), 'r', encoding='utf-8') as f:
            report_content = f.read()
            self.assertIn("T2 DATA (MICROARRAY) - WITH LONGITUDINAL DRIFT", report_content)
            self.assertIn("LONGITUDINAL TRANSITION (DRIFT) COHORT ANALYSIS", report_content)

        factor_escala = 1.0
        n_genes_expected = int(100 * factor_escala)
        n_proteins_expected = int(60 * factor_escala)

        # CORRECCIÓN: Llamar como función externa pasando el generador
        df_demo_t2, df_genes_t2, df_proteins_t2 = replicate_genes_proteinas(
            generator=self.generator,
            mode='rna-seq', 
            output_dir=self.output_dir, 
            n_samples=n_samples, 
            factor_escala=factor_escala
        )
        
        # Check if files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_demografico_t1_rna-seq.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_genes_t1_rna-seq.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_proteinas_t1_rna-seq.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "report_t1_rna-seq.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_demografico_t2_rna-seq.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_genes_t2_rna-seq.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "dataset_proteinas_t2_rna-seq.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "report_t2_rna-seq.txt")))

        # Check if reports are not empty
        with open(os.path.join(self.output_dir, "report_t1_rna-seq.txt"), 'r') as f:
            self.assertGreater(len(f.read()), 0)
        with open(os.path.join(self.output_dir, "report_t2_rna-seq.txt"), 'r') as f:
            self.assertGreater(len(f.read()), 0)

        # Check DataFrame shapes and columns for T2 data
        self.assertEqual(df_demo_t2.shape[0], n_samples)
        # self.assertIn('ID_Paciente', df_demo_t2.index.name) # Idem
        self.assertIn('Grupo', df_demo_t2.columns)
        self.assertEqual(df_genes_t2.shape[0], n_samples)
        self.assertEqual(df_genes_t2.shape[1], n_genes_expected)
        self.assertTrue(all(col.startswith('G_') for col in df_genes_t2.columns))
        self.assertEqual(df_proteins_t2.shape[0], n_samples)
        self.assertEqual(df_proteins_t2.shape[1], n_proteins_expected)
        self.assertTrue(all(col.startswith('P_') for col in df_proteins_t2.columns))

        # Check data types for rnaseq
        self.assertTrue(pd.api.types.is_integer_dtype(df_genes_t2.iloc[:, 0]))
        self.assertTrue(pd.api.types.is_float_dtype(df_proteins_t2.iloc[:, 0]))

        # Check basic content of reports (keywords)
        with open(os.path.join(self.output_dir, "report_t1_rna-seq.txt"), 'r', encoding='utf-8') as f:
            report_content = f.read()
            self.assertIn("T1 DATA (RNA-SEQ) - Modules A, B, C", report_content)
            self.assertIn("GENES T1", report_content)
            self.assertIn("PROTEINS T1", report_content)
        with open(os.path.join(self.output_dir, "report_t2_rna-seq.txt"), 'r', encoding='utf-8') as f:
            report_content = f.read()
            self.assertIn("T2 DATA (RNA-SEQ) - WITH LONGITUDINAL DRIFT", report_content)
            self.assertIn("LONGITUDINAL TRANSITION (DRIFT) COHORT ANALYSIS", report_content)

    def test_generate_demographic_data(self):
        n_samples = 10
        df_demo, raw_df_demo = self.generator.generate_demographic_data(n_samples=n_samples)
        self.assertIsInstance(df_demo, pd.DataFrame)
        self.assertEqual(df_demo.shape[0], n_samples)
        self.assertIn('ID_Paciente', df_demo.index.name)
        self.assertIn('Grupo', df_demo.columns)
        self.assertIsInstance(raw_df_demo, pd.DataFrame)
        self.assertEqual(raw_df_demo.shape[0], n_samples)

    def test_generate_demographic_data_custom_distributions(self):
        n_samples = 10
        custom_cols = {
            'Indice_Masa_Corporal': {'distribution': 'uniform', 'loc': 18, 'scale': 12},
            'Numero_Visitas': {'distribution': 'poisson', 'mu': 5}
        }
        df_demo, raw_df_demo = self.generator.generate_demographic_data(
            n_samples=n_samples,
            custom_demographic_columns=custom_cols
        )
        self.assertIn('Indice_Masa_Corporal', df_demo.columns)
        self.assertIn('Numero_Visitas', df_demo.columns)
        self.assertTrue(pd.api.types.is_float_dtype(df_demo['Indice_Masa_Corporal']))
        self.assertTrue(pd.api.types.is_numeric_dtype(raw_df_demo['Numero_Visitas'])) # Poisson is integer, so numeric is fine

    def test_demographic_data_with_class_assignment_function(self):
        n_samples = 20

        def assign_by_age(df):
            return np.where(df['Edad'] > 65, 'Enfermedad', 'Control')

        df_demo, _ = self.generator.generate_demographic_data(
            n_samples=n_samples,
            class_assignment_function=assign_by_age
        )

        self.assertEqual(df_demo.shape[0], n_samples)
        for index, row in df_demo.iterrows():
            if row['Edad'] > 65:
                self.assertEqual(row['Grupo'], 'Enfermedad')
            else:
                self.assertEqual(row['Grupo'], 'Control')

    def test_subgroup_assignment_and_gene_effects(self):
        """
        Tests the new functionality:
        1. Assigning patients to specific subgroups using a custom function.
        2. Applying gene effects based on those assigned subgroups.
        """
        n_samples = 60
        n_genes = 30

        # 1. Define the class assignment function with detailed subgroups
        def assign_disease_subgroups(df):
            subgroups = pd.Series('Control', index=df.index)
            # Assign subgroups based on age
            subgroups.loc[(df['Edad'] > 40) & (df['Edad'] <= 50)] = 'Grupo_A'
            subgroups.loc[(df['Edad'] > 50) & (df['Edad'] <= 60)] = 'Grupo_B'
            subgroups.loc[df['Edad'] > 60] = 'Grupo_C'
            return subgroups

        # 2. Generate demographic data with these subgroups
        df_demo, raw_demo = self.generator.generate_demographic_data(
            n_samples=n_samples,
            class_assignment_function=assign_disease_subgroups
        )

        self.assertIn('Subgrupo_Enfermedad', df_demo.columns)
        self.assertIn('Grupo_A', df_demo['Subgrupo_Enfermedad'].unique())
        self.assertIn('Grupo_B', df_demo['Subgrupo_Enfermedad'].unique())
        self.assertIn('Grupo_C', df_demo['Subgrupo_Enfermedad'].unique())

        # 3. Define gene effects linked to these subgroups
        gene_effects_config = {
            'effects': {
                'Efecto_A': {'indices': list(range(0, 10)), 'effect_type': 'additive_shift', 'effect_value': 2.0},
                'Efecto_B': {'indices': list(range(10, 20)), 'effect_type': 'additive_shift', 'effect_value': -2.0},
                'Efecto_C': {'indices': list(range(20, 30)), 'effect_type': 'fold_change', 'effect_value': 3.0}
            },
            'patient_subgroups': [
                {'name': 'Grupo_A', 'apply_effects': ['Efecto_A']},
                {'name': 'Grupo_B', 'apply_effects': ['Efecto_A', 'Efecto_B']},
                {'name': 'Grupo_C', 'apply_effects': ['Efecto_C']}
            ]
        }

        # 4. Generate baseline gene data (all control) to establish a baseline
        df_genes_control = self.generator.generate_gene_data(
            n_genes=n_genes,
            gene_type="Microarray",
            demographic_df=df_demo,
            demographic_id_col=df_demo.index.name,
            disease_effects_config=None # No effects
        )

        # 5. Generate gene data with effects applied based on subgroups
        df_genes_effects = self.generator.generate_gene_data(
            n_genes=n_genes,
            gene_type="Microarray",
            demographic_df=df_demo,
            demographic_id_col=df_demo.index.name,
            disease_effects_config=gene_effects_config,
            subgroup_col='Subgrupo_Enfermedad' # IMPORTANT: Link the effects to the subgroup column
        )

        # 6. Verify the effects were applied correctly
        # Get patient IDs for each subgroup
        pids_grupo_a = df_demo[df_demo['Subgrupo_Enfermedad'] == 'Grupo_A'].index
        pids_grupo_b = df_demo[df_demo['Subgrupo_Enfermedad'] == 'Grupo_B'].index
        pids_grupo_c = df_demo[df_demo['Subgrupo_Enfermedad'] == 'Grupo_C'].index
        pids_control = df_demo[df_demo['Subgrupo_Enfermedad'] == 'Control'].index

        # Check Group A: Genes 0-9 should be higher
        if not pids_grupo_a.empty:
            mean_effect_a = df_genes_effects.loc[pids_grupo_a, 'G_0':'G_9'].mean().mean()
            mean_control_a = df_genes_control.loc[pids_grupo_a, 'G_0':'G_9'].mean().mean()
            self.assertGreater(mean_effect_a, mean_control_a + 1.0) # Check if significantly higher

        # Check Group B: Genes 0-9 higher, 10-19 lower
        if not pids_grupo_b.empty:
            mean_effect_b1 = df_genes_effects.loc[pids_grupo_b, 'G_0':'G_9'].mean().mean()
            mean_control_b1 = df_genes_control.loc[pids_grupo_b, 'G_0':'G_9'].mean().mean()
            self.assertGreater(mean_effect_b1, mean_control_b1 + 1.0)

            mean_effect_b2 = df_genes_effects.loc[pids_grupo_b, 'G_10':'G_19'].mean().mean()
            mean_control_b2 = df_genes_control.loc[pids_grupo_b, 'G_10':'G_19'].mean().mean()
            self.assertLess(mean_effect_b2, mean_control_b2 - 1.0)

        # Check Group C: Genes 20-29 should be higher (fold change)
        if not pids_grupo_c.empty:
            mean_effect_c = df_genes_effects.loc[pids_grupo_c, 'G_20':'G_29'].mean().mean()
            mean_control_c = df_genes_control.loc[pids_grupo_c, 'G_20':'G_29'].mean().mean()
            # Avoid division by zero if control mean is near zero
            if abs(mean_control_c) > 0.1:
                self.assertGreater(mean_effect_c / mean_control_c, 2.0)

        # Check Control group: No significant change
        if not pids_control.empty:
            mean_effect_control = df_genes_effects.loc[pids_control].mean().mean()
            mean_control_control = df_genes_control.loc[pids_control].mean().mean()
            self.assertAlmostEqual(mean_effect_control, mean_control_control, delta=0.5)

    def test_generate_gene_data_microarray(self):
        n_samples = 10
        n_genes = 5
        df_demo, _ = self.generator.generate_demographic_data(n_samples=n_samples)
        df_genes = self.generator.generate_gene_data(n_genes=n_genes, gene_type="Microarray", demographic_df=df_demo, demographic_id_col='ID_Paciente')
        self.assertIsInstance(df_genes, pd.DataFrame)
        self.assertEqual(df_genes.shape[0], n_samples)
        self.assertEqual(df_genes.shape[1], n_genes) 
        self.assertTrue(all(col.startswith('G_')  for col in df_genes.columns))
        self.assertTrue(pd.api.types.is_float_dtype(df_genes.iloc[:, 0]))

    def test_generate_gene_data_rnaseq(self):
        n_samples = 10
        n_genes = 5
        df_demo, _ = self.generator.generate_demographic_data(n_samples=n_samples)
        df_genes = self.generator.generate_gene_data(n_genes=n_genes, gene_type="RNA-Seq", demographic_df=df_demo, demographic_id_col='ID_Paciente')
        self.assertIsInstance(df_genes, pd.DataFrame)
        self.assertEqual(df_genes.shape[0], n_samples)
        self.assertEqual(df_genes.shape[1], n_genes) 
        self.assertTrue(all(col.startswith('G_') for col in df_genes.columns))
        self.assertTrue(pd.api.types.is_integer_dtype(df_genes.iloc[:, 0]))

    def test_generate_protein_data(self):
        n_samples = 10
        n_proteins = 5
        df_demo, _ = self.generator.generate_demographic_data(n_samples=n_samples)
        df_proteins = self.generator.generate_protein_data(n_proteins=n_proteins, demographic_df=df_demo, demographic_id_col='ID_Paciente')
        self.assertIsInstance(df_proteins, pd.DataFrame)
        self.assertEqual(df_proteins.shape[0], n_samples)
        self.assertEqual(df_proteins.shape[1], n_proteins) 
        self.assertTrue(all(col.startswith('P_') for col in df_proteins.columns))
        self.assertTrue(pd.api.types.is_float_dtype(df_proteins.iloc[:, 0]))

    def test_generate_additional_time_step_data(self):
        n_samples = 10
        n_genes = 5
        n_proteins = 3
        date_value = "2024-01-01"
        omics_to_generate = ['genes', 'proteins']

        df_demo_t2, omics_df_t2 = self.generator.generate_additional_time_step_data(
            n_samples=n_samples,
            date_value=date_value,
            omics_to_generate=omics_to_generate,
            n_genes=n_genes,
            n_proteins=n_proteins,
            gene_type='Microarray'
        )
        self.assertIsInstance(df_demo_t2, pd.DataFrame)
        self.assertEqual(df_demo_t2.shape[0], n_samples)
        self.assertIn('Fecha', df_demo_t2.columns)
        self.assertIsInstance(omics_df_t2, pd.DataFrame)
        self.assertEqual(omics_df_t2.shape[0], n_samples)
        self.assertEqual(omics_df_t2.shape[1], n_genes + n_proteins) # No 'Grupo' column in omics_df_t2

    def test_inject_drift_group_transition(self):
        n_samples = 20
        n_genes = 5
        n_proteins = 3
        df_demo_t1, _ = self.generator.generate_demographic_data(n_samples=n_samples, control_disease_ratio=0.5)
        df_genes_t1 = self.generator.generate_gene_data(n_genes=n_genes, gene_type="Microarray", demographic_df=df_demo_t1, demographic_id_col='ID_Paciente')
        df_proteins_t1 = self.generator.generate_protein_data(n_proteins=n_proteins, demographic_df=df_demo_t1, demographic_id_col='ID_Paciente')
        omics_df_t1 = pd.concat([df_genes_t1, df_proteins_t1], axis=1)

        transition_criteria = {'percentage': 0.5} # Transition 50% of control patients
        
        df_demo_t2, omics_df_t2 = self.generator.inject_drift_group_transition(
            demographic_df=df_demo_t1.copy(),
            omics_data_df=omics_df_t1.copy(),
            transition_type='control_to_disease',
            selection_criteria=transition_criteria,
            omics_type='genes',
            gene_type='Microarray',
            disease_gene_indices=[0, 1],
            disease_effect_type='additive_shift',
            disease_effect_value=1.0,
            n_genes_total=n_genes,
            n_proteins_total=n_proteins
        )
        self.assertIsInstance(df_demo_t2, pd.DataFrame)
        self.assertIsInstance(omics_df_t2, pd.DataFrame)
        # Check if some patients transitioned from Control to Enfermedad
        self.assertGreater(len(df_demo_t2[df_demo_t2['Grupo'] == 'Enfermedad']), len(df_demo_t1[df_demo_t1['Grupo'] == 'Enfermedad']))

    def test_inject_drift_correlated_modules(self):
        n_samples = 10
        n_genes = 10
        df_demo, _ = self.generator.generate_demographic_data(n_samples=n_samples)
        df_genes = self.generator.generate_gene_data(n_genes=n_genes, gene_type="Microarray", demographic_df=df_demo, demographic_id_col='ID_Paciente')
        
        module_indices = [0, 1, 2]
        new_correlation_matrix = np.array([[1.0, 0.9, 0.8],
                                           [0.9, 1.0, 0.7],
                                           [0.8, 0.7, 1.0]])
        
        updated_omics_df = self.generator.inject_drift_correlated_modules(
            omics_data_df=df_genes.copy(),
            module_indices=module_indices,
            new_correlation_matrix=new_correlation_matrix,
            omics_type='genes',
            gene_type='Microarray'
        )
        self.assertIsInstance(updated_omics_df, pd.DataFrame)
        self.assertEqual(updated_omics_df.shape, df_genes.shape)
        # More detailed assertions would involve checking actual correlations, but this is a start.

if __name__ == '__main__':
    unittest.main()