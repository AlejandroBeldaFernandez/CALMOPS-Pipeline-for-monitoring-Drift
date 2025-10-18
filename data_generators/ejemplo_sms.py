#!/usr/bin/env python3
"""
Example script for comparing data synthesis methods: CART, RF, and CTGAN.

This script demonstrates the usage of the RealGenerator to synthesize data from a real-world
dataset (`Pacientes_Estratificacion.txt`). It performs the following steps:

1.  **Load Data**: Loads the patient stratification dataset from a specified file path.
2.  **Preprocessing**: 
    - Drops high-cardinality columns (`FECHA_DIAGNOSTICO`, `ETIQUETA`).
    - Removes rows with null values.
    - Encodes all categorical (object type) columns into numerical codes.
3.  **Sampling**: Takes a random sample of 20,000 instances from the preprocessed dataset to serve as the training data.
4.  **Synthesize**: Iterates through three synthesis methods: 'cart', 'rf', and 'ctgan'.
5.  **Generate and Report**: For each method, it:
    - Creates a dedicated output directory.
    - Initializes a `RealGenerator` with the training data and method-specific parameters.
    - Synthesizes 40,000 new instances.
    - Automatically generates a comprehensive quality report comparing the synthetic data to the original sample.

This script serves as a practical example of how to use the `RealGenerator` and assess the quality of the generated data.
"""
import os
import pandas as pd
import sys
import numpy as np

# --- Path Correction for Module Import ---
# Add the project root to the Python path to ensure that modules can be found
# regardless of where the script is executed from.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of Correction ---

# Import the generator from the absolute path
from data_generators.Real.RealGenerator import RealGenerator

def main():
    """Main function of the script."""
    print("Starting synthesis method comparison script (CART, RF, CTGAN)...")

    # Create the main output directory
    base_output_dir = "calmops"
    os.makedirs(base_output_dir, exist_ok=True)

    file_path = '/opt/covid/Pacientes_Estratificacion.txt'

    # Load the original dataset
    print(f"Loading original dataset from: {file_path}...")
    try:
        estratificacion_full = pd.read_csv(file_path, sep='|', encoding='latin1')
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        print("Please ensure the path is correct and you have read permissions.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- General Preprocessing ---
    print("\n--- Preprocessing the dataset ---")
    columns_to_drop = ['FECHA_DIAGNOSTICO', 'ETIQUETA']
    print(f"Dropping high-cardinality columns: {columns_to_drop}")
    estratificacion_full = estratificacion_full.drop(columns=columns_to_drop)
    
    print("--- Handling null values ---")
    rows_before = len(estratificacion_full)
    estratificacion_full = estratificacion_full.dropna()
    rows_after = len(estratificacion_full)
    print(f"Removed {rows_before - rows_after} rows with null values.")
    # --- End of Preprocessing ---

    # --- Categorical Variable Preprocessing ---
    print("\n--- Encoding categorical variables to numeric ---")
    categorical_cols = estratificacion_full.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"Columns to encode: {list(categorical_cols)}")
        for col in categorical_cols:
            # Convert column to 'category' type and then to numerical codes
            estratificacion_full[col] = estratificacion_full[col].astype('category').cat.codes
        print("Encoding complete.")
    else:
        print("No categorical columns found to encode.")
    print("--- End of Encoding ---\n")
    # --- End of Categorical Preprocessing ---
    print(estratificacion_full.describe)
    # --- Original Dataset Sampling ---
    n_original_samples = 20000
    print(f"--- Taking a sample of {n_original_samples} instances from the original dataset ---")
    if len(estratificacion_full) > n_original_samples:
        estratificacion = estratificacion_full.sample(n=n_original_samples, random_state=42)
        print(f"Sampling complete. The training dataset has {len(estratificacion)} rows.")
    else:
        estratificacion = estratificacion_full
        print(f"The original dataset has fewer than {n_original_samples} rows, using the full dataset ({len(estratificacion)} rows).")
    print("--- End of Sampling ---\n")
    # --- End of Sampling ---

    # Number of instances to generate
    n_samples_to_generate = 40000

    # Define the methods to compare
    methods_to_compare = {
        'cart': {},
        'rf': {
            'cart_iterations': 3,
            'rf_n_estimators': 50,
            'rf_min_samples_leaf': 5
        },
        'ctgan': {}
    }
    
    target_column_name = 'SITUACION'
    
    if target_column_name not in estratificacion.columns:
        print(f"Error: Target column '{target_column_name}' not found in the dataset.")
        print(f"Available columns: {list(estratificacion.columns)}")
        return

    # Iterate and run each synthesis method
    for method, params in methods_to_compare.items():
        print(f"\n{'='*60}")
        print(f"Generating {n_samples_to_generate} instances with method: '{method}'")
        print(f"{'='*60}")

        # Create a specific directory for this method's reports
        report_dir = os.path.join(base_output_dir, f"report_{method}")
        os.makedirs(report_dir, exist_ok=True)
        print(f"Report directory for '{method.upper()}' created at: {report_dir}")

        generator = RealGenerator(
            original_data=estratificacion,
            method=method,
            target_column=target_column_name,
            random_state=42,
            **params
        )

        # Call synthesize, passing the output directory for reports
        df_synth = generator.synthesize(n_samples=n_samples_to_generate, output_dir=report_dir)

        if df_synth is not None:
            print(f"\nSynthesis with method '{method.upper()}' complete.")
            print(f"Quality report for '{method.upper()}' saved in: {report_dir}")
            
            # NOTE: The original proportion comparison is omitted because the text columns
            # are now numeric. A "decoding" step would be needed to see the original
            # text labels for comparison.
            
        else:
            print(f"\nSynthesis with method '{method}' failed.")

    print(f"\nComparison script finished. All reports are in the '{base_output_dir}' folder.")

if __name__ == "__main__":
    main()
