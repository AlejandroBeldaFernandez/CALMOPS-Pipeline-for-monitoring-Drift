#!/usr/bin/env python3
"""
Simple usage example for CalmOps Synthetic Data Generation

This script demonstrates how to use the synthetic data generation framework
with different generators and drift scenarios.
"""

import os
import sys
import pandas as pd

# Add the current directory to path for imports
sys.path.append('/home/alex/calmops/data-generators')

from Synthetic.GeneratorFactory import GeneratorFactory, GeneratorConfig, GeneratorType
from SimpleSyntheticGenerator import SimpleSyntheticGenerator

def main():
    print("CalmOps Synthetic Data Generation - Simple Usage Example")
    print("=" * 60)
    
    # Create output directory
    output_dir = "examples_synthetic_usage"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the simplified generator
    generator = SimpleSyntheticGenerator()
    
    print("This example shows how to:")
    print("1. Generate basic synthetic datasets")
    print("2. Create datasets with concept drift") 
    print("3. Create datasets with data drift (class distribution change)")
    print()
    
    # Example 1: Basic AGRAWAL dataset
    print("EXAMPLE 1: Basic AGRAWAL Dataset")
    print("-" * 40)
    
    # Configure AGRAWAL generator
    agrawal_config = GeneratorConfig(
        random_state=42,
        classification_function=0,  # Function variant
        perturbation=0.1           # Noise level
    )
    
    agrawal_gen = GeneratorFactory.create_generator(GeneratorType.AGRAWAL, agrawal_config)
    
    # Generate basic dataset
    basic_file = generator.generate_simple(
        generator_instance=agrawal_gen,
        output_path=output_dir,
        filename="basic_agrawal.csv",
        n_samples=1500,
        target_col="target"
    )
    print(f"Generated basic AGRAWAL dataset: {basic_file}")
    print()
    
    # Example 2: SEA dataset with balance
    print("EXAMPLE 2: Balanced SEA Dataset")
    print("-" * 40)
    
    sea_config = GeneratorConfig(
        random_state=42,
        function=2,              # SEA function variant
        noise_percentage=0.15    # Noise level
    )
    
    sea_gen = GeneratorFactory.create_generator(GeneratorType.SEA, sea_config)
    
    # Generate and manually balance
    all_samples = list(sea_gen.take(2500))
    
    # Separate by class and balance
    class_0 = [s for s in all_samples if s[1] == False]
    class_1 = [s for s in all_samples if s[1] == True]
    
    balanced_samples = class_0[:750] + class_1[:750]
    
    # Convert to DataFrame
    data = []
    for x, y in balanced_samples:
        row = list(x.values()) + [int(y)]
        data.append(row)
    
    columns = list(balanced_samples[0][0].keys()) + ["target"]
    df = pd.DataFrame(data, columns=columns)
    
    sea_file = os.path.join(output_dir, "balanced_sea.csv")
    df.to_csv(sea_file, index=False)
    
    print(f"Generated balanced SEA dataset: {sea_file}")
    print(f"Samples: {len(df)}")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    print()
    
    # Example 3: HYPERPLANE with concept drift
    print("EXAMPLE 3: HYPERPLANE with Concept Drift")
    print("-" * 40)
    
    # Create two different HYPERPLANE configurations
    hyperplane_base = GeneratorConfig(
        random_state=42,
        n_features=8,
        mag_change=0.0,           # No change initially
        noise_percentage_hyperplane=0.05,
        sigma=0.1
    )
    
    hyperplane_drift = GeneratorConfig(
        random_state=43,
        n_features=8,
        mag_change=0.6,           # Significant change
        noise_percentage_hyperplane=0.2,
        sigma=0.3
    )
    
    hyp_base = GeneratorFactory.create_generator(GeneratorType.HYPERPLANE, hyperplane_base)
    hyp_drift = GeneratorFactory.create_generator(GeneratorType.HYPERPLANE, hyperplane_drift)
    
    concept_drift_file = generator.generate_with_concept_drift(
        generator_base=hyp_base,
        generator_drift=hyp_drift,
        output_path=output_dir,
        filename="hyperplane_concept_drift.csv",
        n_samples=2000,
        drift_position=1000,
        target_col="target"
    )
    print(f"Generated concept drift dataset: {concept_drift_file}")
    print()
    
    # Example 4: AGRAWAL with data drift
    print("EXAMPLE 4: AGRAWAL with Data Drift")
    print("-" * 40)
    
    data_drift_config = GeneratorConfig(
        random_state=42,
        classification_function=1,
        perturbation=0.1
    )
    
    data_drift_gen = GeneratorFactory.create_generator(GeneratorType.AGRAWAL, data_drift_config)
    
    data_drift_file = generator.generate_with_data_drift(
        generator_instance=data_drift_gen,
        output_path=output_dir,
        filename="agrawal_data_drift.csv",
        n_samples=1800,
        drift_position=900,
        ratio_before={0: 0.8, 1: 0.2},  # 80% class 0, 20% class 1
        ratio_after={0: 0.3, 1: 0.7},   # 30% class 0, 70% class 1
        target_col="target"
    )
    print(f"Generated data drift dataset: {data_drift_file}")
    print()
    
    # Summary
    print("SUMMARY")
    print("=" * 60)
    print("Generated synthetic datasets:")
    
    expected_files = [
        "basic_agrawal.csv",
        "balanced_sea.csv", 
        "hyperplane_concept_drift.csv",
        "agrawal_data_drift.csv"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            df_check = pd.read_csv(filepath)
            print(f"  - {filename}: {len(df_check)} samples ({size_kb:.1f} KB)")
        else:
            print(f"  - {filename}: NOT FOUND")
    
    print(f"\nAll files saved to: {output_dir}")
    print("\nKey Features Demonstrated:")
    print("- AGRAWAL generator for loan approval simulation")
    print("- SEA generator for streaming scenarios") 
    print("- HYPERPLANE generator for high-dimensional classification")
    print("- Concept drift (model/boundary changes)")
    print("- Data drift (class distribution changes)")
    print("- Balanced and imbalanced datasets")
    
    print("\nNext Steps:")
    print("1. Load datasets with pandas: pd.read_csv('filename')")
    print("2. Train ML models to test drift detection")
    print("3. Use datasets for pipeline testing")
    print("4. Experiment with different generator configurations")

if __name__ == "__main__":
    main()