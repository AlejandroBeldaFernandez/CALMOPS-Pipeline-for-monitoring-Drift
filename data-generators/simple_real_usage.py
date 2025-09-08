#!/usr/bin/env python3
"""
Simple usage example for CalmOps Real Data Processing

This script demonstrates how to use the real data processing framework
for data augmentation, drift injection, and distribution manipulation.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine

# Add the current directory to path for imports
sys.path.append('/home/alex/calmops/data-generators')

from Real.DriftInjector import DriftInjector
from Real.DistributionChanger import DistributionChanger
from Real.BlockDriftGenerator import RealBlockDriftGenerator

def load_sample_dataset():
    """Load a sample real dataset for demonstration"""
    print("Loading sample dataset (Breast Cancer Wisconsin)...")
    
    # Load breast cancer dataset from sklearn
    data = load_breast_cancer()
    
    # Convert to DataFrame
    feature_names = [name.replace(' ', '_').replace('(', '').replace(')', '') for name in data.feature_names[:10]]  # Use first 10 features
    df = pd.DataFrame(data.data[:, :10], columns=feature_names)
    df['target'] = data.target
    
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)-1} features")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"Features: {', '.join(feature_names[:5])}...")
    
    return df

def main():
    print("CalmOps Real Data Processing - Simple Usage Example") 
    print("=" * 60)
    
    # Create output directory
    output_dir = "examples_real_usage"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample dataset
    original_df = load_sample_dataset()
    
    print("\nThis example shows how to:")
    print("1. Inject different types of drift into real data")
    print("2. Change class distributions")
    print("3. Create block-based datasets")
    print("4. Generate temporal drift patterns")
    print()
    
    # Example 1: Feature Drift Injection
    print("EXAMPLE 1: Feature Drift Injection")
    print("-" * 40)
    
    drift_injector = DriftInjector(random_state=42)
    
    # Apply feature drift to numerical features
    numeric_features = original_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('target')  # Remove target column
    
    feature_drift_df = drift_injector.inject_feature_drift(
        df=original_df,
        feature_cols=numeric_features[:5],  # Use first 5 features
        drift_magnitude=0.25,
        drift_type="gaussian_noise"
    )
    
    feature_drift_file = os.path.join(output_dir, "feature_drift_data.csv")
    feature_drift_df.to_csv(feature_drift_file, index=False)
    
    print(f"Applied feature drift to {len(numeric_features[:5])} features")
    print(f"Drift magnitude: 25% noise injection")
    print(f"Saved to: {feature_drift_file}")
    
    # Show the effect
    original_mean = original_df[numeric_features[0]].mean()
    drift_mean = feature_drift_df[numeric_features[0]].mean()
    print(f"Feature '{numeric_features[0]}' mean: {original_mean:.3f} -> {drift_mean:.3f}")
    print()
    
    # Example 2: Label Drift Injection
    print("EXAMPLE 2: Label Drift Injection")
    print("-" * 40)
    
    label_drift_df = drift_injector.inject_label_drift(
        df=original_df,
        target_col='target',
        drift_magnitude=0.1  # Flip 10% of labels
    )
    
    label_drift_file = os.path.join(output_dir, "label_drift_data.csv")
    label_drift_df.to_csv(label_drift_file, index=False)
    
    print("Applied label drift: 10% of labels flipped")
    print(f"Original distribution: {original_df['target'].value_counts().to_dict()}")
    print(f"Drift distribution: {label_drift_df['target'].value_counts().to_dict()}")
    print(f"Saved to: {label_drift_file}")
    print()
    
    # Example 3: Distribution Manipulation
    print("EXAMPLE 3: Distribution Manipulation")
    print("-" * 40)
    
    dist_changer = DistributionChanger(random_state=42)
    
    # Create imbalanced version
    imbalanced_df = dist_changer.create_imbalanced_version(
        df=original_df,
        target_col='target',
        imbalance_ratio=0.2  # 20% minority class
    )
    
    imbalanced_file = os.path.join(output_dir, "imbalanced_data.csv")
    imbalanced_df.to_csv(imbalanced_file, index=False)
    
    print("Created imbalanced version with 20% minority class")
    print(f"New distribution: {imbalanced_df['target'].value_counts().to_dict()}")
    print(f"Saved to: {imbalanced_file}")
    
    # Balance it back
    balanced_df = dist_changer.balance_dataset(
        df=imbalanced_df,
        target_col='target',
        method='undersample'
    )
    
    balanced_file = os.path.join(output_dir, "rebalanced_data.csv") 
    balanced_df.to_csv(balanced_file, index=False)
    
    print(f"Re-balanced dataset: {balanced_df['target'].value_counts().to_dict()}")
    print(f"Saved to: {balanced_file}")
    print()
    
    # Example 4: Block-based Data Generation
    print("EXAMPLE 4: Block-based Data Generation")
    print("-" * 40)
    
    block_generator = RealBlockDriftGenerator(random_state=42)
    
    # Generate blocks with different distributions
    block_file = block_generator.generate_blocks_from_real_data(
        df=original_df,
        target_col='target',
        output_path=output_dir,
        filename='blocks_data.csv',
        n_blocks=3,
        block_sizes=[200, 180, 150],
        block_distributions=[
            {0: 0.7, 1: 0.3},  # Block 1: More class 0
            {0: 0.5, 1: 0.5},  # Block 2: Balanced  
            {0: 0.3, 1: 0.7}   # Block 3: More class 1
        ]
    )
    
    print(f"Generated block-based dataset: {block_file}")
    print()
    
    # Example 5: Temporal Drift
    print("EXAMPLE 5: Temporal Drift Patterns") 
    print("-" * 40)
    
    temporal_file = block_generator.generate_temporal_drift_blocks(
        df=original_df,
        target_col='target',
        output_path=output_dir,
        filename='temporal_drift_data.csv',
        n_blocks=4,
        drift_strength=0.3,
        drift_type='gradual'
    )
    
    print(f"Generated temporal drift dataset: {temporal_file}")
    print("Drift type: Gradual distribution change over time")
    print()
    
    # Summary
    print("SUMMARY")
    print("=" * 60)
    print("Generated real data processing examples:")
    
    expected_files = [
        "feature_drift_data.csv",
        "label_drift_data.csv",
        "imbalanced_data.csv", 
        "rebalanced_data.csv",
        "blocks_data.csv",
        "temporal_drift_data.csv"
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
    print("- Feature drift injection (Gaussian noise, shift, scale)")
    print("- Label drift injection (controlled label flipping)")
    print("- Distribution manipulation (imbalance creation, rebalancing)")
    print("- Block-based data generation with custom distributions")
    print("- Temporal drift patterns (gradual, sudden, cyclic)")
    
    print("\nReal Data Processing Benefits:")
    print("- Augment small datasets")
    print("- Test model robustness to different types of drift")
    print("- Create controlled experimental conditions")
    print("- Simulate distribution changes over time")
    
    print("\nNext Steps:")
    print("1. Load your own CSV data: pd.read_csv('your_data.csv')")
    print("2. Apply different drift injection techniques")
    print("3. Test ML pipelines with processed data")
    print("4. Compare model performance on original vs. modified data")

if __name__ == "__main__":
    main()