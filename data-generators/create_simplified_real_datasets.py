#!/usr/bin/env python3
"""
Create simplified real datasets for presentation
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Add the current directory to path for imports
sys.path.append('/home/alex/calmops/data-generators')

from Real.RealGenerator import RealGenerator
from Real.DriftInjector import DriftInjector
from Real.DistributionChanger import DistributionChanger
from Real.BlockDriftGenerator import RealBlockDriftGenerator

# Setup directories
OUTPUT_DIR = "presentation_real_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("CalmOps Real Dataset Generation - Simplified")
print("=" * 50)

# =============================================================================
# LOAD REAL DATASET FOR PROCESSING
# =============================================================================

print("Loading real dataset: Palmer Penguins (preprocessed)")

# Load the real penguins dataset
penguins_df = pd.read_csv('penguins_preprocessed_tratado.csv')

print(f"Penguins dataset: {len(penguins_df)} samples, {len(penguins_df.columns)-1} features")
print(f"Target distribution (species): {penguins_df['species'].value_counts().to_dict()}")
print(f"Features: {list(penguins_df.columns)}")
print(f"Dataset shape: {penguins_df.shape}")

# Show basic statistics
print("\nDataset statistics:")
print(penguins_df.describe())

print(f"\nTarget variable 'species' classes: {sorted(penguins_df['species'].unique())}")
print(f"Missing values: {penguins_df.isnull().sum().sum()}")

# Use species as target for classification
target_col = 'species'

# =============================================================================
# SCENARIO 1: Basic Real Data Generation with Multiple Methods
# =============================================================================
print("\n" + "="*50)
print("SCENARIO 1: Multiple Generation Methods Comparison")
print("="*50)

try:
    # Test multiple generation methods on penguins data
    penguins_generator = RealGenerator(target_col=target_col, df=penguins_df)
    
    methods = ['resample', 'smote', 'gmm', 'ctgan', 'copula']  # All available methods
    
    for method in methods:
        print(f"\nTesting {method.upper()} method...")
        
        try:
            output_file = penguins_generator.generate(
                output_path=OUTPUT_DIR,
                filename=f"penguins_{method}.csv",
                n_samples=200,
                method=method,
                balance=True
            )
            print(f"  SUCCESS: Generated {output_file}")
            
            # Quick verification
            generated_df = pd.read_csv(output_file)
            print(f"  Generated {len(generated_df)} samples")
            print(f"  Target distribution: {generated_df[target_col].value_counts().to_dict()}")
            
        except Exception as e:
            print(f"  FAILED: {str(e)}")

except Exception as e:
    print(f"Penguins generation failed: {e}")

# =============================================================================
# SCENARIO 2: Drift Injection Demonstration
# =============================================================================
print("\n" + "="*50)
print("SCENARIO 2: Drift Injection")
print("="*50)

try:
    drift_injector = DriftInjector(random_state=42)
    
    # Feature drift on numeric features
    numeric_features = penguins_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    
    print(f"Applying feature drift to penguins data (features: {numeric_features[:3]})...")
    feature_drift_df = drift_injector.inject_feature_drift(
        penguins_df,
        feature_cols=numeric_features[:3],  # First 3 numeric features
        drift_magnitude=0.2,
        drift_type='gaussian_noise'
    )
    
    feature_drift_file = os.path.join(OUTPUT_DIR, "penguins_feature_drift.csv")
    feature_drift_df.to_csv(feature_drift_file, index=False)
    
    print(f"  Feature drift applied and saved: {feature_drift_file}")
    if len(numeric_features) > 0:
        feature_name = numeric_features[0]
        print(f"  Original mean {feature_name}: {penguins_df[feature_name].mean():.2f}")
        print(f"  Drift mean {feature_name}: {feature_drift_df[feature_name].mean():.2f}")
    
    # Label drift
    print("\nApplying label drift...")
    label_drift_df = drift_injector.inject_label_drift(
        penguins_df,
        target_col=target_col,
        drift_magnitude=0.1
    )
    
    label_drift_file = os.path.join(OUTPUT_DIR, "penguins_label_drift.csv")
    label_drift_df.to_csv(label_drift_file, index=False)
    
    print(f"  Label drift applied and saved: {label_drift_file}")
    print(f"  Original target dist: {penguins_df[target_col].value_counts().to_dict()}")
    print(f"  Drift target dist: {label_drift_df[target_col].value_counts().to_dict()}")

except Exception as e:
    print(f"Drift injection failed: {e}")

# =============================================================================
# SCENARIO 3: Distribution Manipulation
# =============================================================================
print("\n" + "="*50)
print("SCENARIO 3: Distribution Changes")
print("="*50)

try:
    dist_changer = DistributionChanger(random_state=42)
    
    # Create imbalanced version (for multi-class, we'll focus on making one class minority)
    classes = penguins_df[target_col].unique()
    minority_class = classes[0]  # Use first class as minority
    
    print(f"Creating imbalanced penguins dataset (minority class: {minority_class})...")
    imbalanced_df = dist_changer.create_imbalanced_version(
        penguins_df,
        target_col=target_col,
        imbalance_ratio=0.2,  # 20% minority class
        minority_class=minority_class
    )
    
    imbalanced_file = os.path.join(OUTPUT_DIR, "penguins_imbalanced.csv")
    imbalanced_df.to_csv(imbalanced_file, index=False)
    
    print(f"  Imbalanced dataset created: {imbalanced_file}")
    print(f"  New distribution: {imbalanced_df[target_col].value_counts().to_dict()}")
    
    # Create balanced version
    print("\nBalancing the penguins dataset...")
    balanced_df = dist_changer.balance_dataset(
        penguins_df,
        target_col=target_col,
        method='undersample'
    )
    
    balanced_file = os.path.join(OUTPUT_DIR, "penguins_balanced.csv")
    balanced_df.to_csv(balanced_file, index=False)
    
    print(f"  Balanced dataset created: {balanced_file}")
    print(f"  Balanced distribution: {balanced_df[target_col].value_counts().to_dict()}")

except Exception as e:
    print(f"Distribution manipulation failed: {e}")

# =============================================================================
# SCENARIO 4: Block-based Generation
# =============================================================================
print("\n" + "="*50)
print("SCENARIO 4: Block-based Data Generation")
print("="*50)

try:
    block_generator = RealBlockDriftGenerator(random_state=42)
    
    # Generate blocks with different distributions
    print("Generating penguins blocks with different distributions...")
    
    # Create block distributions for the classes in penguins
    classes = sorted(penguins_df[target_col].unique())
    n_classes = len(classes)
    
    if n_classes == 3:  # Typical for penguins (3 species)
        block_distributions = [
            {classes[0]: 0.6, classes[1]: 0.3, classes[2]: 0.1},  # Block 1: First species dominant
            {classes[0]: 0.3, classes[1]: 0.6, classes[2]: 0.1},  # Block 2: Second species dominant
            {classes[0]: 0.2, classes[1]: 0.2, classes[2]: 0.6},  # Block 3: Third species dominant
        ]
        block_sizes = [80, 70, 50]
    else:
        # Fallback for different number of classes
        equal_prob = 1.0 / n_classes
        block_distributions = [{cls: equal_prob for cls in classes}] * 3
        block_sizes = [80, 70, 50]
    
    block_file = block_generator.generate_blocks_from_real_data(
        df=penguins_df,
        target_col=target_col,
        output_path=OUTPUT_DIR,
        filename='penguins_blocks.csv',
        n_blocks=3,
        block_sizes=block_sizes,
        block_distributions=block_distributions
    )
    
    print(f"  Block-based dataset created: {block_file}")

except Exception as e:
    print(f"Block generation failed: {e}")

# =============================================================================
# SCENARIO 5: Temporal Drift
# =============================================================================
print("\n" + "="*50)
print("SCENARIO 5: Temporal Drift Patterns")
print("="*50)

try:
    # Generate temporal drift
    print("Generating temporal drift in penguins data...")
    
    temporal_file = block_generator.generate_temporal_drift_blocks(
        df=penguins_df,
        target_col=target_col,
        output_path=OUTPUT_DIR,
        filename='penguins_temporal_drift.csv',
        n_blocks=4,
        drift_strength=0.3,
        drift_type='gradual'
    )
    
    print(f"  Temporal drift dataset created: {temporal_file}")

except Exception as e:
    print(f"Temporal drift generation failed: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*50)
print("REAL DATASET GENERATION SUMMARY")
print("="*50)

expected_files = [
    "penguins_resample.csv",
    "penguins_smote.csv",
    "penguins_gmm.csv", 
    "penguins_ctgan.csv",
    "penguins_copula.csv",
    "penguins_feature_drift.csv",
    "penguins_label_drift.csv",
    "penguins_imbalanced.csv",
    "penguins_balanced.csv",
    "penguins_blocks.csv",
    "penguins_temporal_drift.csv"
]

print("Generated datasets:")
success_count = 0
for filename in expected_files:
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  SUCCESS: {filename} ({size:.1f} KB)")
        success_count += 1
    else:
        print(f"  MISSING: {filename}")

print(f"\nSuccess rate: {success_count}/{len(expected_files)} datasets")
print(f"Output directory: {OUTPUT_DIR}")
print("\nReal datasets demonstrate:")
print("- Multiple generation methods (RESAMPLE, SMOTE, GMM, CTGAN, COPULA)")
print("- Drift injection (feature and label drift)")
print("- Distribution manipulation (balancing, imbalancing)")
print("- Block-based generation with varying distributions")
print("- Temporal drift patterns")
print(f"Base dataset: Palmer Penguins ({len(penguins_df)} samples, {len(penguins_df.columns)} features)")
print(f"Target variable: {target_col} with {len(penguins_df[target_col].unique())} classes")
print("\nReady for presentation!")