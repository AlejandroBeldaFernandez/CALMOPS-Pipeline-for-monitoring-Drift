#!/usr/bin/env python3
"""
Create simplified synthetic datasets according to specifications
"""

import os
import sys
import pandas as pd

# Add the current directory to path for imports
sys.path.append('/home/alex/calmops/data-generators')

from Synthetic.GeneratorFactory import GeneratorFactory, GeneratorConfig, GeneratorType
from Synthetic.SyntheticGenerator import SyntheticGenerator
from Synthetic.BlockDriftGenerator import BlockDriftGenerator

# Setup directories
OUTPUT_DIR = "presentation_synthetic_datasets"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

for directory in [OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

print("CalmOps Synthetic Dataset Generation")
print("=" * 50)

# Initialize generators
synthetic_gen = SyntheticGenerator()
block_gen = BlockDriftGenerator()

# =============================================================================
# DATASET 1: Standard dataset generated with Hyperplane
# =============================================================================
print("1. Standard Hyperplane Dataset")
print("-" * 30)

config_hyperplane = GeneratorConfig(
    random_state=42,
    n_features=10,
    mag_change=0.0,
    noise_percentage_hyperplane=0.05,
    sigma=0.1
)

gen_hyperplane = GeneratorFactory.create_generator(GeneratorType.HYPERPLANE, config_hyperplane)

try:
    hyperplane_path = synthetic_gen.generate(
        generator_instance=gen_hyperplane,
        output_path=OUTPUT_DIR,
        filename="standard_hyperplane.csv",
        n_samples=2000,
        drift_type="none",
        target_col="target",
        extra_info={
            "generator_type": "HYPERPLANE",
            "description": "Standard 10-dimensional hyperplane dataset"
        }
    )
    print(f"  Generated: standard_hyperplane.csv")
except Exception as e:
    print(f"  Failed: {str(e)[:100]}...")

print()

# =============================================================================
# DATASET 2: SEA Balanced Dataset
# =============================================================================
print("2. SEA Balanced Dataset")
print("-" * 30)

config_sea = GeneratorConfig(
    random_state=42,
    function=1,
    noise_percentage=0.1
)

gen_sea = GeneratorFactory.create_generator(GeneratorType.SEA, config_sea)

try:
    sea_path = synthetic_gen.generate(
        generator_instance=gen_sea,
        output_path=OUTPUT_DIR,
        filename="sea_balanced.csv",
        n_samples=1800,
        ratio_before={0: 0.5, 1: 0.5},  # Balanced classes
        drift_type="none",
        target_col="target",
        extra_info={
            "generator_type": "SEA",
            "description": "Balanced SEA dataset with function 1"
        }
    )
    print(f"  Generated: sea_balanced.csv")
except Exception as e:
    print(f"  Failed: {str(e)[:100]}...")

print()

# =============================================================================
# DATASET 3: AGRAWAL with Concept Drift
# =============================================================================
print("3. AGRAWAL Concept Drift Dataset")
print("-" * 30)

config_agrawal_base = GeneratorConfig(
    random_state=42,
    classification_function=0,
    perturbation=0.05
)

config_agrawal_drift = GeneratorConfig(
    random_state=42,
    classification_function=2,  # Different function for concept drift
    perturbation=0.15
)

gen_agrawal_base = GeneratorFactory.create_generator(GeneratorType.AGRAWAL, config_agrawal_base)
gen_agrawal_drift = GeneratorFactory.create_generator(GeneratorType.AGRAWAL, config_agrawal_drift)

try:
    concept_path = synthetic_gen.generate(
        generator_instance=gen_agrawal_base,
        generator_instance_drift=gen_agrawal_drift,
        output_path=OUTPUT_DIR,
        filename="agrawal_concept_drift.csv",
        n_samples=2200,
        position_of_drift=1100,
        drift_type="concept",
        target_col="target",
        extra_info={
            "generator_type": "AGRAWAL",
            "description": "Concept drift from function 0 to 2",
            "drift_position": 1100
        }
    )
    print(f"  Generated: agrawal_concept_drift.csv")
except Exception as e:
    print(f"  Failed: {str(e)[:100]}...")

print()

# =============================================================================
# DATASET 4: AGRAWAL with Data Drift
# =============================================================================
print("4. AGRAWAL Data Drift Dataset")
print("-" * 30)

config_agrawal_data = GeneratorConfig(
    random_state=42,
    classification_function=1,
    perturbation=0.1
)

gen_agrawal_data = GeneratorFactory.create_generator(GeneratorType.AGRAWAL, config_agrawal_data)

try:
    data_path = synthetic_gen.generate(
        generator_instance=gen_agrawal_data,
        output_path=OUTPUT_DIR,
        filename="agrawal_data_drift.csv",
        n_samples=2400,
        position_of_drift=1200,
        ratio_before={0: 0.7, 1: 0.3},  # Initial distribution
        ratio_after={0: 0.3, 1: 0.7},   # After drift distribution
        drift_type="data",
        target_col="target",
        extra_info={
            "generator_type": "AGRAWAL",
            "description": "Data drift from 70/30 to 30/70 distribution",
            "drift_position": 1200
        }
    )
    print(f"  Generated: agrawal_data_drift.csv")
except Exception as e:
    print(f"  Failed: {str(e)[:100]}...")

print()

# =============================================================================
# DATASET 5: 7-Block AGRAWAL with Multiple Drift Types
# =============================================================================
print("5. AGRAWAL 7-Block Multi-Drift Dataset")
print("-" * 30)

# Create generators for different blocks
generators = []
block_sizes = [500, 500, 600, 400, 550, 300, 450]  # Different sizes, blocks 1&2 same
class_ratios = []

for i in range(7):
    if i == 0:  # Block 1: Balanced
        config = GeneratorConfig(random_state=42+i, classification_function=0, perturbation=0.05)
        class_ratios.append({0: 0.5, 1: 0.5})
    elif i == 1:  # Block 2: Same size as block 1, no drift
        config = GeneratorConfig(random_state=42+i, classification_function=0, perturbation=0.05)
        class_ratios.append({0: 0.6, 1: 0.4})
    elif i == 2:  # Block 3: Concept drift
        config = GeneratorConfig(random_state=42+i, classification_function=1, perturbation=0.1)
        class_ratios.append({0: 0.65, 1: 0.35})
    elif i == 3:  # Block 4: No drift
        config = GeneratorConfig(random_state=42+i, classification_function=1, perturbation=0.1)
        class_ratios.append({0: 0.6, 1: 0.4})
    elif i == 4:  # Block 5: Concept drift
        config = GeneratorConfig(random_state=42+i, classification_function=2, perturbation=0.15)
        class_ratios.append({0: 0.4, 1: 0.6})
    elif i == 5:  # Block 6: No drift
        config = GeneratorConfig(random_state=42+i, classification_function=2, perturbation=0.15)
        class_ratios.append({0: 0.45, 1: 0.55})
    else:  # Block 7: Data drift
        config = GeneratorConfig(random_state=42+i, classification_function=2, perturbation=0.15)
        class_ratios.append({0: 0.2, 1: 0.8})  # Strong data drift
    
    generators.append(GeneratorFactory.create_generator(GeneratorType.AGRAWAL, config))

try:
    multiblock_path = block_gen.generate_blocks(
        output_path=OUTPUT_DIR,
        filename="agrawal_7blocks_multidrift.csv",
        n_blocks=7,
        total_samples=sum(block_sizes),
        instances_per_block=block_sizes,
        generators=generators,
        class_ratios=class_ratios,
        target_col="target",
        extra_info={
            "generator_type": "AGRAWAL",
            "description": "7-block dataset with concept drift in blocks 3,5 and data drift in block 7",
            "block_sizes": block_sizes,
            "drift_blocks": [3, 5, 7]
        }
    )
    print(f"  Generated: agrawal_7blocks_multidrift.csv")
except Exception as e:
    print(f"  Failed: {str(e)[:100]}...")

print()

# =============================================================================
# SUMMARY
# =============================================================================
print("Dataset Generation Summary")
print("=" * 50)

datasets = [
    "standard_hyperplane.csv",
    "sea_balanced.csv",
    "agrawal_concept_drift.csv", 
    "agrawal_data_drift.csv",
    "agrawal_7blocks_multidrift.csv"
]

print("Generated datasets:")
for dataset in datasets:
    file_path = os.path.join(OUTPUT_DIR, dataset)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {dataset} ({size:.1f} KB)")
    else:
        print(f"  - {dataset} (missing)")

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Visualizations: {PLOTS_DIR}")
print("\nAll datasets ready for presentation!")