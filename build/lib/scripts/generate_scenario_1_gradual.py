import os
import pandas as pd
import numpy as np
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from calmops.data_generators.Synthetic.GeneratorFactory import (
    GeneratorFactory,
    GeneratorType,
    GeneratorConfig,
)

# Configuration
OUTPUT_PATH = "/home/alex/datos/gradual"
FILENAME = "gradual_drift_dataset.csv"
N_BLOCKS = 12
TOTAL_SAMPLES = 60000
SAMPLES_PER_SEGMENT = 20000  # 3 segments of 20k = 60k
TRANSITION_WIDTH = 5000
DRIFT_POSITION_IN_SEGMENT = 15000  # Drift happens towards the end of the segment


def generate_gradual_scenario():
    print("Generating Gradual Drift Scenario (SEA)...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    factory = GeneratorFactory()
    synth_gen = SyntheticGenerator(random_state=42)

    # Define the sequence of concepts (SEA functions)
    # Segment 1: 0 -> 1
    # Segment 2: 1 -> 2
    # Segment 3: 2 -> 3
    concepts = [0, 1, 2, 3]

    all_dfs = []

    for i in range(len(concepts) - 1):
        concept_a_idx = concepts[i]
        concept_b_idx = concepts[i + 1]

        print(
            f"Generating segment {i + 1}: Transition SEA({concept_a_idx}) -> SEA({concept_b_idx})"
        )

        # Create generators
        config_a = GeneratorConfig(
            function=concept_a_idx, noise_percentage=0.1, random_state=42
        )
        config_b = GeneratorConfig(
            function=concept_b_idx, noise_percentage=0.1, random_state=42
        )

        gen_a = factory.create_generator(GeneratorType.SEA, config_a)
        gen_b = factory.create_generator(GeneratorType.SEA, config_b)

        # Generate segment with gradual drift
        df_segment = synth_gen.generate(
            generator_instance=gen_a,
            drift_generator=gen_b,
            output_path=OUTPUT_PATH,  # Temporary, we won't save individual segments
            filename=f"temp_segment_{i}.csv",
            n_samples=SAMPLES_PER_SEGMENT,
            drift_type="gradual",
            position_of_drift=DRIFT_POSITION_IN_SEGMENT,
            transition_width=TRANSITION_WIDTH,
            target_col="target",
            save_dataset=False,
            generate_report=False,
        )

        all_dfs.append(df_segment)

    # Concatenate all segments
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Add block column (simulated, 5000 samples per block)
    samples_per_block = TOTAL_SAMPLES // N_BLOCKS
    full_df["block"] = (full_df.index // samples_per_block) + 1

    # Save final dataset
    full_path = os.path.join(OUTPUT_PATH, FILENAME)
    full_df.to_csv(full_path, index=False)

    print(f"Generated {len(full_df)} samples at: {full_path}")
    print(
        f"Transitions at indices (approx): {DRIFT_POSITION_IN_SEGMENT}, {SAMPLES_PER_SEGMENT + DRIFT_POSITION_IN_SEGMENT}, {2 * SAMPLES_PER_SEGMENT + DRIFT_POSITION_IN_SEGMENT}"
    )


if __name__ == "__main__":
    generate_gradual_scenario()
