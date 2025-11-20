#!/usr/bin/env python3
"""
Infinitely generates Agrawal datasets every hour with varying parameters.

This script runs in a continuous loop to generate synthetic datasets for simulating
a real-time data stream. Each dataset is created with a different random seed and
cycles through the available Agrawal classification functions.
"""

import time
import random
from pathlib import Path
from datetime import datetime

# It's good practice to handle potential import errors if the script is moved
try:
    from calmops.data_generators.Synthetic import (
        SyntheticGenerator,
        GeneratorFactory,
        GeneratorConfig,
        GeneratorType,
    )
except ImportError as e:
    print(f"Error: Could not import CalmOps modules. Make sure the script is run from the project root.")
    print(f"Details: {e}")
    exit(1)


def main():
    """
    Main function to run the infinite data generation loop.
    """
    output_dir = Path("/home/alex/datos")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory configured at: {output_dir.resolve()}")

    iteration = 0
    while True:
        try:
            # --- 1. Configure Parameters for this run ---
            # Use a new random seed for each dataset
            current_seed = random.randint(0, 1_000_000)
            
            # Cycle through the 10 Agrawal classification functions
            current_function = iteration % 10
            
            print(f"\n--- Starting new generation cycle ({iteration + 1}) ---")
            print(f"  - Seed: {current_seed}")
            print(f"  - Classification Function: {current_function}")

            # --- 2. Create the Agrawal generator instance ---
            factory = GeneratorFactory()
            config = GeneratorConfig(
                random_state=current_seed,
                classification_function=current_function
            )
            agrawal_generator = factory.create_generator(GeneratorType.AGRAWAL, config)

            # --- 3. Use SyntheticGenerator to create the dataset ---
            synthetic_generator = SyntheticGenerator()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agrawal_{timestamp}_seed{current_seed}_func{current_function}.csv"
            
            print(f"  - Generating 10,000 samples into '{filename}'...")

            synthetic_generator.generate(
                generator_instance=agrawal_generator,
                output_path=str(output_dir),
                filename=filename,
                n_samples=10000,
                drift_type="none",
                save_dataset=True,       # Ensure the dataset is saved to a file
                generate_report=False,   # Disable report generation for efficiency
            )
            
            print(f"  - Dataset successfully generated and saved.")

            # --- 4. Wait for the next cycle ---
            iteration += 1
            print(f"--- Cycle complete. Waiting for 1 hour before next generation. ---")
            time.sleep(3600)  # 3600 seconds = 1 hour

        except KeyboardInterrupt:
            print("\nScript interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Waiting for 1 hour before trying again.")
            time.sleep(3600)


if __name__ == "__main__":
    main()
