# /home/alex/calmops/data_generators/generate_agrawal_no_drift.py
import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data_generators.Synthetic.SyntheticBlockGenerator import SyntheticBlockGenerator

def generate_dataset():
    """
    Generates a synthetic Agrawal dataset with 6 blocks and no drift using SyntheticBlockGenerator.
    """
    # Instantiate the SyntheticBlockGenerator
    block_gen = SyntheticBlockGenerator()

    # Define parameters for generation
    output_dir = project_root
    filename = "synthetic_agrawal_6_blocks_no_drift.csv"
    n_blocks = 6
    total_samples = 6000 # 1000 samples per block

    print(f"Generating dataset with {total_samples} samples in {n_blocks} blocks using SyntheticBlockGenerator...")
    print(f"Output path: {os.path.join(output_dir, filename)}")

    # Generate the dataset using the simplified block generation method
    block_gen.generate_blocks_simple(
        output_path=output_dir,
        filename=filename,
        n_blocks=n_blocks,
        total_samples=total_samples,
        methods="agrawal",
        method_params={},
        random_state=42,
        target_col="target",
        balance=False,
        date_start="2023-01-01", # Example start date for timestamps
        date_step={"days": 1}, # Each block starts a day after the previous one
        date_col="timestamp",
        generate_report=False # Do not generate the report
    )

    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_dataset()