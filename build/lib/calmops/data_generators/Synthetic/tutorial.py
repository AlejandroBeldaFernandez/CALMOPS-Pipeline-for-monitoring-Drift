import pandas as pd
from calmops.data_generators.Synthetic.SyntheticBlockGenerator import (
    SyntheticBlockGenerator,
)
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
import os


def run_tutorial():
    print("=== SyntheticGenerator Tutorial ===")

    # 1. Simple Stream Generation
    print("\nGenerating simple SEA stream...")
    # Initialize the wrapper
    generator = SyntheticGenerator(random_state=42)

    # Create the River generator instance
    from river.datasets import synth

    sea_gen = synth.SEA(seed=42)

    # Generate 100 samples
    stream_data = generator.generate(
        generator_instance=sea_gen,
        n_samples=100,
        filename="simple_sea.csv",
        output_path="tutorial_output",
        generate_report=False,
    )

    # 2. Block Generation with Drift
    print("\nGenerating blocks with concept drift...")
    block_gen = SyntheticBlockGenerator()

    output_dir = "synthetic_tutorial_output"
    os.makedirs(output_dir, exist_ok=True)

    # Define a scenario:
    # Block 1: SEA concept 1
    # Block 2: SEA concept 2 (Abrupt Drift)
    # Block 3: SEA concept 1 (Recurrent Drift)

    block_gen.generate_blocks_simple(
        output_path=output_dir,
        filename="drift_scenario.csv",
        n_blocks=3,
        total_samples=3000,  # 1000 per block
        methods="sea",
        method_params=[{"function": 1}, {"function": 2}, {"function": 1}],
        date_start="2024-01-01",
        date_step={"days": 1},
        generate_report=False,
    )

    print(f"\nBlock data saved to {output_dir}/drift_scenario.csv")

    # Load and verify
    df = pd.read_csv(os.path.join(output_dir, "drift_scenario.csv"))
    print("Combined Data Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Block counts:\n", df["block"].value_counts())


if __name__ == "__main__":
    run_tutorial()
