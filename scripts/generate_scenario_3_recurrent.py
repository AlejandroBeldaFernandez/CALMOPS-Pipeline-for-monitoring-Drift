import os
import sys

# Add the project root to the path so we can import calmops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calmops.data_generators.Synthetic import SyntheticBlockGenerator


def generate_recurrent_drift():
    # Configuración
    OUTPUT_PATH = "data/scenarios/generic_recurrent"
    FILENAME = "recurrent_drift.csv"
    N_BLOCKS = 12
    TOTAL_SAMPLES = 60000

    # Agrawal tiene 10 funciones. Usaremos la 0 y la 1 alternativamente.
    # A -> A -> B -> B -> A -> A ...
    method_params = [
        {"classification_function": 0},  # Estado A
        {"classification_function": 0},
        {"classification_function": 1},  # Estado B
        {"classification_function": 1},
        {"classification_function": 0},  # Estado A (Recurrente)
        {"classification_function": 0},
        {"classification_function": 1},  # Estado B (Recurrente)
        {"classification_function": 1},
        {"classification_function": 0},  # Estado A
        {"classification_function": 0},
        {"classification_function": 1},  # Estado B
        {"classification_function": 1},
    ]

    print(f"Generando escenario de Drift Recurrente (AGRAWAL)...")
    generator = SyntheticBlockGenerator()
    dataset_path = generator.generate_blocks_simple(
        output_path=OUTPUT_PATH,
        filename=FILENAME,
        n_blocks=N_BLOCKS,
        total_samples=TOTAL_SAMPLES,
        methods="agrawal",
        method_params=method_params,
        date_start="2024-01-01",
        date_step={"days": 1},
        date_col="timestamp",
        random_state=42,
        generate_report=False,
    )

    print(f"Dataset de Drift Recurrente generado en: {dataset_path}")


if __name__ == "__main__":
    generate_recurrent_drift()
