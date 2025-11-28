import os
import sys

# Add the project root to the path so we can import calmops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calmops.data_generators.Synthetic import SyntheticBlockGenerator


def generate_abrupt_drift():
    # Configuración
    OUTPUT_PATH = "data/scenarios/generic_abrupt"
    FILENAME = "abrupt_drift.csv"
    N_BLOCKS = 10
    TOTAL_SAMPLES = 50000

    # Stagger tiene 3 funciones que son ortogonales entre sí.
    # Cambiamos de función bruscamente.
    method_params = [
        {"classification_function_stagger": 0},  # Bloques 1-4: Regla 0
        {"classification_function_stagger": 0},
        {"classification_function_stagger": 0},
        {"classification_function_stagger": 0},
        {"classification_function_stagger": 1},  # Bloque 5: CAMBIO ABRUPTO a Regla 1
        {"classification_function_stagger": 1},
        {"classification_function_stagger": 1},
        {"classification_function_stagger": 2},  # Bloque 8: CAMBIO ABRUPTO a Regla 2
        {"classification_function_stagger": 2},
        {"classification_function_stagger": 2},
    ]

    print(f"Generando escenario de Drift Abrupto (STAGGER)...")
    generator = SyntheticBlockGenerator()
    dataset_path = generator.generate_blocks_simple(
        output_path=OUTPUT_PATH,
        filename=FILENAME,
        n_blocks=N_BLOCKS,
        total_samples=TOTAL_SAMPLES,
        methods="stagger",
        method_params=method_params,
        date_start="2024-01-01",
        date_step={"days": 1},
        date_col="timestamp",
        random_state=42,
        generate_report=False,
    )

    print(f"Dataset de Drift Abrupto generado en: {dataset_path}")


if __name__ == "__main__":
    generate_abrupt_drift()
