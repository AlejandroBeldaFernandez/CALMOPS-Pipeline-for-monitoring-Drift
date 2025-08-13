from river.datasets import synth
from Synthetic.SyntheticGenerator import SyntheticGenerator

# Crear carpeta de salida
output_dir = "generated-data"

# Instanciar el generador
generator = SyntheticGenerator()

# Definir dos generadores Agrawal con diferentes classification_function
agrawal_1 = synth.Agrawal(seed=42, classification_function=0)
agrawal_2 = synth.Agrawal(seed=42, classification_function=4)

ratio_before = {0: 0.8, 1: 0.2}  # Antes del drift
ratio_after = {0: 0.2, 1: 0.8}   # Después del drift

generator.generate(
    generator_instance=agrawal_1,
    generator_instance_drift=agrawal_2,
    output_path=output_dir,
    filename="concept-drift.csv",
    n_samples=10000,
    drift_type="concept",
    position_of_drift=5000)

