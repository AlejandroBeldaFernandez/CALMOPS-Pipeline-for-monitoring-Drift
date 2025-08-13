import os
import pandas as pd
from Real.RealGenerator import RealGenerator

# Cargar dataset real
df = pd.read_csv("penguins_preprocessed_tratado.csv")

# Instanciar el generador con el dataset cargado y la columna objetivo 'Survived'
real_gen = RealGenerator(dataset_path="penguins_preprocessed_tratado.csv", target_col="species")

# Configuraciones para la generación de datos
output_path = "generated-data"
filename_prefix = "penguins"
n_samples = 1000  # Número de muestras a generar
random_state = 42

# Generar datos utilizando diferentes métodos
methods = ["resample", "smote", "gmm", "ctgan", "copula"]
for method in methods:
    # Configuración específica para cada método (si es necesario)
    method_params = {}  # Aquí puedes añadir parámetros específicos si es necesario

    # Definir el nombre del archivo para cada método
    filename = f"{filename_prefix}{method}.csv"

    # Generar datos sintéticos con el método seleccionado
    generated_file = real_gen.generate(
        output_path=output_path,
        filename=filename,
        n_samples=n_samples,
        method=method,
        random_state=random_state,
        method_params=method_params
    )

    print(f"✅ Dataset generado con el método '{method}' y guardado en: {generated_file}")
