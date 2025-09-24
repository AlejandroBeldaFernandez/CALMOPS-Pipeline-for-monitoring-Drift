from Synthetic.SyntheticGenerator import SyntheticGenerator

if __name__ == "__main__":
    # Lanza el dashboard de solo lectura en http://localhost:8061
    gen = SyntheticGenerator(port=8061)

    # Genera un CSV en 'salida_tiempo_real/' y abre el dashboard
    gen.generate(
        output_path="salida_tiempo_real",
        filename="dataset_agrawal.csv",
        n_samples=5000,
        method="agrawal",
        method_params={
            "classification_function": 0,
            "perturbation": 0.1
        },
        drift_type="none",
        position_of_drift=None,
        target_col="target",
        balance=False,
        random_state=42
    )