import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from river.datasets import synth

# Calmops Imports
from calmops.data_generators.Synthetic.SyntheticGenerator import SyntheticGenerator
from calmops.data_generators.Clinic.Clinic import ClinicGenerator
from calmops.data_generators.Real.RealGenerator import RealGenerator
from calmops.data_generators.DriftInjection.DriftInjector import DriftInjector
from calmops.privacy.privacy import (
    pseudonymize_columns,
    add_laplace_noise,
    generalize_numeric_to_ranges,
    generalize_categorical_by_mapping,
)


# Helper for Clinic Scenario
def build_correlation_matrix(n_demo, group_sizes, correlations):
    n_omics = sum(group_sizes)
    n_total = n_demo + n_omics
    matrix = np.eye(n_total)

    current_idx = n_demo
    for i, size in enumerate(group_sizes):
        end_idx = current_idx + size
        config = correlations[i]

        if "internal" in config:
            internal_val = config["internal"]
            if isinstance(internal_val, (tuple, list)) and len(internal_val) == 2:
                val = np.random.uniform(internal_val[0], internal_val[1])
            else:
                val = float(internal_val)

            if val > 0:
                block = matrix[current_idx:end_idx, current_idx:end_idx]
                block[:] = val
                np.fill_diagonal(block, 1.0)
                matrix[current_idx:end_idx, current_idx:end_idx] = block

        if "demo_idx" in config and "demo_corr" in config:
            demo_idx = config["demo_idx"]
            corr = config["demo_corr"]
            if demo_idx is not None:
                matrix[demo_idx, current_idx:end_idx] = corr
                matrix[current_idx:end_idx, demo_idx] = corr

        current_idx = end_idx
    return matrix


def rename_features(df, prefix, group_sizes, group_names):
    new_columns = []
    current_idx = 0
    for size, name in zip(group_sizes, group_names):
        for i in range(size):
            new_columns.append(f"{prefix}_{name}_{current_idx + i}")
        current_idx += size
    df.columns = new_columns
    return df


def generate_synthetic_agrawal():
    print("=== Scenario 1: Synthetic Agrawal with Abrupt Drift ===")
    output_dir = os.path.join("scenarios", "agrawal")
    os.makedirs(output_dir, exist_ok=True)

    generator = SyntheticGenerator(random_state=42)

    # 1. Generate Dataset 1 (Function 0)
    print("Generating Dataset 1 (Agrawal Function 0)...")
    agrawal_0 = synth.Agrawal(classification_function=0, seed=42)
    path0 = generator.generate(
        generator_instance=agrawal_0,
        n_samples=1000,
        filename="synthetic_agrawal_func0.csv",
        output_path=output_dir,
        generate_report=False,
    )
    df0 = pd.read_csv(path0)

    # 2. Generate Dataset 2 (Function 1)
    print("Generating Dataset 2 (Agrawal Function 1)...")
    agrawal_1 = synth.Agrawal(classification_function=1, seed=42)
    path1 = generator.generate(
        generator_instance=agrawal_1,
        n_samples=200,
        filename="synthetic_agrawal_func1.csv",
        output_path=output_dir,
        generate_report=False,
    )
    df1 = pd.read_csv(path1)

    # 3. Model Training & Evaluation
    print("Training and Evaluating Model...")

    target_col = df0.columns[-1]
    print(f"Target column identified as: {target_col}")

    X0 = df0.drop(columns=[target_col])
    y0 = df0[target_col]

    X1 = df1.drop(columns=[target_col])
    y1 = df1[target_col]

    # Split Dataset 1
    X_train, X_test, y_train, y_test = train_test_split(
        X0, y0, test_size=0.2, random_state=42
    )

    # Train Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on Test 1 (Baseline)
    y_pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average="weighted")

    # Evaluate on Dataset 2 (Drift)
    y_pred_drift = clf.predict(X1)
    acc_drift = accuracy_score(y1, y_pred_drift)
    f1_drift = f1_score(y1, y_pred_drift, average="weighted")

    results = {
        "scenario": "Agrawal Abrupt Drift",
        "baseline_test": {"accuracy": acc_test, "f1_score": f1_test},
        "drift_dataset": {"accuracy": acc_drift, "f1_score": f1_drift},
        "drift_impact": {
            "accuracy_drop": acc_test - acc_drift,
            "f1_drop": f1_test - f1_drift,
        },
    }

    results_path = os.path.join(output_dir, "agrawal_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")

    # Plot Comparison
    metrics = ["Accuracy", "F1 Score"]
    baseline_vals = [acc_test, f1_test]
    drift_vals = [acc_drift, f1_drift]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, baseline_vals, width, label="Baseline (Test)")
    plt.bar(x + width / 2, drift_vals, width, label="Drifted (Dataset 2)")

    plt.ylabel("Score")
    plt.title("Model Performance Comparison: Baseline vs Drift")
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend()

    plt.savefig(os.path.join(output_dir, "agrawal_comparison.png"))
    plt.close()
    print("Comparison plot saved.")


def generate_clinic_patients():
    print("=== Scenario 2: Clinic with Correlations ===")
    output_dir = os.path.join("scenarios", "clinic")
    os.makedirs(output_dir, exist_ok=True)

    generator = ClinicGenerator(seed=42)
    n_samples = 1000

    # 1. Generate Demographics
    print("Generating Demographics...")
    custom_demo_cols = {
        "Age": {
            "distribution": "truncnorm",
            "a": -2.0,
            "b": 2.5,
            "loc": 60,
            "scale": 10,
        },
        "RIN": {"distribution": "truncnorm", "a": -3.0, "b": 2.0, "loc": 6, "scale": 2},
        "Lote": {"distribution": "randint", "low": 1, "high": 5},
        "Sex": {"distribution": "binom", "n": 1, "p": 0.5},
    }
    demographic_df, raw_demographic_data = generator.generate_demographic_data(
        n_samples=n_samples,
        custom_demographic_columns=custom_demo_cols,
    )

    # Identify conditioning columns
    cond_cols = [
        c for c in raw_demographic_data.columns if c != "Patient_ID" and c != "Group"
    ]
    col_to_idx = {col: i for i, col in enumerate(cond_cols)}
    n_demo = len(cond_cols)

    # 2. Define Gene Scenario
    # Minimal counts as requested (Total 20)
    gene_group_sizes = [2, 3, 5, 10]
    gene_group_names = ["GroupA", "GroupB", "GroupD", "Ruido"]
    n_genes = sum(gene_group_sizes)

    gene_correlations_config = [
        {"internal": (0.3, 0.6), "demo_idx": col_to_idx.get("Age"), "demo_corr": 0.4},
        {"internal": 0.3, "demo_idx": col_to_idx.get("Sex"), "demo_corr": 0.4},
        {"internal": 0.3},
        {"internal": 0.0},
    ]

    gene_corr_matrix = build_correlation_matrix(
        n_demo, gene_group_sizes, gene_correlations_config
    )

    print(f"Generating {n_genes} Genes...")
    genes_df = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type="Microarray",
        demographic_df=demographic_df,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demographic_data,
        demographic_gene_correlations=gene_corr_matrix,
        n_samples=n_samples,
    )
    genes_df = rename_features(genes_df, "Gene", gene_group_sizes, gene_group_names)

    # 3. Define Protein Scenario
    # Minimal counts as requested (Total 20)
    prot_group_sizes = [6, 1, 8, 5]
    prot_group_names = ["GroupA", "GroupB", "GroupD", "Ruido"]
    n_proteins = sum(prot_group_sizes)

    prot_correlations_config = [
        {"internal": 0.45, "demo_idx": col_to_idx.get("Age"), "demo_corr": 0.4},
        {"internal": 0.3, "demo_idx": col_to_idx.get("Sex"), "demo_corr": 0.4},
        {"internal": 0.3},
        {"internal": 0.0},
    ]

    prot_corr_matrix = build_correlation_matrix(
        n_demo, prot_group_sizes, prot_correlations_config
    )

    print(f"Generating {n_proteins} Proteins...")
    prot_df = generator.generate_protein_data(
        n_proteins=n_proteins,
        demographic_df=demographic_df,
        demographic_id_col="Patient_ID",
        raw_demographic_data=raw_demographic_data,
        demographic_protein_correlations=prot_corr_matrix,
        n_samples=n_samples,
    )
    prot_df = rename_features(prot_df, "Prot", prot_group_sizes, prot_group_names)

    # 4. Generate Target (Diagnosis)
    print("Generating Diagnosis...")
    weights = {}
    if "Age" in raw_demographic_data.columns:
        weights["Age"] = 0.3
    if "Sex" in raw_demographic_data.columns:
        weights["Sex"] = 0.1

    # Add Group A weights (Genes) - 0.2 distributed
    group_a_genes = [c for c in genes_df.columns if "GroupA" in c]
    for col in group_a_genes:
        weights[col] = 0.2 / len(group_a_genes)

    # Add Group B weights (Genes) - 0.5 distributed
    group_b_genes = [c for c in genes_df.columns if "GroupB" in c]
    for col in group_b_genes:
        weights[col] = 0.5 / len(group_b_genes)

    diagnosis = generator.generate_target_variable(
        demographic_df=raw_demographic_data,
        omics_dfs=pd.concat([genes_df, prot_df], axis=1),
        weights=weights,
        binary_threshold=0.0,
    )
    diagnosis.name = "diagnosis"

    # 5. Save Data
    raw_demographic_data["diagnosis"] = diagnosis
    genes_df["diagnosis"] = diagnosis
    prot_df["diagnosis"] = diagnosis

    raw_demographic_data.to_csv(os.path.join(output_dir, "clinic_patients.csv"))
    genes_df.to_csv(os.path.join(output_dir, "clinic_genes.csv"))
    prot_df.to_csv(os.path.join(output_dir, "clinic_proteins.csv"))
    print("Clinic data saved.")

    # 6. Visualizations (Correlation Matrices)
    print("Generating Correlation Plots...")

    # Matrix 1: Demo + Genes + Proteins
    # Since we have ~9000 features now, plotting full matrix is impossible/useless.
    # We should plot a subset or averages per group.
    # Let's plot Demo + Averages of each Group.

    # Matrix 1: Demo + Genes (No Proteins, No Grouping)
    # Plot individual features as requested
    full_df = pd.concat(
        [
            raw_demographic_data,
            genes_df.drop(columns=["diagnosis"]),
        ],
        axis=1,
    )
    numeric_df = full_df.select_dtypes(include=[np.number])

    # Make plot larger as requested
    plt.figure(figsize=(24, 20))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Correlation Matrix: Demographics + Genes")
    plt.savefig(os.path.join(output_dir, "clinic_correlations_genes_demo.png"))
    plt.close()
    print("Correlation plots saved.")


def generate_real_benchmark():
    print("=== Scenario 4: Real (Iris) Drift Robustness Benchmark ===")
    output_dir = os.path.join("scenarios", "real")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Iris
    print("Loading Iris Dataset...")
    iris = load_iris(as_frame=True)
    real_df = iris.frame
    target_col = "target"

    # 2. Train Models on ALL Original Data
    print("Training SVC on ALL Original Data...")
    models = {"SVC": SVC(random_state=42)}

    X_real = real_df.drop(columns=[target_col])
    y_real = real_df[target_col]

    for name, model in models.items():
        model.fit(X_real, y_real)

    # 3. Generate Synthetic Data
    synthetic_datasets = {}
    methods = ["tvae"]

    for method in methods:
        print(f"Generating Synthetic Data using {method.upper()}...")
        try:
            gen = RealGenerator(original_data=real_df, method=method, auto_report=False)
            # Generate same amount as original
            syn_df = gen.synthesize(
                n_samples=len(real_df), output_dir=output_dir, save_dataset=False
            )
            synthetic_datasets[method] = syn_df
        except Exception as e:
            print(f"Generation failed for {method}: {e}")

    # 4. Visualizations (PCA & Distributions)
    print("Generating Visualizations...")

    # PCA Comparison
    plt.figure(figsize=(15, 5))
    pca = PCA(n_components=2)

    # Fit PCA on Real Data
    pca.fit(X_real)

    real_pca = pca.transform(X_real)

    for i, (method, syn_df) in enumerate(synthetic_datasets.items()):
        plt.subplot(1, 1, i + 1)

        # Plot Real
        plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, label="Real", c="blue")

        # Plot Synthetic
        if not syn_df.empty:
            syn_X = syn_df.drop(columns=[target_col])
            syn_pca = pca.transform(syn_X)
            plt.scatter(
                syn_pca[:, 0], syn_pca[:, 1], alpha=0.3, label="Synthetic", c="orange"
            )

        plt.title(f"PCA: Real vs {method.upper()}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_pca_comparison.png"))
    plt.close()

    # Distribution Comparison (Feature: sepal width (cm))
    feature_to_plot = "sepal width (cm)"
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        real_df[feature_to_plot], label="Real", fill=True, color="blue", alpha=0.2
    )

    colors = {"tvae": "purple"}
    for method, syn_df in synthetic_datasets.items():
        if not syn_df.empty:
            sns.kdeplot(
                syn_df[feature_to_plot],
                label=f"Synthetic ({method.upper()})",
                linestyle="--",
                color=colors.get(method, "black"),
            )

    plt.title(f"Distribution Comparison: {feature_to_plot}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "real_distribution_comparison.png"))
    plt.close()

    # 5. Inject Drift & Evaluate
    print("Injecting Drift and Evaluating...")

    results = []

    # Initialize Injector (dummy init for access to methods)
    injector = DriftInjector(
        original_df=real_df, output_dir=output_dir, generator_name="drift_injector"
    )

    for method, syn_df in synthetic_datasets.items():
        if syn_df.empty:
            continue

        # A. Evaluate on Synthetic (No Drift)
        X_syn = syn_df.drop(columns=[target_col])
        y_syn = syn_df[target_col]

        # B. Inject Drift (Create Drifted Test Set)
        # We drift the WHOLE synthetic dataset to create a "Drifted Test Set"
        drifted_df = injector.inject_feature_drift_abrupt(
            df=syn_df,
            feature_cols=["sepal width (cm)"],
            drift_type="shift",
            drift_magnitude=1.5,  # Strong drift
            change_index=0,  # Apply to ALL data
            direction="up",
        )

        X_drift = drifted_df.drop(columns=[target_col])
        y_drift = drifted_df[target_col]

        for model_name, model in models.items():
            # Evaluate Clean
            y_pred = model.predict(X_syn)
            f1_no_drift = f1_score(y_syn, y_pred, average="weighted")

            # Evaluate Drifted
            y_pred_drift = model.predict(X_drift)
            f1_drift = f1_score(y_drift, y_pred_drift, average="weighted")

            results.append(
                {
                    "Method": method.upper(),
                    "Model": model_name,
                    "F1_No_Drift": f1_no_drift,
                    "F1_Drift": f1_drift,
                    "Drop": f1_no_drift - f1_drift,
                }
            )
            print(
                f"  [{method.upper()}] {model_name}: F1 Clean={f1_no_drift:.3f}, F1 Drift={f1_drift:.3f}, Drop={f1_no_drift - f1_drift:.3f}"
            )

        # Store for post-drift visualizations (using the last valid method's data)
        last_syn_df = syn_df
        last_drifted_df = drifted_df
        last_method = method

    # 6. Visualizations (Post-Drift)
    print("Generating Post-Drift Visualizations...")

    # A. PCA Post-Drift (using the last method's data, e.g., TVAE)
    if not last_drifted_df.empty:
        plt.figure(figsize=(10, 5))

        # Transform Real Data again to be sure
        real_pca_post = pca.transform(X_real)
        plt.scatter(
            real_pca_post[:, 0], real_pca_post[:, 1], alpha=0.3, label="Real", c="blue"
        )

        # Transform Drifted Data
        drifted_X = last_drifted_df.drop(columns=[target_col])
        drifted_pca = pca.transform(drifted_X)
        plt.scatter(
            drifted_pca[:, 0],
            drifted_pca[:, 1],
            alpha=0.3,
            label="Synthetic (Post-Drift)",
            c="red",
        )

        plt.title(f"PCA: Real vs {last_method.upper()} (Post-Drift)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "real_pca_postdrift.png"))
        plt.close()

    # B. Correlation Comparison (Real vs Pre vs Post)
    if not last_syn_df.empty and not last_drifted_df.empty:
        plt.figure(figsize=(20, 6))

        # Real
        plt.subplot(1, 3, 1)
        sns.heatmap(
            real_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1
        )
        plt.title("Real Data Correlations")

        # Synthetic Pre-Drift
        plt.subplot(1, 3, 2)
        sns.heatmap(
            last_syn_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1
        )
        plt.title(f"Synthetic ({last_method.upper()} Pre-Drift) Correlations")

        # Synthetic Post-Drift
        plt.subplot(1, 3, 3)
        sns.heatmap(
            last_drifted_df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        plt.title(f"Synthetic ({last_method.upper()} Post-Drift) Correlations")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "real_correlations_comparison.png"))
        plt.close()

    # 7. Save Results & Plot Performance
    results_file = os.path.join(output_dir, "real_robustness_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

    if results:
        results_df = pd.DataFrame(results)

        # Plot: Pre vs Post Performance (Bar Chart)
        # Melt for plotting: Method+Model vs F1 Score (Pre/Post)
        melted_res = results_df.melt(
            id_vars=["Method", "Model"],
            value_vars=["F1_No_Drift", "F1_Drift"],
            var_name="Condition",
            value_name="F1_Score",
        )

        # Rename conditions for clarity
        melted_res["Condition"] = melted_res["Condition"].replace(
            {"F1_No_Drift": "Pre-Drift", "F1_Drift": "Post-Drift"}
        )

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=melted_res, x="Model", y="F1_Score", hue="Condition", palette="viridis"
        )
        plt.title("Drift Impact: Model Performance (Pre vs Post)")
        plt.ylim(0, 1.1)
        plt.ylabel("F1 Score")
        plt.savefig(os.path.join(output_dir, "real_drift_impact.png"))
        plt.close()

    print("Real scenario completed.")


def generate_privacy_demo():
    print("=== Scenario 5: Privacy Showcase ===")
    output_dir = os.path.join("scenarios", "privacy")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create Dataset
    data = {
        "ID": [f"User_{i:03d}" for i in range(20)],
        "Name": [f"Person_{i}" for i in range(20)],
        "Age": np.random.randint(20, 70, 20),
        "Salary": np.random.randint(30000, 100000, 20),
        "City": np.random.choice(["Madrid", "Barcelona", "Valencia", "Sevilla"], 20),
    }
    df = pd.DataFrame(data)

    # 2. Apply Privacy
    print("Applying Privacy Techniques...")
    df_priv = df.copy()

    # Pseudonymization
    df_priv = pseudonymize_columns(df_priv, columns=["ID", "Name"], salt="s3cr3t")

    # Generalization (Age)
    df_priv = generalize_numeric_to_ranges(df_priv, columns=["Age"], num_bins=4)

    # Differential Privacy (Salary)
    df_priv = add_laplace_noise(df_priv, columns=["Salary"], epsilon=0.1)
    df_priv["Salary"] = df_priv["Salary"].astype(int)

    # Generalization (City)
    city_map = {
        "Madrid": "Central",
        "Barcelona": "East",
        "Valencia": "East",
        "Sevilla": "South",
    }
    df_priv = generalize_categorical_by_mapping(
        df_priv, columns=["City"], mapping=city_map
    )

    # Save
    df_priv.to_csv(os.path.join(output_dir, "privacy_demo.csv"), index=False)

    # 3. Comparison
    # Interleave columns for better readability
    comparison = pd.DataFrame()
    for col in df.columns:
        comparison[f"{col}_Orig"] = df[col]
        comparison[f"{col}_Priv"] = df_priv[col]

    comparison = comparison.head(10)

    # Save as TXT
    with open(os.path.join(output_dir, "privacy_comparison.txt"), "w") as f:
        f.write(comparison.to_string())

    # Save as PNG
    # Use a much larger figure and tighter layout
    plt.figure(figsize=(20, 8))
    plt.axis("off")

    # Create table
    tbl = plt.table(
        cellText=comparison.values,
        colLabels=comparison.columns,
        loc="center",
        cellLoc="center",
    )

    # Adjust font size and scale
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.0)  # Scale width and height

    plt.title("Privacy Comparison (First 10 Rows)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "privacy_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Privacy scenario completed.")


if __name__ == "__main__":
    generate_synthetic_agrawal()
    generate_clinic_patients()
    generate_real_benchmark()
    generate_privacy_demo()
