import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
import os

# Configuración
dataset_path = "generated-data/agrawal_block_drift.csv"
output_dir = "drift_plots"
os.makedirs(output_dir, exist_ok=True)

# Cargar dataset
df = pd.read_csv(dataset_path)
print("✅ Loaded dataset with blocks")
print(f"Shape: {df.shape}\n")

chunks = df["chunk"].unique()
print(f"Chunks found: {chunks}\n")

# Número de instancias por bloque
print("📊 Number of instances per block:")
for chunk_id, subset in df.groupby("chunk"):
    print(f"Block {chunk_id}: {len(subset)} instances")
print("")

# Distribución de target por bloque
print("🎯 Target distribution per block:\n")
for chunk_id, subset in df.groupby("chunk"):
    dist = subset["target"].value_counts(normalize=True)
    print(f"Block {chunk_id} ({len(subset)} instances):")
    print(dist)
    print("")

    plt.figure(figsize=(5,3))
    sns.countplot(x="target", data=subset)
    plt.title(f"Target distribution in Block {chunk_id}")
    plt.savefig(os.path.join(output_dir, f"target_block_{chunk_id}.png"))
    plt.close()

# Comparar features entre el primer y último bloque
features = [col for col in df.columns if col not in ["target", "chunk"]]
block1 = df[df["chunk"] == chunks.min()]
block_last = df[df["chunk"] == chunks.max()]

print("🔎 Drift analysis by feature between first and last block:\n")
for col in features:
    if pd.api.types.is_numeric_dtype(df[col]):
        stat, p = ks_2samp(block1[col], block_last[col])
        print(f"[NUMERIC] {col} | KS-test p={p:.4f}")
    else:
        pre_counts = block1[col].value_counts(normalize=True)
        post_counts = block_last[col].value_counts(normalize=True)
        cats = sorted(set(pre_counts.index).union(post_counts.index))
        chi2, p, _, _ = chi2_contingency([
            [pre_counts.get(c,0) for c in cats],
            [post_counts.get(c,0) for c in cats]
        ])
        print(f"[CATEGORICAL] {col} | Chi²-test p={p:.4f}")

# Mostrar ejemplos de instancias
print("\n📌 Sample instances per block:\n")
for chunk_id in chunks:
    print(f"Block {chunk_id}:")
    print(df[df["chunk"] == chunk_id].head(3))
    print("")

print(f"\n✅ Plots saved in '{output_dir}/'")
