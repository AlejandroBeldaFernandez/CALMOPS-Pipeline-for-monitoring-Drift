import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# Paths
df = pd.read_csv("generated-data/titanic_block_drift.csv")

print("✅ Loaded dataset with blocks")
print(f"Shape: {df.shape}\n")

# Identificar chunks
chunks = df["chunk"].unique()
print(f"Chunks found: {chunks}\n")

# Instancias por bloque
print("📊 Number of instances per block:")
for c in chunks:
    print(f"Block {c}: {len(df[df['chunk']==c])} samples")
print()

# Distribución del target por bloque
print("🎯 Target distribution per block:")
for c in chunks:
    dist = df[df["chunk"] == c]["Survived"].value_counts(normalize=True)
    print(f"\nBlock {c}:")
    print(dist)

# Analizar drift entre primer y último bloque
print("\n🔎 Drift analysis between Block 1 and Block", chunks[-1], ":\n")
df1 = df[df["chunk"] == chunks[0]]
df_last = df[df["chunk"] == chunks[-1]]

for col in df.columns:
    if col in ["chunk", "Survived"]:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        stat, p = ks_2samp(df1[col], df_last[col])
        print(f"[NUMERIC] {col} | KS-test p={p:.4f}")
    else:
        pre_counts = df1[col].value_counts(normalize=True)
        post_counts = df_last[col].value_counts(normalize=True)
        cats = sorted(set(pre_counts.index).union(post_counts.index))
        chi2, p, _, _ = chi2_contingency([
            [pre_counts.get(c, 0) for c in cats],
            [post_counts.get(c, 0) for c in cats]
        ])
        print(f"[CATEGORICAL] {col} | Chi²-test p={p:.4f}")

# Concept drift: distribución condicional del target por sexo
if "Sex" in df.columns:
    print("\n📊 Conditional distribution of Survived given Sex:")
    ct1 = pd.crosstab(df1["Sex"], df1["Survived"], normalize="index")
    ct_last = pd.crosstab(df_last["Sex"], df_last["Survived"], normalize="index")
    print(f"\nBlock {chunks[0]}:\n", ct1)
    print(f"\nBlock {chunks[-1]}:\n", ct_last)

print("\n✅ Drift block analysis completed")
