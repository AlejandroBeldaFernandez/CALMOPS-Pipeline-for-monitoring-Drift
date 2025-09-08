import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import numpy as np
from scipy.spatial.distance import jensenshannon

class RealReporter:
    def _report_real_dataset(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str,
        method: str,
        extra_info: dict = None
    ):
        """
        Generates and logs a report comparing the real and synthetic datasets,
        including drift analysis and optional block-level analysis.
        """
        print("\n === REAL DATASET REPORT ===")
        print(f"Method: {method}")
        print(f"Original samples: {len(real_df)} | Synthetic samples: {len(synthetic_df)}")

        # Target column distribution
        if target_col in real_df.columns and target_col in synthetic_df.columns:
            print("\nClass distribution comparison:")
            real_dist = real_df[target_col].value_counts(normalize=True).to_dict()
            synth_dist = synthetic_df[target_col].value_counts(normalize=True).to_dict()
            for cls in sorted(set(real_dist) | set(synth_dist)):
                print(f"  - Class {cls}: real={real_dist.get(cls, 0):.2%}, synth={synth_dist.get(cls, 0):.2%}")
        else:
            print(f"Target column '{target_col}' not found in one or both datasets")

        # Feature statistics
        print("\nSynthetic feature statistics:")
        synthetic_stats = synthetic_df.drop(columns=[target_col, "chunk"], errors="ignore").describe().T
        print(synthetic_stats)

        print("\nOriginal feature statistics:")
        real_stats = real_df.drop(columns=[target_col], errors="ignore").describe().T
        print(real_stats)

        # SDV quality report
        common_cols = [col for col in synthetic_df.columns if col in real_df.columns]
        real_clean = real_df[common_cols].copy()
        synth_clean = synthetic_df[common_cols].copy()

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_clean)

        quality_report = QualityReport()
        quality_report.generate(real_data=real_clean, synthetic_data=synth_clean, metadata=metadata.to_dict())
        print(f"\nQuality Score: {quality_report.get_score():.3f}")
        print("Quality details:")
        for prop in ["Column Shapes", "Column Pair Trends"]:
            print(f"\n🔹 {prop}")
            print(quality_report.get_details(prop).head())

        # Extra info
        if extra_info:
            print("\nExtra information:")
            for key, val in extra_info.items():
                print(f"  - {key}: {val}")

        # --- Drift analysis for entire dataset ---
        print("\n=== DRIFT ANALYSIS (REAL vs SYNTHETIC) ===")
        if target_col in real_df.columns and target_col in synthetic_df.columns:
            p = real_df[target_col].value_counts(normalize=True).sort_index()
            q = synthetic_df[target_col].value_counts(normalize=True).reindex(p.index, fill_value=0)
            js_div = jensenshannon(p, q)**2
            print(f"JS divergence (class distribution): {js_div:.4f}")

            # Numeric features
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns.difference([target_col])
            for col in numeric_cols:
                mean_real = real_df[col].mean()
                mean_synth = synthetic_df[col].mean()
                change = mean_synth - mean_real
                print(f"  Feature '{col}': mean real={mean_real:.4f}, mean synth={mean_synth:.4f}, change={change:.4f}")

        # --- Block analysis if present ---
        if "chunk" in synthetic_df.columns:
            print("\n=== BLOCK ANALYSIS ===")
            blocks = synthetic_df["chunk"].unique()
            for b in blocks:
                block_df = synthetic_df[synthetic_df["chunk"] == b]
                print(f"\n--- Block {b} ---")
                print(f"Samples: {len(block_df)}")
                dist = block_df[target_col].value_counts(normalize=True).to_dict()
                print("Class distribution:")
                for cls, prop in dist.items():
                    print(f"  - {cls}: {len(block_df[block_df[target_col]==cls])} ({prop:.2%})")
                print("Feature statistics:")
                try:
                    print(block_df.drop(columns=[target_col, "chunk"], errors="ignore").describe().T)
                except:
                    print("Could not compute feature statistics.")

            # Drift between consecutive blocks
            print("\n=== DRIFT BETWEEN BLOCKS ===")
            for i in range(len(blocks)-1):
                b1, b2 = blocks[i], blocks[i+1]
                df1 = synthetic_df[synthetic_df["chunk"] == b1]
                df2 = synthetic_df[synthetic_df["chunk"] == b2]
                print(f"\nBlock {b1} → Block {b2}")
                p = df1[target_col].value_counts(normalize=True).sort_index()
                q = df2[target_col].value_counts(normalize=True).reindex(p.index, fill_value=0)
                js_div = jensenshannon(p, q)**2
                print(f"JS divergence (class distribution): {js_div:.4f}")
                numeric_cols = df1.select_dtypes(include=[np.number]).columns.difference([target_col, "chunk"])
                for col in numeric_cols:
                    mean_diff = df2[col].mean() - df1[col].mean()
                    print(f"  Feature '{col}': mean change {mean_diff:.4f}")

        print("\n=== END OF REAL DATASET REPORT ===\n")