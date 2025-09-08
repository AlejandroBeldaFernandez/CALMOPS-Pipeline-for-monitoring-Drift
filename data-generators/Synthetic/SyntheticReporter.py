import pandas as pd

class SyntheticReporter:
    def _report_dataset(self, df: pd.DataFrame, target_col: str, extra_info: dict = None):
        """Displays statistics of the generated dataset."""

        print("\n === DATASET REPORT ===")
        print(f"Total instances: {len(df)}")
        print(f"Features: {len(df.columns) - 1} (excluding target)")

        # Class distribution
        if target_col in df.columns:
            print("\n Class distribution:")
            dist = df[target_col].value_counts(normalize=True).to_dict()
            for cls, prop in dist.items():
                print(f"  - {cls}: {df[target_col].value_counts()[cls]} "
                      f"({prop:.2%})")

        # Basic feature statistics
        print("\n Feature statistics:")
        print(df.drop(columns=[target_col], errors="ignore").describe().T)

        # Additional information (drift, blocks, etc.)
        if extra_info:
            print("\nExtra information:")
            for key, val in extra_info.items():
                print(f"  - {key}: {val}")

        print("=== END OF REPORT ===\n")