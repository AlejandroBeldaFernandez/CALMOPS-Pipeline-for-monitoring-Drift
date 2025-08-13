import os
import pandas as pd
from collections import defaultdict
from .SyntheticReporter import SyntheticReporter

class BlockDriftGenerator(SyntheticReporter):
    def _ensure_list(self, value, n_blocks):
        """Ensures the parameter is a list of length n_blocks, handling both single values and lists."""
        if isinstance(value, list):
            if len(value) == 1:
                return value * n_blocks  # Repeat the value to match the number of blocks
            elif len(value) != n_blocks:
                raise ValueError(f"Expected list of length {n_blocks}, got {len(value)}")
            return value
        else:
            return [value] * n_blocks  # Convert single value into a list with n_blocks elements

    def generate_blocks(
        self,
        output_path: str,
        filename: str,
        n_blocks: int,
        total_samples: int,
        instances_per_block,
        generators,
        class_ratios=None,
        target_col="target",
        balance: bool = False
    ) -> str:
        """
        Generates synthetic data divided into blocks using specified generator instances.
        It enforces exact class ratios or balancing if requested.
        """
        # Ensure that inputs are valid and consistent
        instances_per_block = self._ensure_list(instances_per_block, n_blocks)
        generators = self._ensure_list(generators, n_blocks)
        class_ratios = self._ensure_list(class_ratios, n_blocks) if class_ratios else [None]*n_blocks

        if sum(instances_per_block) != total_samples:
            raise ValueError(f"Total samples ({total_samples}) must equal the sum of instances per block ({sum(instances_per_block)})")

        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)

        data = []  # To store all generated data
        cols = None  # Column names will be determined after the first sample is generated
        block_reports = []  # Store information about each block

        for i in range(n_blocks):
            gen = generators[i]
            n_samples_block = instances_per_block[i]

            block_data = []
            gen_iter = iter(gen.take(n_samples_block * 10))  # Extra margin to ensure enough data

            # Auto-balance the data if balance is enabled
            if balance:
                preview = [next(gen_iter) for _ in range(50)]
                classes = set([y for _, y in preview])
                gen_iter = iter(preview + list(gen.take(n_samples_block * 10)))  # Re-initialize the generator
                ratio = {str(cls): 1/len(classes) for cls in classes}
                print(f"⚖️ Block {i+1}: balance enabled with ratios {ratio}")
            else:
                ratio = class_ratios[i]

            counts = {cls: int(ratio[str(cls)] * n_samples_block) for cls in ratio} if ratio else None
            current_counts = {cls: 0 for cls in ratio} if ratio else {}

            while len(block_data) < n_samples_block:
                try:
                    x, y = next(gen_iter)
                except StopIteration:
                    break

                if ratio:
                    y_str = str(y)
                    if y_str not in counts:
                        continue
                    if current_counts[y_str] >= counts[y_str]:
                        continue
                    current_counts[y_str] += 1

                row = list(x.values()) + [y, i+1]
                if cols is None:
                    cols = list(x.keys()) + [target_col, "chunk"]
                block_data.append(row)

            data.extend(block_data)

            block_df = pd.DataFrame(block_data, columns=cols)
            block_dist = block_df[target_col].value_counts(normalize=True).to_dict()

            block_reports.append({
                "block": i+1,
                "samples": len(block_data),
                "distribution": block_dist
            })

            print(f"✅ Block {i+1}: {len(block_data)} samples, target distribution {block_dist}")

        df = pd.DataFrame(data[:total_samples], columns=cols)  # Create final DataFrame with total_samples
        df.to_csv(full_path, index=False)

        print(f"✅ Generated {total_samples} samples in {n_blocks} blocks at: {full_path}")

        # Drift detection between blocks (comparing class ratios)
        drift_info = self._detect_drift_between_blocks(block_reports)

        # Automatic reporting of the dataset
        self._report_dataset(
            df,
            target_col,
            extra_info={
                "Number of blocks": n_blocks,
                "Samples per block": instances_per_block,
                "Block distributions": {r["block"]: r["distribution"] for r in block_reports},
                "Drift between blocks": drift_info
            }
        )

        return full_path

    def _detect_drift_between_blocks(self, block_reports):
        """Detects drift between consecutive blocks by comparing class distributions."""
        drift_info = []
        for idx in range(1, len(block_reports)):
            prev = block_reports[idx-1]["distribution"]
            curr = block_reports[idx]["distribution"]
            drift = {cls: curr.get(cls, 0) - prev.get(cls, 0) for cls in set(prev) | set(curr)}
            drift_info.append({"from_block": idx, "to_block": idx+1, "drift": drift})
        return drift_info
