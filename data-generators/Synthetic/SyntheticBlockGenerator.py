import os
import pandas as pd
from collections import defaultdict
from .SyntheticReporter import SyntheticReporter

class SyntheticBlockGenerator:
    def generate_blocks_simple(self,
                              output_path: str,
                              filename: str,
                              n_blocks: int,
                              total_samples: int,
                              methods,
                              method_params=None,
                              instances_per_block=None,
                              class_ratios=None,
                              target_col="target",
                              balance: bool = False,
                              random_state: int = None):
        """
        Simplified interface for generating block datasets (similar to SyntheticGenerator)
        
        Args:
            output_path: Directory to save the generated data
            filename: Name of the output file
            n_blocks: Number of blocks to generate
            total_samples: Total number of samples across all blocks
            methods: List of generator methods for each block (['sea', 'agrawal', etc.])
            method_params: List of parameter dicts for each method
            instances_per_block: List of samples per block (if None, distributes evenly)
            class_ratios: List of class ratio dicts for each block
            target_col: Name of target column
            balance: Whether to balance classes
            random_state: Random seed for reproducibility
        """
        from .GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Method string to enum mapping
        method_mapping = {
            "sea": GeneratorType.SEA,
            "agrawal": GeneratorType.AGRAWAL,
            "hyperplane": GeneratorType.HYPERPLANE,
            "sine": GeneratorType.SINE,
            "stagger": GeneratorType.STAGGER,
            "random_tree": GeneratorType.RANDOM_TREE,
            "mixed": GeneratorType.MIXED,
            "friedman": GeneratorType.FRIEDMAN,
            "random_rbf": GeneratorType.RANDOM_RBF
        }
        
        # Ensure methods is a list
        methods = self._ensure_list(methods, n_blocks)
        
        # Validate all methods
        for method in methods:
            if method not in method_mapping:
                available_methods = list(method_mapping.keys())
                raise ValueError(f"Invalid method '{method}'. Choose one of {available_methods}")
        
        # Set defaults
        if method_params is None:
            method_params = [{}] * n_blocks
        else:
            method_params = self._ensure_list(method_params, n_blocks)
        
        if instances_per_block is None:
            # Distribute evenly
            base_samples = total_samples // n_blocks
            instances_per_block = [base_samples] * n_blocks
            # Add remaining samples to last block
            instances_per_block[-1] += total_samples % n_blocks
        else:
            instances_per_block = self._ensure_list(instances_per_block, n_blocks)
        
        # Create generator instances
        factory = GeneratorFactory()
        generators = []
        
        for i, (method, params) in enumerate(zip(methods, method_params)):
            # Add random state if provided
            if random_state is not None:
                params = params.copy()
                params['random_state'] = random_state + i  # Different seed per block
            
            config = GeneratorConfig(**params)
            generator_instance = factory.create_generator(method_mapping[method], config)
            generators.append(generator_instance)
        
        # Call original generate_blocks method
        return self.generate_blocks(
            output_path=output_path,
            filename=filename,
            n_blocks=n_blocks,
            total_samples=total_samples,
            instances_per_block=instances_per_block,
            generators=generators,
            class_ratios=class_ratios,
            target_col=target_col,
            balance=balance
        )

    def _ensure_list(self, value, n_blocks):
        """
        Ensures the parameter is a list of length n_blocks.
        - If it's a single value, repeat it across n_blocks.
        - If it's a list of length 1, repeat it across n_blocks.
        - If it's a list of length n_blocks, return as is.
        - Otherwise, raise ValueError.
        """
        if isinstance(value, list):
            if len(value) == 1:
                return value * n_blocks
            elif len(value) == n_blocks:
                return value
            else:
                raise ValueError(
                    f"List length {len(value)} does not match n_blocks={n_blocks}. "
                    "Use block_assignments to specify custom mapping."
                )
        else:
            return [value] * n_blocks

    def _apply_block_assignments(self, base_values, n_blocks, block_assignments, key):
        """
        Expands values based on block_assignments mapping for a given key.
        Example:
            base_values = [gen1, gen2]
            block_assignments["generators"] = {0: [1], 1: [2, 3]}
            => [gen1, gen2, gen2]
        """
        if not block_assignments or key not in block_assignments:
            return base_values

        mapping = block_assignments[key]
        expanded = [None] * n_blocks
        for base_idx, block_idxs in mapping.items():
            for b in block_idxs:
                if b < 1 or b > n_blocks:
                    raise ValueError(
                        f"Invalid block index {b} in block_assignments[{key}] "
                        f"(must be between 1 and {n_blocks})."
                    )
                if expanded[b - 1] is not None:
                    raise ValueError(
                        f"Block {b} for {key} is assigned more than once."
                    )
                expanded[b - 1] = base_values[base_idx]

        if any(v is None for v in expanded):
            raise ValueError(
                f"block_assignments for {key} did not cover all {n_blocks} blocks."
            )
        return expanded

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
        balance: bool = False,
        block_assignments: dict = None
    ) -> str:
        """
        Generates synthetic data divided into blocks using specified generator instances.
        Supports custom block assignments for flexible mapping.
        """

        # Step 1: Normalize inputs with default expansion
        instances_per_block = self._ensure_list(instances_per_block, n_blocks)
        generators = self._ensure_list(generators, n_blocks)
        class_ratios = (
            self._ensure_list(class_ratios, n_blocks) if class_ratios else [None] * n_blocks
        )

        # Step 2: Apply block_assignments if provided
        if block_assignments:
            generators = self._apply_block_assignments(generators, n_blocks, block_assignments, "generators")
            instances_per_block = self._apply_block_assignments(instances_per_block, n_blocks, block_assignments, "instances_per_block")
            class_ratios = self._apply_block_assignments(class_ratios, n_blocks, block_assignments, "class_ratios")

        # Validate sample counts
        if sum(instances_per_block) != total_samples:
            raise ValueError(
                f"Total samples ({total_samples}) must equal the sum of instances per block ({sum(instances_per_block)})"
            )

        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)

        all_data = []
        block_reports = []
        all_columns = set()

        # Detect column structure
        for i in range(n_blocks):
            gen = generators[i]
            gen_iter = iter(gen.take(5))
            try:
                x, y = next(gen_iter)
                all_columns.update(x.keys())
            except StopIteration:
                continue

        feature_cols = sorted([str(col) for col in all_columns])
        all_cols = feature_cols + [target_col, "block"]

        for i in range(n_blocks):
            gen = generators[i]
            n_samples_block = instances_per_block[i]

            block_data = []
            gen_iter = iter(gen.take(n_samples_block * 10))

            if balance:
                preview = [next(gen_iter) for _ in range(50)]
                classes = set([y for _, y in preview])
                gen_iter = iter(preview + list(gen.take(n_samples_block * 10)))
                ratio = {str(cls): 1 / len(classes) for cls in classes}
                print(f"Block {i+1}: balance enabled with ratios {ratio}")
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

                row = []
                for col in feature_cols:
                    value = None
                    for orig_key in x.keys():
                        if str(orig_key) == col:
                            value = x[orig_key]
                            break
                    if value is None:
                        value = 0
                    row.append(value)
                row.extend([y, i + 1])

                block_data.append(row)

            all_data.extend(block_data)

            block_df = pd.DataFrame(block_data, columns=all_cols)
            block_dist = block_df[target_col].value_counts(normalize=True).to_dict()

            block_reports.append({
                "block": i + 1,
                "samples": len(block_data),
                "distribution": block_dist
            })

            print(f" Block {i+1}: {len(block_data)} samples, target distribution {block_dist}")

        df = pd.DataFrame(all_data[:total_samples], columns=all_cols)
        df.to_csv(full_path, index=False)

        print(f" Generated {total_samples} samples in {n_blocks} blocks at: {full_path}")

        drift_info = self._detect_drift_between_blocks(block_reports)

        reporter = SyntheticReporter()
        reporter.generate_report(
            df=df,
            target_col=target_col,
            drift_type="none",
            position_of_drift=None,
            is_block_dataset=True,
            output_path=output_path
        )

        return full_path

    def _detect_drift_between_blocks(self, block_reports):
        """Detects drift between consecutive blocks by comparing class distributions."""
        drift_info = []
        for idx in range(1, len(block_reports)):
            prev = block_reports[idx - 1]["distribution"]
            curr = block_reports[idx]["distribution"]
            drift = {cls: curr.get(cls, 0) - prev.get(cls, 0) for cls in set(prev) | set(curr)}
            drift_info.append({"from_block": idx, "to_block": idx + 1, "drift": drift})
        return drift_info

