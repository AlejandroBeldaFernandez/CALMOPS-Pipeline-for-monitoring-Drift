import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Literal
from Real.RealGenerator import RealGenerator
from Real.DistributionChanger import DistributionChanger
from Real.DriftInjector import DriftInjector
from .RealReporter import RealReporter
import logging

# Setup logger for tracking events and errors
logger = logging.getLogger('RealBlockGeneratorLogger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class RealBlockGenerator(RealReporter):
    def __init__(self, df: pd.DataFrame, target_col: str = "target"):
        """
        Initializes the RealBlockGenerator with a DataFrame and target column.
        """
        self.df = df.copy()
        self.target_col = target_col

    def _ensure_list(self, value, n_blocks):
        """
        Ensures that a parameter is a list of length n_blocks.
        If a single value is passed, it will be expanded to a list of length n_blocks.
        """
        if isinstance(value, list):
            if len(value) == 1:
                return value * n_blocks
            elif len(value) != n_blocks:
                raise ValueError(f"List must have length 1 or {n_blocks}, got {len(value)}")
            return value
        else:
            return [value] * n_blocks

    def generate_blocks(
            self,
            output_path: str,
            filename: str,
            n_blocks: int,
            instances_per_block: List[int],
            methods: Optional[List[Literal["resample", "smote", "gmm", "ctgan", "copula"]]] = None,
            distributions: Optional[List[Dict[int, float]]] = None,
            drift_events: Optional[List[List[Dict[str, Any]]]] = None,
            random_state: Optional[int] = None,
            add_chunk: bool = True
        ) -> str:
        """
        Generates synthetic data divided into blocks with optional drift and distribution changes.
        """
        # Ensure the lists have the correct length
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)

        methods = self._ensure_list(methods or "resample", n_blocks)
        instances_per_block = self._ensure_list(instances_per_block, n_blocks)
        distributions = self._ensure_list(distributions or [None], n_blocks)
        drift_events = self._ensure_list(drift_events or [None], n_blocks)

        all_blocks = []
        temp_files = []
        block_reports = []  # To store information about each block

        for i in range(n_blocks):
            temp_filename = f"temp_block_{i+1}.csv"
            temp_path = os.path.join(output_path, temp_filename)
            temp_files.append(temp_path)

            # Generate block with RealGenerator
            gen = RealGenerator(dataset_path="titanic.csv", target_col=self.target_col)
            block_path = gen.generate(
                output_path=output_path,
                filename=temp_filename,
                n_samples=instances_per_block[i],
                method=methods[i],
                random_state=random_state
            )
            block_df = pd.read_csv(block_path)

            # Adjust the number of instances to match the required count
            if len(block_df) > instances_per_block[i]:
                block_df = block_df.sample(
                    n=instances_per_block[i], random_state=random_state
                ).reset_index(drop=True)
            elif len(block_df) < instances_per_block[i]:
                extra = block_df.sample(
                    n=instances_per_block[i] - len(block_df),
                    replace=True,
                    random_state=random_state
                )
                block_df = pd.concat([block_df, extra]).reset_index(drop=True)

            # Apply target distribution changes if necessary
            if distributions[i]:
                changer = DistributionChanger(block_df, target_col=self.target_col)
                block_df = changer.change_target_distribution(
                    distribution_before=distributions[i],
                    distribution_after=distributions[i],
                    start_idx=0
                )

            # Apply drift events if necessary
            if drift_events[i]:
                injector = DriftInjector(block_df, target_col=self.target_col)
                for ev in drift_events[i]:
                    injector.add_event(
                        start_idx=ev["start_idx"],
                        affected_col=ev["col"],
                        change=ev["change"]
                    )
                block_df = injector.apply()

            # Add chunk column if required
            if add_chunk:
                block_df["chunk"] = i + 1

            # Calculate the distribution of the target in the block
            block_dist = block_df[self.target_col].value_counts(normalize=True).to_dict()

            block_reports.append({
                "block": i + 1,
                "samples": len(block_df),
                "method": methods[i],   # Store the method used
                "distribution": block_dist
            })

            all_blocks.append(block_df)

            logger.info(f"✅ Block {i+1}: {len(block_df)} samples generated using method {methods[i]}, "
                        f"distribution {block_dist}")

        # Concatenate all blocks
        result = pd.concat(all_blocks, ignore_index=True)
        result.to_csv(full_path, index=False)

        # Delete temporary files
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

        logger.info(f"✅ Generated {len(result)} samples in {n_blocks} blocks at: {full_path}")

        # Generate report using RealReporter
        self._report_real_dataset(
            real_df=self.df,
            synthetic_df=result,
            target_col=self.target_col,
            method="RealBlockGenerator",
            extra_info={
                "Number of blocks": n_blocks,
                "Block reports": block_reports  # Pass all block info
            }
        )
        return full_path
