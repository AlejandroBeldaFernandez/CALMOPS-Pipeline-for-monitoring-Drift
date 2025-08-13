import pandas as pd
import numpy as np
import logging
from typing import Dict, List

# Setup logger to track events and errors
logger = logging.getLogger('DistributionChangerLogger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class DistributionChanger:
    def __init__(self, df: pd.DataFrame, target_col: str):
        """
        Initializes the DistributionChanger with a DataFrame and target column.
        """
        self.df = df.copy()
        self.target_col = target_col

    def _generate_samples(self, distribution: Dict[int, float], n_samples: int) -> List[int]:
        """
        Generates samples based on a given distribution of class values.
        """
        values, probs = list(distribution.keys()), list(distribution.values())
        counts = [int(p * n_samples) for p in probs]

        # Adjust counts to ensure they sum up to n_samples
        while sum(counts) < n_samples:
            counts[np.argmin(counts)] += 1
        samples = []
        for val, count in zip(values, counts):
            samples.extend([val] * count)
        
        np.random.shuffle(samples)
        return samples

    def change_target_distribution(
        self,
        distribution_before: Dict[int, float],
        distribution_after: Dict[int, float],
        start_idx: int
    ) -> pd.DataFrame:
        """
        Apply exact class distributions before and after start_idx.
        """
        n_total = len(self.df)

        if not np.isclose(sum(distribution_before.values()), 1.0):
            raise ValueError("Probabilities before drift must sum to 1.")
        if not np.isclose(sum(distribution_after.values()), 1.0):
            raise ValueError("Probabilities after drift must sum to 1.")

        # BEFORE DRIFT
        n_before = start_idx
        samples_b = self._generate_samples(distribution_before, n_before)
        self.df.loc[:start_idx-1, self.target_col] = samples_b

        # AFTER DRIFT
        n_after = n_total - start_idx
        samples_a = self._generate_samples(distribution_after, n_after)
        self.df.loc[start_idx:, self.target_col] = samples_a

        # 📊 Report
        logger.info("\n📊 === DistributionChanger Report ===")
        logger.info(f"Applied change at index {start_idx}")
        logger.info(f"Target distribution before drift: {distribution_before}")
        logger.info(f"Target distribution after drift: {distribution_after}")
        logger.info(f"New class proportions: {self.df[self.target_col].value_counts(normalize=True).to_dict()}")
        logger.info("=== END OF REPORT ===\n")

        return self.df
