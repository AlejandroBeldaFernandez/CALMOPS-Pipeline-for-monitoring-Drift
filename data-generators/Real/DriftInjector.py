import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Setup logger for tracking events and errors
logger = logging.getLogger('DriftInjectorLogger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class DriftInjector:
    def __init__(self, df: pd.DataFrame, target_col: str = "target"):
        """
        Initializes the DriftInjector with a DataFrame and target column.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.events = []

    def add_event(self, start_idx: int, affected_col: str, change: Dict[str, Any]):
        """
        Adds a drift event to be applied at a specified index on a given column.
        """
        self.events.append({"start_idx": start_idx, "col": affected_col, "change": change})

    def apply(self) -> pd.DataFrame:
        """
        Applies all drift events to the DataFrame and returns the modified DataFrame.
        """
        result = self.df.copy()

        logger.info("\n📊 === DriftInjector Report ===")
        for event in self.events:
            mask = result.index >= event["start_idx"]
            col = event["col"]
            change = event["change"]

            logger.info(f"🔀 Applying drift on column '{col}' starting at index {event['start_idx']}")
            logger.info(f"   Change details: {change}")

            # Apply drift based on the type of change specified
            if np.issubdtype(result[col].dtype, np.number) and "operation" in change:
                self._apply_numeric_drift(result, mask, col, change)

            elif "distribution" in change:  # Categorical drift
                self._apply_categorical_drift(result, mask, col, change)

            elif "conditional_on" in change:  # Conditional drift
                self._apply_conditional_drift(result, mask, col, change)

        logger.info("✅ Drift events applied successfully.")
        logger.info("=== END OF REPORT ===\n")

        return result

    def _apply_numeric_drift(self, result: pd.DataFrame, mask: pd.Series, col: str, change: Dict[str, Any]):
        """
        Applies numeric drift to a column based on the specified operation and value.
        """
        op = change["operation"]
        if op == "shift":
            result.loc[mask, col] += change["value"]
        elif op == "scale":
            result.loc[mask, col] *= change["value"]
        elif op == "noise":
            sigma = change.get("value", 1)
            result.loc[mask, col] += np.random.normal(0, sigma, mask.sum())
        elif op == "clip":
            min_val, max_val = change["value"]
            result.loc[mask, col] = np.clip(result.loc[mask, col], min_val, max_val)
        elif op == "replace":
            mu, sigma = change["value"]
            result.loc[mask, col] = np.random.normal(mu, sigma, mask.sum())
        else:
            logger.warning(f"Unknown numeric operation: {op}")

    def _apply_categorical_drift(self, result: pd.DataFrame, mask: pd.Series, col: str, change: Dict[str, Any]):
        """
        Applies categorical drift by adjusting the distribution of values in a column.
        """
        dist = change["distribution"]
        values, probs = list(dist.keys()), list(dist.values())
        n = mask.sum()
        counts = [int(p * n) for p in probs]
        while sum(counts) < n:
            counts[np.argmin(counts)] += 1
        samples = []
        for val, count in zip(values, counts):
            samples.extend([val] * count)
        np.random.shuffle(samples)
        result.loc[mask, col] = samples

    def _apply_conditional_drift(self, result: pd.DataFrame, mask: pd.Series, col: str, change: Dict[str, Any]):
        """
        Applies conditional drift, where the drift is applied based on a condition in another column.
        """
        cond_col = change["conditional_on"]
        probs = change["probs"]
        for cat_val, prob in probs.items():
            submask = mask & (result[cond_col] == cat_val)
            n = submask.sum()
            n_ones = int(prob * n)
            values = [1] * n_ones + [0] * (n - n_ones)
            np.random.shuffle(values)
            result.loc[submask, self.target_col] = values
