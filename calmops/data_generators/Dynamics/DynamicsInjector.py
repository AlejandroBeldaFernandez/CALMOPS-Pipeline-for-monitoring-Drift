import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Callable


class DynamicsInjector:
    """
    A standalone module to modify scenarios by evolving features and constructing target variables.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def evolve_features(
        self,
        df: pd.DataFrame,
        evolution_config: Dict[str, Dict[str, Union[str, float, int]]],
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evolves features in the DataFrame based on the provided configuration.

        Args:
            df (pd.DataFrame): Input DataFrame.
            evolution_config (dict): Dictionary mapping column names to evolution specs.
                Example: {'Age': {'type': 'linear', 'slope': 0.1},
                          'Sales': {'type': 'cycle', 'period': 12, 'amplitude': 10}}
            time_col (str, optional): Column to use as the time variable t.
                                      If None, uses the DataFrame index (must be numeric or convertible).

        Returns:
            pd.DataFrame: DataFrame with evolved features.
        """
        df_evolved = df.copy()

        if time_col:
            if time_col not in df.columns:
                raise ValueError(f"Time column '{time_col}' not found in DataFrame.")
            t = df[time_col].values
            # Ensure t is numeric
            if not np.issubdtype(t.dtype, np.number):
                # Try to convert to numeric if it's datetime
                if np.issubdtype(t.dtype, np.datetime64):
                    t = t.astype(np.int64) // 10**9  # Seconds
                else:
                    # Fallback to range
                    t = np.arange(len(df))
        else:
            t = np.arange(len(df))

        for col, config in evolution_config.items():
            if col not in df_evolved.columns:
                continue  # Or raise warning

            drift_type = config.get("type")

            delta = np.zeros_like(t, dtype=float)

            if drift_type == "linear":
                slope = config.get("slope", 0.0)
                intercept = config.get("intercept", 0.0)
                delta = slope * t + intercept

            elif drift_type == "cycle" or drift_type == "sinusoidal":
                period = config.get("period", 100)
                amplitude = config.get("amplitude", 1.0)
                phase = config.get("phase", 0.0)
                delta = amplitude * np.sin(2 * np.pi * t / period + phase)

            elif drift_type == "sigmoid":
                center = config.get("center", len(t) / 2)
                width = config.get("width", len(t) / 10)
                amplitude = config.get("amplitude", 1.0)
                # Sigmoid function: 1 / (1 + exp(-(t-center)/width))
                # Scaled by amplitude
                # Avoid overflow
                z = (t - center) / (width + 1e-9)
                sigmoid = 1.0 / (1.0 + np.exp(-z))
                delta = amplitude * sigmoid

            # Apply delta
            # Assuming additive drift for now. Could add 'mode': 'multiplicative' later.
            df_evolved[col] = df_evolved[col] + delta

        return df_evolved

    def construct_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        formula: Union[str, Callable],
        noise_std: float = 0.0,
        task_type: str = "regression",
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Constructs or overwrites a target variable based on a formula.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column to create/overwrite.
            formula (str or callable): Formula to calculate the raw target score.
                - If str: Used in df.eval(). Example: "0.5 * Age + 2.0 * Income"
                - If callable: Accepts df and returns a Series/array.
            noise_std (float): Standard deviation of Gaussian noise to add.
            task_type (str): 'regression' or 'classification'.
            threshold (float, optional): Threshold for binary classification.
                                         If None and task_type='classification', defaults to 0 (assuming raw score is logit-like)
                                         or requires sigmoid probability sampling.
                                         Here we implement simple thresholding: Y = 1 if Score > threshold else 0.

        Returns:
            pd.DataFrame: DataFrame with the new target column.
        """
        df_target = df.copy()

        # 1. Calculate Raw Score
        if isinstance(formula, str):
            try:
                # Use pandas eval for string formulas
                # We perform eval in the context of the dataframe
                raw_score = df_target.eval(formula)
            except Exception as e:
                raise ValueError(f"Error evaluating formula '{formula}': {e}")
        elif callable(formula):
            raw_score = formula(df_target)
        else:
            raise ValueError("Formula must be a string or a callable.")

        # Ensure raw_score is numeric
        raw_score = np.array(raw_score, dtype=float)

        # 2. Add Noise
        if noise_std > 0:
            noise = self.rng.normal(0, noise_std, size=len(df_target))
            raw_score += noise

        # 3. Finalize Target based on Task Type
        if task_type == "regression":
            df_target[target_col] = raw_score

        elif task_type == "classification":
            if threshold is None:
                # Default threshold 0 (e.g. if formula outputs log-odds or centered score)
                # Or we could use mean/median if we wanted balanced classes dynamically?
                # Let's stick to 0.0 as default for explicit formulas.
                threshold = 0.0

            df_target[target_col] = (raw_score > threshold).astype(int)

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return df_target
