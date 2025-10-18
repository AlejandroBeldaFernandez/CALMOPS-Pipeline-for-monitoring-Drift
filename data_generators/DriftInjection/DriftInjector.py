#!/usr/bin/env python3
"""
Drift Injector for Real Data - Injects various types of drift into real datasets.

This module provides the DriftInjector class, which is designed to introduce a wide range of controlled
drifts into a pandas DataFrame. It supports various drift types, including feature drift, label drift,
and more complex patterns like gradual, abrupt, and recurrent drifts.

Key Features:
- **Multiple Drift Types**: Inject gaussian_noise, shift, scale, and other transformations.
- **Flexible Targeting**: Apply drift to the entire dataset, specific blocks, or row indices.
- **Advanced Drift Profiles**: Simulate gradual, abrupt, incremental, and recurrent drifts using window functions (sigmoid, linear, cosine).
- **Label and Concept Drift**: Includes methods for label flipping (label_drift), changing target distribution (label_shift), and introducing new categories (new_category_drift).
- **Covariate and Virtual Drift**: Modify correlation structures (correlation_matrix_drift) and introduce missing values (missing_values_drift).
- **Integrated Reporting**: Automatically generates detailed reports and visualizations comparing the original and drifted datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Sequence, Tuple, Any
import warnings
import os

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from data_generators.Real.RealReporter import RealReporter

class DriftInjector:
    """
    A class to inject various types of drift into a pandas DataFrame.
    """
    # -------------------------
    # Init & utils
    # -------------------------
    def __init__(self, original_df: pd.DataFrame, output_dir: str, generator_name: str, target_column: Optional[str] = None, block_column: Optional[str] = None, random_state: Optional[int] = None):
        """
        Initializes the DriftInjector.

        Args:
            original_df (pd.DataFrame): The original, clean DataFrame.
            output_dir (str): Directory to save reports and drifted datasets.
            generator_name (str): A name for the generator, used in output file names.
            target_column (Optional[str]): The name of the target variable column.
            block_column (Optional[str]): The name of the column defining data blocks or chunks.
            random_state (Optional[int]): Seed for the random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)

        self.original_df = original_df
        self.output_dir = output_dir
        self.generator_name = generator_name
        self.target_column = target_column
        self.block_column = block_column
        self.reporter = RealReporter()
        os.makedirs(self.output_dir, exist_ok=True) # Ensure output_dir exists

    def _frac(self, x: float) -> float:
        """Clips a float to the [0.0, 1.0] range."""
        return float(np.clip(x, 0.0, 1.0))

    def _generate_reports(self, original_df, drifted_df, drift_config):
        """Helper to generate the standard report."""
        # Generate the primary report in the main output directory
        self.reporter.update_report_after_drift(
            original_df=original_df,
            drifted_df=drifted_df,
            output_dir=self.output_dir,
            drift_config=drift_config
        )

    def _get_target_rows(
        self,
        df: pd.DataFrame,
        start_index: Optional[int],
        block_index: Optional[int],
        block_column: Optional[str]
    ) -> pd.Index:
        """
        Selects rows where to apply the drift.

        - By block (if block_index and block_column are provided).
        - By position from start_index otherwise.
        """
        if block_index is not None:
            if block_column is None:
                block_column = self.block_column
            if block_column not in df.columns:
                raise ValueError(f"Block column '{block_column}' not found in dataframe")
            return df.index[df[block_column] == block_index]
        start_index = 0 if start_index is None else max(0, start_index)
        return df.iloc[start_index:].index

    # -------------------------
    # Advanced block selection
    # -------------------------
    def _select_rows_by_blocks(
        self,
        df: pd.DataFrame,
        block_column: str,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
    ) -> pd.Index:
        """
        Selects rows based on block identifiers.

        - `blocks`: A list/tuple of specific block IDs to select.
        - `block_start` and `n_blocks`: Selects a continuous range of blocks.
        """
        if block_column not in df.columns:
            raise ValueError(f"Block column '{block_column}' not found in dataframe")

        if blocks is not None:
            mask = df[block_column].isin(blocks)
            return df.index[mask]

        if block_start is not None and n_blocks is not None:
            uniq = df[block_column].dropna().unique().tolist()
            if block_start not in uniq:
                warnings.warn(f"block_start '{block_start}' does not appear in '{block_column}'")
                return df.iloc[0:0].index
            i0 = uniq.index(block_start)
            take = uniq[i0:i0 + max(1, int(n_blocks))]
            mask = df[block_column].isin(take)
            return df.index[mask]

        # if nothing is specified, select nothing
        return df.iloc[0:0].index

    # -------------------------
    # Windows (profiles + speed)
    # -------------------------
    def _sigmoid_weights(
        self,
        n: int,
        center: float,
        width: int
    ) -> np.ndarray:
        """
        Creates weights w in [0,1] over n positions with a sigmoid transition.
        
        Args:
            n (int): Number of positions (rows).
            center (float): The center of the transition (in coordinates 0..n-1).
            width (int): Controls how many rows it takes to go from ~10% to ~90%.
        
        Returns:
            np.ndarray: An array of weights.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)
        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        # Map width -> sigmoid scale. Approximately 4*scale ~ width (10%->90%)
        scale = width / 4.0
        z = (i - float(center)) / max(1e-9, scale)
        w = 1.0 / (1.0 + np.exp(-z))
        return w

    def _window_weights(
        self,
        n: int,
        center: float,
        width: int,
        profile: str = "sigmoid",
        k: float = 1.0,
        direction: str = "up",
    ) -> np.ndarray:
        """
        Returns weights w in [0,1] of size n with a transition centered at `center` and of `width`.
        
        Args:
            n (int): Number of positions.
            center (float): Center of the transition.
            width (int): Width of the transition.
            profile (str): The shape of the transition window ("sigmoid", "linear", "cosine").
            k (float): Controls the "speed" (slope) of the transition.
            direction (str): "up" (0->1) or "down" (1->0).
            
        Returns:
            np.ndarray: An array of weights.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)

        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        center = float(center)

        if profile == "sigmoid":
            base_scale = width / 4.0
            scale = max(1e-9, base_scale / max(1e-9, float(k)))  # high k -> faster
            z = (i - center) / scale
            w = 1.0 / (1.0 + np.exp(-z))
        elif profile == "linear":
            left = center - width / 2.0
            right = center + width / 2.0
            w = (i - left) / max(1e-9, (right - left))
            w = np.clip(w, 0.0, 1.0)
            if k != 1.0:
                w = np.clip((w - 0.5) * k + 0.5, 0.0, 1.0)
        elif profile == "cosine":
            left = center - width / 2.0
            right = center + width / 2.0
            t = (i - left) / max(1e-9, (right - left))
            t = np.clip(t, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * t)
            if k != 1.0:
                w = np.clip((w - 0.5) * k + 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown profile: {profile}")

        if direction == "down":
            w = 1.0 - w

        return w

    # -------------------------
    # Common engine for features
    # -------------------------
    def _apply_numeric_op_with_weights(
        self,
        values: np.ndarray,
        drift_type: str,
        drift_magnitude: float,
        w: np.ndarray,
        rng: np.random.Generator,
        column_drift_value: Optional[float]
    ) -> np.ndarray:
        """
        Applies a numeric drift operation, scaled by weights `w` row by row.
        """
        x = values.astype(float, copy=True)
        n = len(x)
        if n == 0:
            return x

        mean = float(np.mean(x)) if n > 0 else 0.0
        std = float(np.std(x)) if n > 0 else 0.0
        w = np.asarray(w, dtype=float)
        
        # Fix for broadcasting error when w is shorter than x
        if len(w) < n:
            w = np.pad(w, (0, n - len(w)), 'edge')
            
        w = np.clip(w, 0.0, 1.0)

        if drift_type == "gaussian_noise":
            if std == 0:
                return x
            noise = rng.normal(0.0, drift_magnitude * std, size=n)
            x = x + noise * w

        elif drift_type == "shift":
            shift_amt = drift_magnitude * mean
            x = x + shift_amt * w

        elif drift_type == "scale":
            # row-wise factor: 1 + w*m
            factor = 1.0 + (w * drift_magnitude)
            x = mean + (x - mean) * factor

        elif drift_type == "add_value":
            if column_drift_value is None:
                raise ValueError("add_value requires drift_value/drift_values[col]")
            x = x + (w * column_drift_value)

        elif drift_type == "subtract_value":
            if column_drift_value is None:
                raise ValueError("subtract_value requires drift_value/drift_values[col]")
            x = x - (w * column_drift_value)

        elif drift_type == "multiply_value":
            if column_drift_value is None:
                raise ValueError("multiply_value requires drift_value/drift_values[col]")
            # mix towards the indicated factor: x * (1 + w*(f-1))
            factor = 1.0 + w * (float(column_drift_value) - 1.0)
            x = x * factor

        elif drift_type == "divide_value":
            if column_drift_value is None:
                raise ValueError("divide_value requires drift_value/drift_values[col]")
            if float(column_drift_value) == 0.0:
                raise ValueError("drift_value cannot be zero for 'divide_value'")
            # dividing is equivalent to multiplying by (1/val); we mix towards that factor
            target = 1.0 / float(column_drift_value)
            factor = 1.0 + w * (target - 1.0)
            x = x * factor

        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")

        # Preserve original dtype to avoid FutureWarnings
        original_dtype = values.dtype
        if pd.api.types.is_integer_dtype(original_dtype):
            x = np.round(x).astype(original_dtype)

        return x

    def _apply_categorical_with_weights(
        self,
        original_vals: pd.Series,
        w: np.ndarray,
        drift_magnitude: float,
        rng: np.random.Generator
    ) -> pd.Series:
        """
        Changes categorical values with a probability per row p = clamp(w * drift_magnitude).
        Replaces the value with a random category different from the current one.
        """
        s = original_vals.copy()
        uniques = s.dropna().unique()
        if len(uniques) <= 1:
            return s

        w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
        p = np.clip(w * self._frac(drift_magnitude), 0.0, 1.0)

        # flip a coin for each row
        mask = rng.random(len(s)) < p
        idxs = s.index[mask]
        if len(idxs) == 0:
            return s

        # for each row to change, choose a different category
        current = s.loc[idxs].to_numpy()
        new_vals = []
        for cur in current:
            choices = [u for u in uniques if u != cur]
            if choices:
                new_vals.append(rng.choice(choices))
            else:
                new_vals.append(cur)
        s.loc[idxs] = new_vals
        return s

    def _validate_feature_op(self, drift_type: str, drift_magnitude: float):
        """Validates the feature drift operation and its magnitude."""
        if drift_type in {"gaussian_noise", "shift", "scale"} and drift_magnitude < 0:
            raise ValueError("drift_magnitude must be >= 0 for gaussian_noise/shift/scale")
        valid = {
            "gaussian_noise", "shift", "scale",
            "add_value", "subtract_value", "multiply_value", "divide_value"
        }
        if drift_type not in valid:
            raise ValueError(f"Unknown drift_type: {drift_type}")

    # -------------------------
    # Feature drift (full block)
    # -------------------------
    def inject_feature_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Applies drift at once from a start_index or in a specific block.
        This is equivalent to a window with weight=1 for all selected rows.
        """
        if block_column is None: block_column = self.block_column
        self._validate_feature_op(drift_type, drift_magnitude)
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)

        w = np.ones(len(rows), dtype=float)

        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in dataframe")
                continue

            column_drift_value = None
            if drift_type in {"add_value", "subtract_value", "multiply_value", "divide_value"}:
                if drift_values and col in drift_values:
                    column_drift_value = drift_values[col]
                else:
                    column_drift_value = drift_value
                if column_drift_value is None:
                    raise ValueError(f"For '{drift_type}', provide drift_value or drift_values['{col}']")
                if drift_type == "divide_value" and float(column_drift_value) == 0.0:
                    raise ValueError("drift_value cannot be zero for 'divide_value'")

            if pd.api.types.is_numeric_dtype(df[col]):
                x = df_drift.loc[rows, col].to_numpy(copy=True)
                x2 = self._apply_numeric_op_with_weights(
                    x, drift_type, drift_magnitude, w, self.rng, column_drift_value
                )
                df_drift.loc[rows, col] = x2
            else:
                s = df_drift.loc[rows, col]
                s2 = self._apply_categorical_with_weights(s, w, drift_magnitude, self.rng)
                df_drift.loc[rows, col] = s2

        drift_config = {
            "drift_method": "inject_feature_drift",
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            "drift_value": drift_value,
            "drift_values": drift_values,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "generator_name": f"{self.generator_name}_feature_drift"
        }
        df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_drift, drift_config)
        return df_drift

    # -------------------------
    # Feature drift “windowed”: gradual, abrupt, incremental, recurrent
    # -------------------------
    def inject_feature_drift_gradual(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        inconsistency: float = 0.0 # New parameter for inconsistent drift
    ) -> pd.DataFrame:
        """
        Injects gradual drift on selected rows using a transition window.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_cols (List[str]): Columns to apply drift to.
            drift_type (str): Type of numeric drift (e.g., 'shift', 'scale').
            drift_magnitude (float): Magnitude of the drift.
            center (Optional[int]): Center of the transition within the selected section (0..len(rows)-1).
            width (Optional[int]): Transition “width” (approx 10% to 90% of the effect).
            profile (str): Shape of the transition ramp ('sigmoid', 'linear', 'cosine').
            speed_k (float): Multiplier for the transition speed.
            direction (str): 'up' for 0->1 transition, 'down' for 1->0.
            inconsistency (float): Factor (0-1) to add random noise and variability to the drift speed.
        
        Returns:
            pd.DataFrame: The DataFrame with gradual drift injected.
        """
        if block_column is None: block_column = self.block_column
        self._validate_feature_op(drift_type, drift_magnitude)
        df_drift = df.copy()

        if blocks is not None or (block_start is not None and n_blocks is not None):
            if not block_column:
                raise ValueError("block_column required for block selection")
            rows = self._select_rows_by_blocks(df, block_column, blocks, block_start, n_blocks)
        else:
            rows = self._get_target_rows(df, start_index, block_index, block_column)

        n = len(rows)
        if n == 0:
            return df_drift

        c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))
        w = self._window_weights(n, center=c, width=w_width, profile=profile, k=float(speed_k), direction=direction)

        if inconsistency > 0 and n > 0:
            inconsistency = np.clip(inconsistency, 0.0, 1.0)
            
            # Random walk component for variability
            random_noise = self.rng.normal(0, 0.1 * inconsistency, n)
            random_walk = np.cumsum(random_noise)
            random_walk -= np.mean(random_walk) # Center it
            if np.max(np.abs(random_walk)) > 1e-9:
                random_walk /= np.max(np.abs(random_walk)) # Scale to [-1, 1]
            
            # Sinusoidal component for reversion
            num_cycles = self.rng.uniform(1, 5)
            sin_wave = np.sin(np.linspace(0, num_cycles * 2 * np.pi, n))

            # Combine noise and apply to weights
            combined_noise = (random_walk + sin_wave) * 0.5 * inconsistency
            w = np.clip(w + combined_noise, 0.0, 1.0)

        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in dataframe")
                continue

            column_drift_value = None
            if drift_type in {"add_value", "subtract_value", "multiply_value", "divide_value"}:
                column_drift_value = (drift_values.get(col) if (drift_values and col in drift_values) else drift_value)
                if column_drift_value is None:
                    raise ValueError(f"For '{drift_type}', provide drift_value or drift_values['{col}']")
                if drift_type == "divide_value" and float(column_drift_value) == 0.0:
                    raise ValueError("drift_value cannot be zero for 'divide_value'")

            if pd.api.types.is_numeric_dtype(df[col]):
                x = df_drift.loc[rows, col].to_numpy(copy=True)
                x2 = self._apply_numeric_op_with_weights(
                    x, drift_type, drift_magnitude, w, self.rng, column_drift_value
                )
                df_drift.loc[rows, col] = x2
            else:
                s = df_drift.loc[rows, col]
                s2 = self._apply_categorical_with_weights(s, w, drift_magnitude, self.rng)
                df_drift.loc[rows, col] = s2

        drift_config = {
            "drift_method": "inject_feature_drift_gradual",
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            "drift_value": drift_value,
            "drift_values": drift_values,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "blocks": blocks,
            "block_start": block_start,
            "n_blocks": n_blocks,
            "center": center,
            "width": width,
            "profile": profile,
            "speed_k": speed_k,
            "direction": direction,
            "inconsistency": inconsistency,
            "generator_name": f"{self.generator_name}_feature_drift_gradual"
        }
        df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_drift, drift_config)
        return df_drift

    def inject_feature_drift_abrupt(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        change_index: Optional[int] = None,
        width: int = 3,
        direction: str = "up",
        **kwargs
    ) -> pd.DataFrame:
        """
        Injects abrupt drift, implemented as a very narrow and steep sigmoid transition.
        
        Args:
            change_index (Optional[int]): Position of the change within the selected section.
            width (int): A small width (e.g., 2-5) to simulate a sudden jump.
        """
        if block_column is None: block_column = self.block_column
        df_drift = self.inject_feature_drift_gradual(
            df=df,
            feature_cols=feature_cols,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            center=change_index,
            width=max(1, int(width)),
            profile="sigmoid",
            speed_k=5.0,        # high slope to make it more abrupt
            direction=direction,
            **kwargs
        )
        return df_drift

    def inject_feature_drift_incremental(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Injects a constant and smooth drift using a single wide sigmoid transition.
        This creates a steady, incremental change across the selected rows.
        """
        if block_column is None: block_column = self.block_column
        df_out = df.copy()
        if blocks is not None or (block_start is not None and n_blocks is not None):
            if not block_column:
                raise ValueError("block_column is required for block-based selection")
            rows = self._select_rows_by_blocks(df, block_column, blocks, block_start, n_blocks)
        else:
            rows = self._get_target_rows(df, start_index, block_index, block_column)

        n = len(rows)
        if n == 0:
            return df_out

        # A single, wide transition covering the entire span of the selected rows
        center = n / 2
        width = n

        # Call the gradual drift function with a single, wide profile
        tmp = self.inject_feature_drift_gradual(
            df=df_out,
            feature_cols=feature_cols,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            center=int(round(center)),
            width=width,
            profile="sigmoid",
            speed_k=1.0, # Standard speed
            direction="up",
            **kwargs
        )
        
        drift_config = {
            "drift_method": "inject_feature_drift_incremental",
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "blocks": blocks,
            "block_start": block_start,
            "n_blocks": n_blocks,
            "center": int(round(center)),
            "width": width,
            "generator_name": f"{self.generator_name}_feature_drift_incremental"
        }
        tmp.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, tmp, drift_config)
        return tmp

    def inject_feature_drift_recurrent(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        windows: Optional[Sequence[Tuple[int, int]]] = None,
        block_column: Optional[str] = None,
        cycle_blocks: Optional[Sequence] = None,
        repeats: int = 1,
        random_repeat_order: bool = False, # New parameter for random repetition
        center_in_block: Optional[int] = None,
        width_in_block: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Injects recurrent drift by applying several drift windows separated in time.
        
        Args:
            windows (Optional[Sequence[Tuple[int, int]]]): Explicit list of (center, width) tuples for drift windows.
            cycle_blocks (Optional[Sequence]): A sequence of block IDs to cycle through for applying drift.
            repeats (int): How many times to repeat the `cycle_blocks` pattern.
            random_repeat_order (bool): If True, shuffles the order of repeated blocks.
            center_in_block (Optional[int]): Relative center of the drift window within each block.
            width_in_block (Optional[int]): Width of the drift window within each block.
        """
        if block_column is None: block_column = self.block_column
        df_out = df.copy()
        intermediate_kwargs = kwargs.copy()

        if windows is not None:
            if blocks is not None or (block_start is not None and n_blocks is not None):
                if not block_column:
                    raise ValueError("block_column required for block selection")
                rows = self._select_rows_by_blocks(df, block_column, blocks, block_start, n_blocks)
            else:
                rows = self._get_target_rows(df, start_index, block_index, block_column)

            for (c, w_width) in windows:
                df_out = self.inject_feature_drift_gradual(
                    df=df_out,
                    feature_cols=feature_cols,
                    drift_type=drift_type,
                    drift_magnitude=drift_magnitude,
                    start_index=start_index,
                    block_index=block_index,
                    block_column=block_column,
                    blocks=blocks,
                    block_start=block_start,
                    n_blocks=n_blocks,
                    center=int(c),
                    width=max(1, int(w_width)),
                    profile=profile,
                    speed_k=speed_k,
                    direction=direction,
                    **intermediate_kwargs
                )

            drift_config = {
                "drift_method": "inject_feature_drift_recurrent",
                "feature_cols": feature_cols,
                "drift_type": drift_type,
                "drift_magnitude": drift_magnitude,
                "windows": windows,
                "start_index": start_index,
                "block_index": block_index,
                "block_column": block_column,
                "blocks": blocks,
                "block_start": block_start,
                "n_blocks": n_blocks,
                "generator_name": f"{self.generator_name}_feature_drift_recurrent"
            }
            df_out.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
            self._generate_reports(df, df_out, drift_config)
            return df_out

        if not (block_column and cycle_blocks):
            raise ValueError("For cycle pattern you must indicate block_column and cycle_blocks")

        uniq = df[block_column].dropna().unique().tolist()
        if not uniq: return df_out

        cycle_blocks = list(cycle_blocks)
        cycle_len = len(cycle_blocks)
        if cycle_len == 0: return df_out

        pattern = cycle_blocks * max(1, int(repeats))
        
        if random_repeat_order:
            self.rng.shuffle(pattern)

        applied_windows = []
        for target_block in pattern:
            rows = df.index[df[block_column] == target_block]
            n = len(rows)
            if n == 0: continue

            c = int(n // 2) if center_in_block is None else int(np.clip(center_in_block, 0, n - 1))
            w_width = max(1, int(width_in_block if width_in_block is not None else max(1, n // 3)))

            applied_windows.append({
                "block_id": target_block,
                "center": c,
                "width": w_width
            })

            df_out = self.inject_feature_drift_gradual(
                df=df_out,
                feature_cols=feature_cols,
                drift_type=drift_type,
                drift_magnitude=drift_magnitude,
                block_column=block_column,
                blocks=[target_block],
                center=c,
                width=w_width,
                profile=profile,
                speed_k=speed_k,
                direction=direction,
                **intermediate_kwargs
            )

        drift_config = {
            "drift_method": "inject_feature_drift_recurrent",
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            "cycle_blocks": cycle_blocks,
            "repeats": repeats,
            "applied_windows": applied_windows,
            "center_in_block": center_in_block,
            "width_in_block": width_in_block,
            "profile": profile,
            "speed_k": speed_k,
            "direction": direction,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "blocks": blocks,
            "block_start": block_start,
            "n_blocks": n_blocks,
            "generator_name": f"{self.generator_name}_feature_drift_recurrent"
        }
        df_out.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_out, drift_config)
        return df_out

    # -------------------------
    # Label drift (original) + gradual version
    # -------------------------
    def inject_label_drift(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        drift_magnitude: float = 0.1,
        drift_magnitudes: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        auto_report: bool = True
    ) -> pd.DataFrame:
        """
        Injects random label flips for a specified section (simple abrupt drift).
        
        Args:
            target_cols (List[str]): The target column(s) to apply drift to.
            drift_magnitude (float): The fraction of labels to flip.
        """
        if block_column is None: block_column = self.block_column
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)

        if isinstance(target_cols, str):
            target_cols = [target_cols]

        for target_col in target_cols:
            if target_col not in df.columns:
                warnings.warn(f"Target column '{target_col}' not found in dataframe")
                continue

            col_mag = float(drift_magnitudes.get(target_col, drift_magnitude)) if drift_magnitudes else float(drift_magnitude)
            uniques = df_drift[target_col].unique()
            if len(uniques) < 2:
                warnings.warn(f"Cannot inject label drift in '{target_col}' with less than 2 unique labels")
                continue

            n = len(rows)
            n_flips = int(n * self._frac(col_mag))
            if n_flips <= 0 or n == 0:
                continue

            idxs = self.rng.choice(rows, size=min(n_flips, n), replace=False)
            cur = df_drift.loc[idxs, target_col].to_numpy()
            new_vals = []
            for v in cur:
                choices = [u for u in uniques if u != v]
                new_vals.append(self.rng.choice(choices) if choices else v)
            df_drift.loc[idxs, target_col] = new_vals

        if auto_report:
            drift_config = {
                "drift_method": "inject_label_drift", "target_cols": target_cols,
                "drift_magnitude": drift_magnitude, "drift_magnitudes": drift_magnitudes,
                "start_index": start_index, "block_index": block_index, "block_column": block_column,
                "generator_name": f"{self.generator_name}_label_drift"
            }
            df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
            self._generate_reports(df, df_drift, drift_config)
        return df_drift

    def inject_label_drift_gradual(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float = 0.3,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        inconsistency: float = 0.0, # New parameter for inconsistent drift
        auto_report: bool = True
    ) -> pd.DataFrame:
        """Injects gradual label drift using a transition window."""
        if block_column is None: block_column = self.block_column
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)
        n = len(rows)
        if n == 0 or target_col not in df.columns: return df_drift

        uniques = df_drift[target_col].unique()
        if len(uniques) < 2: return df_drift

        c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))
        
        w = self._window_weights(n, center=c, width=w_width, profile=profile, k=speed_k, direction=direction)

        if inconsistency > 0 and n > 0:
            inconsistency = np.clip(inconsistency, 0.0, 1.0)
            
            # Random walk component for variability
            random_noise = self.rng.normal(0, 0.1 * inconsistency, n)
            random_walk = np.cumsum(random_noise)
            random_walk -= np.mean(random_walk) # Center it
            if np.max(np.abs(random_walk)) > 1e-9:
                random_walk /= np.max(np.abs(random_walk)) # Scale to [-1, 1]
            
            # Sinusoidal component for reversion
            num_cycles = self.rng.uniform(1, 5)
            sin_wave = np.sin(np.linspace(0, num_cycles * 2 * np.pi, n))

            # Combine noise and apply to weights
            combined_noise = (random_walk + sin_wave) * 0.5 * inconsistency
            w = np.clip(w + combined_noise, 0.0, 1.0)

        p = np.clip(w * self._frac(drift_magnitude), 0.0, 1.0)

        mask = self.rng.random(n) < p
        idxs = rows[mask]
        cur = df_drift.loc[idxs, target_col].to_numpy()
        new_vals = [self.rng.choice([u for u in uniques if u != v]) if len(uniques) > 1 else v for v in cur]
        df_drift.loc[idxs, target_col] = new_vals

        if auto_report:
            drift_config = {
                "drift_method": "inject_label_drift_gradual", "target_column": target_col,
                "drift_magnitude": drift_magnitude, "start_index": start_index, "block_index": block_index,
                "block_column": block_column, "center": center, "width": width, "profile": profile,
                "speed_k": speed_k, "direction": direction, "inconsistency": inconsistency,
                "generator_name": f"{self.generator_name}_label_drift_gradual"
            }
            df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
            self._generate_reports(df, df_drift, drift_config)
        return df_drift

    def inject_label_drift_abrupt(self, df: pd.DataFrame, target_col: str, drift_magnitude: float, change_index: int, **kwargs) -> pd.DataFrame:
        """Wrapper for a very fast gradual drift to simulate an abrupt change."""
        return self.inject_label_drift_gradual(
            df=df, target_col=target_col, drift_magnitude=drift_magnitude,
            center=change_index, width=3, speed_k=5.0, **kwargs
        )

    def inject_label_drift_incremental(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float,
        **kwargs
    ) -> pd.DataFrame:
        """Applies a constant and smooth label drift over the selected rows."""
        df_out = df.copy()
        rows = self._get_target_rows(df, kwargs.get("start_index"), kwargs.get("block_index"), kwargs.get("block_column"))
        n = len(rows)
        if n == 0: return df

        # A single, wide transition
        center = n / 2
        width = n

        tmp = self.inject_label_drift_gradual(
            df=df_out, 
            target_col=target_col, 
            drift_magnitude=drift_magnitude,
            center=int(round(center)), 
            width=width, 
            auto_report=False, 
            **kwargs
        )
        
        drift_config = {
            "drift_method": "inject_label_drift_incremental", 
            "target_column": target_col,
            "drift_magnitude": drift_magnitude, 
            **kwargs,
            "generator_name": f"{self.generator_name}_label_drift_incremental"
        }
        tmp.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, tmp, drift_config)
        return tmp

    def inject_label_drift_recurrent(self, df: pd.DataFrame, target_col: str, drift_magnitude: float, windows: List[Tuple[int, int]], **kwargs) -> pd.DataFrame:
        """Applies label drift over a series of explicit windows."""
        df_out = df.copy()
        intermediate_kwargs = kwargs.copy()

        for (center, width) in windows:
            df_out = self.inject_label_drift_gradual(
                df=df_out, target_col=target_col, drift_magnitude=drift_magnitude,
                center=center, width=width, auto_report=False, **intermediate_kwargs
            )

        drift_config = {
            "drift_method": "inject_label_drift_recurrent", "target_column": target_col,
            "drift_magnitude": drift_magnitude, "windows": windows, **kwargs,
            "generator_name": f"{self.generator_name}_label_drift_recurrent"
        }
        df_out.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_out, drift_config)
        return df_out

    # -------------------------
    # Target distribution drift
    # -------------------------
    def inject_label_shift(
        self,
        df: pd.DataFrame,
        target_col: str,
        target_distribution: dict,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Injects label shift by resampling the target column to match a new distribution.
        
        Args:
            target_distribution (dict): A dictionary mapping each label to its desired proportion (e.g., {0: 0.8, 1: 0.2}).
                                        Proportions must sum to 1.0.
        """
        if block_column is None: block_column = self.block_column
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        if len(rows) == 0:
            warnings.warn("No rows selected for target distribution drift")
            return df_drift

        uniques = df_drift.loc[rows, target_col].unique()
        total_prop = float(sum(target_distribution.values()))
        if abs(total_prop - 1.0) > 1e-6:
            raise ValueError(f"Target distribution proportions must sum to 1.0, got {total_prop}")

        for label in target_distribution.keys():
            if label not in uniques:
                warnings.warn(f"Label {label} not found in target column")

        n = len(rows)
        labels = list(target_distribution.keys())
        proportions = list(target_distribution.values())
        new_labels = self.rng.choice(labels, size=n, p=proportions)
        
        # Shuffle to ensure randomness
        self.rng.shuffle(new_labels)
        
        # Get the original dtype and cast new_labels
        original_dtype = df[target_col].dtype
        df_drift.loc[rows, target_col] = new_labels.astype(original_dtype)
        
        drift_config = {
            "drift_method": "inject_label_shift",
            "target_column": target_col,
            "target_distribution": target_distribution,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "generator_name": f"{self.generator_name}_label_shift"
        }
        df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_drift, drift_config)
        return df_drift

    # -------------------------
    # Virtual Drift (Missing Values)
    # -------------------------
    def inject_missing_values_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        missing_fraction: float = 0.1,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Injects missing values (NaN) into specified columns for a subset of data.
        This simulates virtual drift where the data distribution itself doesn't change,
        but the data quality does.
        
        Args:
            missing_fraction (float): The fraction of rows in the selected subset to introduce NaNs.
        """
        if block_column is None: block_column = self.block_column
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)

        if not isinstance(feature_cols, list):
            feature_cols = [feature_cols]

        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in dataframe, skipping.")
                continue

            # Select a random subset of rows to set to NaN
            n_missing = int(len(rows) * self._frac(missing_fraction))
            if n_missing > 0:
                missing_indices = self.rng.choice(rows, size=n_missing, replace=False)
                df_drift.loc[missing_indices, col] = np.nan

        drift_config = {
            "drift_method": "inject_missing_values_drift",
            "feature_cols": feature_cols,
            "missing_fraction": missing_fraction,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "generator_name": f"{self.generator_name}_missing_values_drift"
        }
        df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_drift, drift_config)
        return df_drift

    # -------------------------
    # Covariate Shift (Correlation Matrix Drift)
    # -------------------------
    def inject_correlation_matrix_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_correlation_matrix: np.ndarray,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Injects covariate drift by transforming numeric features to match a target correlation matrix.
        This method uses Cholesky decomposition for a mathematically robust transformation.

        Args:
            target_correlation_matrix (np.ndarray): The target correlation matrix (numpy array).
        """
        if block_column is None: block_column = self.block_column
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        
        if len(numeric_cols) < 2 or len(rows) == 0:
            warnings.warn("Need at least 2 numeric columns and some rows to apply correlation drift.")
            return df_drift

        k = len(numeric_cols)
        if not isinstance(target_correlation_matrix, np.ndarray) or target_correlation_matrix.shape != (k, k):
            raise ValueError(f"target_correlation_matrix must be a square ndarray of shape ({k}, {k}).")

        data = df_drift.loc[rows, numeric_cols].to_numpy(copy=True)
        
        # Apply Cholesky transformation
        transformed_data = self._apply_cholesky_transformation(data, target_correlation_matrix)
        
        df_drift.loc[rows, numeric_cols] = transformed_data

        drift_config = {
            "drift_method": "inject_correlation_matrix_drift",
            "feature_cols": numeric_cols,
            "target_correlation_matrix": target_correlation_matrix.tolist(),
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "generator_name": f"{self.generator_name}_correlation_matrix_drift"
        }
        df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_drift, drift_config)
        return df_drift

    def inject_covariate_shift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        shift_strength: float = 0.3,
        feature_pairs: Optional[List[tuple]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        [DEPRECATED] Injects covariate drift by modifying correlations between pairs of features.
        This method is deprecated in favor of inject_correlation_matrix_drift for more robust control.
        It now constructs a target correlation matrix and calls the new method.
        """
        warnings.warn(
            "`inject_covariate_shift` is deprecated and will be removed in a future version. "
            "Please use `inject_correlation_matrix_drift` for more robust correlation control.",
            DeprecationWarning,
            stacklevel=2
        )

        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            warnings.warn("Need at least 2 numeric columns for covariate shift.")
            return df.copy()

        # Get original correlation matrix for the selected numeric columns
        original_corr = df[numeric_cols].corr().to_numpy()

        # If no pairs are given, select a few random ones
        if feature_pairs is None:
            from itertools import combinations
            all_pairs = list(combinations(numeric_cols, 2))
            num_pairs_to_select = min(3, len(all_pairs))
            selected_indices = self.rng.choice(len(all_pairs), size=num_pairs_to_select, replace=False)
            feature_pairs = [all_pairs[i] for i in selected_indices]

        # Create the target correlation matrix
        col_to_idx = {col: i for i, col in enumerate(numeric_cols)}
        pair_indices = []
        for col1, col2 in feature_pairs:
            if col1 in col_to_idx and col2 in col_to_idx:
                pair_indices.append((col_to_idx[col1], col_to_idx[col2]))

        target_corr = original_corr.copy()
        for i, j in pair_indices:
            current_corr = target_corr[i, j]
            target_val = np.sign(current_corr) if current_corr != 0 else 1
            new_corr = current_corr * (1 - shift_strength) + target_val * shift_strength
            target_corr[i, j] = target_corr[j, i] = new_corr

        # Ensure the matrix is positive semi-definite (PSD)
        eigenvalues, eigenvectors = np.linalg.eigh(target_corr)
        eigenvalues[eigenvalues < 1e-6] = 1e-6 # Clamp small eigenvalues
        target_corr_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Renormalize to have 1s on the diagonal
        d = np.sqrt(np.diag(target_corr_psd))
        d_inv = np.where(d > 1e-9, 1.0 / d, 0)
        target_corr_psd = np.diag(d_inv) @ target_corr_psd @ np.diag(d_inv)

        # Call the new robust method
        return self.inject_correlation_matrix_drift(
            df=df,
            feature_cols=numeric_cols,
            target_correlation_matrix=target_corr_psd,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column
        )

    def _apply_cholesky_transformation(self, data: np.ndarray, target_corr: np.ndarray) -> np.ndarray:
        """Applies Cholesky decomposition to transform data to a target correlation structure."""
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0, ddof=0)
        safe_stds = np.where(stds == 0, 1.0, stds)
        Z = (data - means) / safe_stds

        try:
            # Decorrelate data using current correlation
            current_corr = np.corrcoef(Z, rowvar=False)
            # Check for NaN/Inf in current_corr
            if not np.all(np.isfinite(current_corr)):
                warnings.warn("Could not compute current correlation matrix, contains NaN/Inf. Skipping transformation.")
                return data

            L_c = np.linalg.cholesky(current_corr)
            L_c_inv = np.linalg.inv(L_c)
            decorrelated_Z = Z @ L_c_inv.T

            # Correlate data using target correlation
            L_t = np.linalg.cholesky(target_corr)
            correlated_Z = decorrelated_Z @ L_t.T

            # Rescale to original mean and std dev
            return correlated_Z * safe_stds + means

        except np.linalg.LinAlgError as e:
            warnings.warn(f"Cholesky decomposition failed: {e}. Using original data.")
            return data

    # -------------------------
    # Drift schedule (optional)
    # -------------------------
    def inject_multiple_types_of_drift(
        self,
        df: pd.DataFrame,
        schedule: List[Dict]
    ) -> pd.DataFrame:
        """
        Applies a sequence of different drift injections based on a schedule.

        Args:
            schedule (List[Dict]): A list of dictionaries, where each dictionary defines a drift injection step.
                                  Example:
                                  [
                                      {
                                          "mode": "gradual",
                                          "feature_cols": ["feature1"],
                                          "drift_type": "shift",
                                          "drift_magnitude": 0.5,
                                          "block_start": 2,
                                          "n_blocks": 3
                                      },
                                      {
                                          "mode": "abrupt",
                                          "target_col": "target",
                                          "drift_magnitude": 0.4,
                                          "block_index": 5
                                      }
                                  ]
        """
        out = df.copy()
        for i, step in enumerate(schedule):
            mode = step.get("mode", "gradual")
            fn = {
                "gradual": self.inject_feature_drift_gradual,
                "abrupt": self.inject_feature_drift_abrupt,
                "incremental": self.inject_feature_drift_incremental,
                "recurrent": self.inject_feature_drift_recurrent,
            }.get(mode)
            if fn is None:
                raise ValueError(f"Unsupported mode in schedule: {mode}")
            
            args = {k: v for k, v in step.items() if k != "mode"}

            # Ensure block_column from the instance is used if not specified in the step
            if 'block_column' not in args and self.block_column:
                args['block_column'] = self.block_column

            out = fn(df=out, **args)
        return out

    # -------------------------
    # New Category Drift (Concept Drift)
    # -------------------------
    def inject_new_category_drift(
        self,
        df: pd.DataFrame,
        feature_col: str,
        new_category: object,
        candidate_logic: dict,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid"
    ) -> pd.DataFrame:
        """
        Injects a new category into a feature column based on specified logic,
        simulating a form of concept drift. The introduction can be gradual.
        
        Args:
            feature_col (str): The categorical column to modify.
            new_category (object): The new category value to introduce.
            candidate_logic (dict): Logic to identify which rows are candidates for the new category.
                                    Example:
                                    {
                                        "conditions": [
                                            {"column": "age", "operator": ">", "threshold": 50},
                                            {"column": "city", "type": "categorical", "value": "New York"}
                                        ]
                                    }
                                    or
                                    {
                                        "replace_category": "OldValue",
                                        "fraction": 0.5
                                    }
        """
        if block_column is None: block_column = self.block_column
        df_drift = df.copy()

        if feature_col not in df.columns:
            warnings.warn(f"Feature column '{feature_col}' not found in dataframe.")
            return df_drift

        base_rows = self._get_target_rows(df, start_index, block_index, block_column)
        df_subset = df.loc[base_rows].copy()
        if df_subset.empty:
            return df_drift

        candidate_mask = pd.Series(False, index=df_subset.index)
        if 'conditions' in candidate_logic:
            final_mask = pd.Series(True, index=df_subset.index)
            for condition in candidate_logic['conditions']:
                col = condition.get('column')
                if col not in df_subset.columns:
                    warnings.warn(f"Column '{col}' from candidate_logic not in dataframe, skipping condition.")
                    continue
                
                cond_type = condition.get('type')
                if not cond_type:
                    cond_type = 'categorical' if pd.api.types.is_string_dtype(df_subset[col]) or pd.api.types.is_categorical_dtype(df_subset[col]) else 'numeric'

                if cond_type == 'categorical':
                    value = condition.get('value')
                    final_mask &= (df_subset[col] == value)
                elif cond_type == 'numeric':
                    op_str = condition.get('operator')
                    threshold = condition.get('threshold')
                    if op_str == '<': final_mask &= (df_subset[col] < threshold)
                    elif op_str == '>': final_mask &= (df_subset[col] > threshold)
                    elif op_str == '==': final_mask &= (df_subset[col] == threshold)
                    elif op_str == '<=': final_mask &= (df_subset[col] <= threshold)
                    elif op_str == '>=': final_mask &= (df_subset[col] >= threshold)
            candidate_mask = final_mask

        elif 'replace_category' in candidate_logic:
            cat_to_replace = candidate_logic['replace_category']
            fraction = candidate_logic.get('fraction', 1.0)
            
            base_candidates_mask = (df_subset[feature_col] == cat_to_replace)
            base_candidates_idx = df_subset.index[base_candidates_mask]

            if fraction < 1.0:
                n_to_take = int(len(base_candidates_idx) * fraction)
                selected_idx = self.rng.choice(base_candidates_idx, size=n_to_take, replace=False)
                temp_mask = pd.Series(False, index=df_subset.index)
                temp_mask.loc[selected_idx] = True
                candidate_mask = temp_mask
            else:
                candidate_mask = base_candidates_mask
        
        candidate_rows = df_subset.index[candidate_mask]
        n = len(candidate_rows)
        if n == 0:
            return df_drift

        relative_center = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))
        w = self._window_weights(n, center=relative_center, width=w_width, profile=profile, k=1.0, direction="up")

        change_probability = w
        change_mask = self.rng.random(n) < change_probability
        rows_to_change = candidate_rows[change_mask]
        
        if isinstance(df_drift[feature_col].dtype, pd.CategoricalDtype):
            if new_category not in df_drift[feature_col].cat.categories:
                df_drift[feature_col] = df_drift[feature_col].cat.add_categories([new_category])

        df_drift.loc[rows_to_change, feature_col] = new_category
        
        drift_config = {
            "drift_method": "inject_new_category_drift",
            "feature_col": feature_col,
            "new_category": new_category,
            "candidate_logic": candidate_logic,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "center": center,
            "width": width,
            "profile": profile,
            "generator_name": f"{self.generator_name}_new_category_drift"
        }
        df_drift.to_csv(os.path.join(self.output_dir, f'{drift_config["generator_name"]}.csv'), index=False)
        self._generate_reports(df, df_drift, drift_config)
        
        return df_drift