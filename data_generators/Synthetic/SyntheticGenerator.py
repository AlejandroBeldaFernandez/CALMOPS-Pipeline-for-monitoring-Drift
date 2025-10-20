import os
import json
import logging
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Iterator, Union
import pandas as pd
import numpy as np
from collections import defaultdict

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from .SyntheticReporter import SyntheticReporter  # reporter to save JSON reports
from river.datasets import synth as rv_synth

logger = logging.getLogger(__name__)

class SyntheticGenerator:
    """
    Synthetic data generator using River-backed generators with detailed configuration and reporting.

    This class orchestrates the generation of synthetic datasets, handling various types of drift,
    data balancing, timestamp injection, and comprehensive reporting.

    Key Features:
    - **Drift Simulation**: Supports 'none', 'virtual', 'gradual', 'incremental', and 'abrupt' drift types.
    - **Data Balancing**: Can balance the class distribution of the generated dataset.
    - **Timestamp Injection**: Adds a configurable timestamp column to simulate time-series data.
    - **Flexible Drift Control**: Allows fine-grained control over drift characteristics like position, width, and inconsistency.
    - **Comprehensive Reporting**: Automatically generates a detailed JSON report and visualizations using `SyntheticReporter`.
    """
    DEFAULT_OUTPUT_DIR = "real_time_output"

    @classmethod
    def set_default_output_dir(cls, path: str):
        """Sets the default directory for saving generated files."""
        cls.DEFAULT_OUTPUT_DIR = path

    @classmethod
    def get_default_output_dir(cls):
        """Gets the default directory for saving generated files."""
        return cls.DEFAULT_OUTPUT_DIR

    def __init__(self, random_state: Optional[int] = None):
        """
        Initializes the SyntheticGenerator.

        Args:
            random_state (Optional[int]): Seed for the random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)

    def generate(
        self,
        generator_instance,
        output_path: Optional[str],
        filename: str,
        n_samples: int,
        drift_type: str = "none",
        position_of_drift: int = None,
        target_col: str = "target",
        block_column: Optional[str] = None,
        balance: bool = False,
        inconsistency: float = 0.0,
        drift_options: Optional[Dict] = None,
        date_start: Optional[str] = None,
        date_every: int = 1,
        date_step: Optional[Dict[str, int]] = None,
        date_col: str = "timestamp",
        report_path_override: Optional[str] = None,
        prebuilt_instances: Optional[List[object]] = None,
        segment_lengths: Optional[List[int]] = None,
        segment_widths: Optional[List[int]] = None,
        segment_positions: Optional[List[Optional[int]]] = None,
        last_segment_pure: bool = True,
        durations_seconds: Optional[List[float]] = None,
        samples_per_second: Optional[float] = None,
        transition_width: Optional[int] = None,
        segment_label_ratios: Optional[List[Optional[Dict]]] = None,
        drift_generator: Optional[object] = None,
        save_dataset: bool = True
    ) -> Union[pd.DataFrame, str]:
        """
        Main public method to generate a synthetic dataset.

        Args:
            generator_instance: An instantiated River generator.
            output_path (Optional[str]): Directory to save the output files. Defaults to `DEFAULT_OUTPUT_DIR`.
            filename (str): Name of the output CSV file.
            n_samples (int): Total number of samples to generate.
            drift_type (str): Type of drift to inject ('none', 'virtual', 'gradual', 'abrupt', etc.).
            position_of_drift (int): The sample index where the drift should occur.
            target_col (str): Name for the target variable column.
            balance (bool): If True, balances the class distribution.
            inconsistency (float): A factor to add random noise to gradual drift transitions.
            drift_options (Optional[Dict]): Additional options for specific drift types (e.g., `missing_fraction`).
            date_start (Optional[str]): Start date for timestamp injection (e.g., '2023-01-01').
            date_every (int): Generate a new date every N rows.
            date_step (Optional[Dict[str, int]]): Time step for date injection (e.g., {'days': 1}).
            date_col (str): Name of the timestamp column.
            drift_generator (Optional[object]): A second River generator instance for drift simulation.
            save_dataset (bool): If True, saves the generated DataFrame to a CSV file and returns the path. 
                                If False, returns the DataFrame directly.

        Returns:
            Union[pd.DataFrame, str]: The generated DataFrame or the path to the saved CSV file.
        """
        out_dir = self._resolve_output_dir(output_path)
        df = self._generate_internal(
            generator_instance=generator_instance,
            output_path=out_dir,
            filename=filename,
            n_samples=n_samples,
            generator_instance_drift=drift_generator,
            position_of_drift=position_of_drift,
            target_col=target_col,
            block_column=block_column,
            balance=balance,
            drift_type=drift_type,
            inconsistency=inconsistency,
            drift_options=drift_options or {},
            extra_info=None,
            date_start=date_start,
            date_every=date_every,
            date_step=date_step,
            date_col=date_col,
            prebuilt_instances=prebuilt_instances,
            segment_lengths=segment_lengths,
            segment_widths=segment_widths,
            segment_positions=segment_positions,
            last_segment_pure=last_segment_pure,
            durations_seconds=durations_seconds,
            samples_per_second=samples_per_second,
            transition_width=transition_width,
            segment_label_ratios=segment_label_ratios,
        )
        if save_dataset:
            full_csv_path = os.path.join(out_dir, filename)
            df.to_csv(full_csv_path, index=False)
            logger.info(f"Data generated and saved at: {full_csv_path}")
            return full_csv_path
        return df

    def _generate_internal(self, **kwargs) -> pd.DataFrame:
        """Internal generation logic that constructs the DataFrame and triggers reporting."""
        n_samples = kwargs["n_samples"]
        generator_instance = kwargs["generator_instance"]
        balance = kwargs["balance"]
        target_col = kwargs["target_col"]
        drift_type = kwargs["drift_type"]

        data_gen_instance = generator_instance

        if drift_type == "virtual_drift":
            drift_options = kwargs.get("drift_options", {})
            pos = kwargs.get("position_of_drift") or n_samples // 2
            missing_fraction = drift_options.get("missing_fraction", 0.1)
            feature_cols = drift_options.get("feature_cols")
            data_gen_instance = self._virtual_drift_generator(
                generator_instance, pos, missing_fraction, feature_cols
            )

        if drift_type in ["gradual", "incremental", "abrupt"]:
            A = generator_instance
            B = kwargs["generator_instance_drift"]
            if not B: raise ValueError(f"drift_generator must be provided for {drift_type} drift")
            
            pos = kwargs.get("position_of_drift") or n_samples // 2
            width = kwargs.get("transition_width")
            inconsistency = kwargs.get("inconsistency", 0.0)

            if drift_type == "gradual":
                width = width if width is not None else n_samples // 10
            elif drift_type == "incremental":
                width = n_samples
                pos = n_samples // 2
            elif drift_type == "abrupt":
                data = (list(A.take(pos)) + list(B.take(n_samples - pos)))
                data = [list(x.values()) + [y] for x, y in data]
                data_gen_instance = None # Data is already generated

            if data_gen_instance:
                data = self._build_drifted_rows(A, B, n_samples, pos, width, inconsistency)
                data_gen_instance = None # Data is already generated
        
        if data_gen_instance:
             data = (self._generate_balanced(data_gen_instance, n_samples)
                    if balance else self._generate_data(data_gen_instance, n_samples))

        try:
            first_sample = next(iter(generator_instance.take(1)))
            columns = list(first_sample[0].keys()) + [target_col]
        except Exception:
            n_features = len(data[0]) - 1 if data else 0
            columns = [f"x{i}" for i in range(n_features)] + [target_col]
            logger.warning("Could not infer feature names; using x0..xK")

        df = pd.DataFrame(data, columns=columns)
        df = self._inject_dates(df, kwargs["date_col"], kwargs["date_start"], kwargs["date_every"], kwargs["date_step"])

        report_kwargs = {k: v for k, v in kwargs.items() if k != 'save_dataset'}
        self._save_report_json(df=df, output_dir=kwargs["output_path"], **report_kwargs)
        return df

    def _virtual_drift_generator(self, generator: Iterator, position_of_drift: int, missing_fraction: float, feature_cols: Optional[List[str]]) -> Iterator[Tuple[Dict, int]]:
        """A generator that injects missing values (NaN) after a certain position."""
        feature_names = feature_cols
        for i, (x, y) in enumerate(generator):
            if i < position_of_drift:
                yield x, y
            else:
                if feature_names is None:
                    feature_names = list(x.keys())
                
                x_drifted = x.copy()
                for col in feature_names:
                    if self.rng.random() < missing_fraction:
                        x_drifted[col] = np.nan
                yield x_drifted, y

    def _window_weights(self, n: int, center: float, width: int, profile: str = "sigmoid", k: float = 1.0) -> np.ndarray:
        """Generates a window of weights for smooth transitions between concepts."""
        if n <= 0: return np.zeros(0, dtype=float)
        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        center = float(center)
        if profile == "sigmoid":
            base_scale = width / 4.0
            scale = max(1e-9, base_scale / max(1e-9, float(k)))
            z = (i - center) / scale
            w = 1.0 / (1.0 + np.exp(-z))
        else:
            left = center - width / 2.0
            right = center + width / 2.0
            w = np.clip((i - left) / max(1e-9, (right - left)), 0.0, 1.0)
        return w

    
    def _build_drifted_rows(self, base, drift, n_samples, position, width, inconsistency) -> List[List]:
        """Builds a dataset with a gradual transition from a base generator to a drift generator."""
        w = self._window_weights(n_samples, center=position, width=width)
        if inconsistency > 0:
            noise = self.rng.normal(0, 0.1*inconsistency, n_samples)
            walk = np.cumsum(noise)
            walk -= np.mean(walk)
            if np.max(np.abs(walk)) > 1e-9: walk /= np.max(np.abs(walk))
            sin_wave = np.sin(np.linspace(0, self.rng.uniform(1,5)*2*np.pi, n_samples))
            w = np.clip(w + (walk + sin_wave) * 0.5 * inconsistency, 0.0, 1.0)

        try:
            base_iter = base.take(n_samples)
            drift_iter = drift.take(n_samples)
        except AttributeError:
          
            base_iter = iter(base)
            drift_iter = iter(drift)

        rows = []
        for i in range(n_samples):
         
            it = drift_iter if self.rng.random() < w[i] else base_iter
            
            x, y = next(it)
            rows.append(list(x.values()) + [y])
        return rows

    def _generate_balanced(self, gen, n_samples) -> List[List]:
        """Generates samples and balances the classes to have roughly equal representation."""
        
        class_samples = defaultdict(list)
        max_samples = max(n_samples * 5, n_samples)
        
        # Handle both River objects and Python generators
        if hasattr(gen, 'take'):
            data_iterator = gen.take(max_samples)
        else:
            data_iterator = (next(gen) for _ in range(max_samples))
            
        for x, y in data_iterator:
            class_samples[y].append(list(x.values()) + [y])
            
        data = []
        per_class = n_samples // len(class_samples) if class_samples else n_samples
            
        for samples in class_samples.values():
            data.extend(samples[:per_class])
            
        if len(data) < n_samples and data:
            data.extend(random.choices(data, k=n_samples - len(data)))
            
        return data[:n_samples] if data else [] # Return empty list if no data

    def _generate_data(self, gen, n_samples) -> List[List]:
        """Generates n_samples from a River generator or a standard Python iterator."""
        
        if hasattr(gen, 'take'):
            data_iterator = gen.take(n_samples)
        else:
            # For standard Python generators (like _virtual_drift_generator)
            data_iterator = (next(gen) for _ in range(n_samples))
            
        return [list(x.values()) + [y] for x, y in data_iterator]

    def _inject_dates(self, df, date_col, date_start, date_every, date_step) -> pd.DataFrame:
        """Injects a date column into the DataFrame with specified frequency and step."""
        if not date_start:
            return df
        
        time_deltas = np.arange(len(df)) // date_every
        
        if date_step:
            # Create a DateOffset from the dictionary, e.g., {'days': 1} -> DateOffset(days=1)
            offset = pd.DateOffset(**date_step)
            series = pd.to_datetime(date_start) + time_deltas * offset
        else:
            # Default to days if no date_step is provided
            series = pd.to_datetime(date_start) + pd.to_timedelta(time_deltas, unit='D')
            
        df[date_col] = series
        return df

    def _save_report_json(self, df: pd.DataFrame, output_dir: str, **kwargs):
        """Saves a comprehensive JSON report of the generated data and its properties."""
        # Map kwargs to SyntheticReporter signature
        report_kwargs = {
            'target_column': kwargs.get('target_col'),
            'time_col': kwargs.get('date_col'),
            'block_column': kwargs.get('block_column'),
            'drift_config': {
                'drift_type': kwargs.get('drift_type'),
                'position_of_drift': kwargs.get('position_of_drift'),
                'transition_width': kwargs.get('transition_width'),
                'drift_options': kwargs.get('drift_options')
            }
        }

        # Filter out None values to avoid passing them
        report_kwargs_filtered = {k: v for k, v in report_kwargs.items() if v is not None}

        try:
            SyntheticReporter(verbose=True).generate_report(
                synthetic_df=df,
                generator_name=kwargs["generator_instance"].__class__.__name__,
                output_dir=output_dir,
                **report_kwargs_filtered
            )
        except Exception as e:
            logger.error(f"Could not generate report: {e}", exc_info=True)

    def validate_params(self, **kwargs):
        """Validates input parameters for the generate method."""
        if not (isinstance(kwargs.get("n_samples"), int) and kwargs.get("n_samples") > 0):
            raise ValueError("n_samples must be a positive integer")

    def _resolve_output_dir(self, path: Optional[str]) -> str:
        """Resolves the output directory path, creating it if it doesn't exist."""
        out = os.path.abspath(path or self.DEFAULT_OUTPUT_DIR)
        os.makedirs(out, exist_ok=True)
        return out