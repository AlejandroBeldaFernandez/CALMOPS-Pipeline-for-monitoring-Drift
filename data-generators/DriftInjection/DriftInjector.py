#!/usr/bin/env python3
"""
Drift Injector for Real Data - Injects various types of drift into real datasets
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings

class DriftInjector:
    """
    Injects various types of drift into real datasets
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the DriftInjector
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_target_rows(self, df: pd.DataFrame, start_index: Optional[int], 
                         block_index: Optional[int], block_column: Optional[str]) -> pd.Index:
        """
        Determine the row indices where the drift should be applied
        """
        if block_index is not None and block_column is not None:
            if block_column not in df.columns:
                raise ValueError(f"Block column '{block_column}' not found in dataframe")
            rows = df.index[df[block_column] == block_index]
        else:
            start_index = 0 if start_index is None else max(0, start_index)
            rows = df.index[start_index:]
        return rows
    
    def inject_feature_drift(self, 
                         df: pd.DataFrame,
                         feature_cols: List[str],
                         drift_type: str = "gaussian_noise",
                         drift_magnitude: float = 0.2,
                         drift_value: Optional[float] = None,
                         drift_values: Optional[Dict[str, float]] = None,
                         start_index: Optional[int] = None,
                         block_index: Optional[int] = None,
                         block_column: Optional[str] = None) -> pd.DataFrame:
        """
        Inject feature drift by modifying feature distributions
        
        Args:
            df: Input dataframe
            feature_cols: List of feature columns to modify
            drift_type: Type of drift to apply
            drift_magnitude: Magnitude for relative drift types (gaussian_noise, shift, scale)
            drift_value: Single value for absolute drift types (add_value, etc.) - applied to all columns
            drift_values: Dictionary of column-specific values (e.g., {'col1': 5, 'col2': -2})
            start_index: Index from which to start applying drift
            block_index: Specific block to apply drift
            block_column: Column name that identifies blocks
            
        Returns:
            DataFrame with modified features
        """
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)
        
        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in dataframe")
                continue
            
            # Determine the value to use for this column
            if drift_values and col in drift_values:
                column_drift_value = drift_values[col]
            else:
                column_drift_value = drift_value
            
            if pd.api.types.is_numeric_dtype(df[col]):
                original_values = df_drift.loc[rows, col].values
                
                if drift_type == "gaussian_noise":
                    noise = np.random.normal(0, drift_magnitude * np.std(original_values), len(original_values))
                    df_drift.loc[rows, col] = original_values + noise
                    
                elif drift_type == "shift":
                    shift_amount = drift_magnitude * np.mean(original_values)
                    df_drift.loc[rows, col] = original_values + shift_amount
                    
                elif drift_type == "scale":
                    mean_val = np.mean(original_values)
                    df_drift.loc[rows, col] = mean_val + (original_values - mean_val) * (1 + drift_magnitude)
                
                elif drift_type == "add_value":
                    if column_drift_value is None:
                        raise ValueError(f"drift_value or drift_values['{col}'] must be specified for add_value")
                    df_drift.loc[rows, col] = original_values + column_drift_value
                
                elif drift_type == "subtract_value":
                    if column_drift_value is None:
                        raise ValueError(f"drift_value or drift_values['{col}'] must be specified for subtract_value")
                    df_drift.loc[rows, col] = original_values - column_drift_value
                
                elif drift_type == "multiply_value":
                    if column_drift_value is None:
                        raise ValueError(f"drift_value or drift_values['{col}'] must be specified for multiply_value")
                    df_drift.loc[rows, col] = original_values * column_drift_value
                
                elif drift_type == "divide_value":
                    if column_drift_value is None:
                        raise ValueError(f"drift_value or drift_values['{col}'] must be specified for divide_value")
                    if column_drift_value == 0:
                        raise ValueError(f"drift_value cannot be zero for divide_value (column: {col})")
                    df_drift.loc[rows, col] = original_values / column_drift_value
                
                else:
                    raise ValueError(f"Unknown drift_type: {drift_type}")
            
            else:
                # Para features categóricos se mantiene el mismo comportamiento
                unique_vals = df_drift[col].unique()
                if len(unique_vals) > 1:
                    n_changes = int(len(rows) * drift_magnitude)
                    if len(rows) > 0 and n_changes > 0:
                        indices_to_change = np.random.choice(rows, min(n_changes, len(rows)), replace=False)
                        for idx in indices_to_change:
                            current_val = df_drift.loc[idx, col]
                            other_vals = [v for v in unique_vals if v != current_val]
                            if other_vals:
                                df_drift.loc[idx, col] = np.random.choice(other_vals)
        
        return df_drift
    
    def inject_label_drift(self,
                           df: pd.DataFrame,
                           target_cols: List[str],
                           drift_magnitude: float = 0.1,
                           drift_magnitudes: Optional[Dict[str, float]] = None,
                           start_index: Optional[int] = None,
                           block_index: Optional[int] = None,
                           block_column: Optional[str] = None) -> pd.DataFrame:
        """
        Inject label drift by flipping some labels
        
        Args:
            df: Input dataframe
            target_cols: List of target columns to modify (supports multiple targets)
            drift_magnitude: Default magnitude for all columns (fraction of labels to flip)
            drift_magnitudes: Column-specific magnitudes (e.g., {'target1': 0.1, 'target2': 0.2})
            start_index: Index from which to start applying drift
            block_index: Specific block to apply drift
            block_column: Column name that identifies blocks
            
        Returns:
            DataFrame with modified labels
        """
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)
        
        # Support both single string and list for backward compatibility
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        
        for target_col in target_cols:
            if target_col not in df.columns:
                warnings.warn(f"Target column '{target_col}' not found in dataframe")
                continue
            
            # Determine magnitude for this column
            if drift_magnitudes and target_col in drift_magnitudes:
                column_magnitude = drift_magnitudes[target_col]
            else:
                column_magnitude = drift_magnitude
            
            unique_labels = df_drift[target_col].unique()
            if len(unique_labels) < 2:
                warnings.warn(f"Cannot inject label drift in '{target_col}' with less than 2 unique labels")
                continue
            
            n_flips = int(len(rows) * column_magnitude)
            if n_flips > 0 and len(rows) > 0:
                indices_to_flip = np.random.choice(rows, min(n_flips, len(rows)), replace=False)
                
                for idx in indices_to_flip:
                    current_label = df_drift.loc[idx, target_col]
                    other_labels = [l for l in unique_labels if l != current_label]
                    if other_labels:
                        df_drift.loc[idx, target_col] = np.random.choice(other_labels)
        
        return df_drift
    
    def inject_target_distribution_drift(self,
                                       df: pd.DataFrame,
                                       target_col: str,
                                       target_distribution: dict,
                                       start_index: Optional[int] = None,
                                       block_index: Optional[int] = None,
                                       block_column: Optional[str] = None) -> pd.DataFrame:
        """
        Inject target distribution drift by changing the proportion of classes
        
        Args:
            df: Input dataframe
            target_col: Name of the target column
            target_distribution: Dictionary with desired class proportions (e.g., {0: 0.9, 1: 0.1})
            start_index: Index from which to start applying drift
            block_index: Specific block to apply drift
            block_column: Column name that identifies blocks
            
        Returns:
            DataFrame with modified target distribution
        """
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        if len(rows) == 0:
            warnings.warn("No rows selected for target distribution drift")
            return df_drift
        
        # Validate target distribution
        unique_labels = df_drift[target_col].unique()
        total_proportion = sum(target_distribution.values())
        if abs(total_proportion - 1.0) > 1e-6:
            raise ValueError(f"Target distribution proportions must sum to 1.0, got {total_proportion}")
        
        for label in target_distribution.keys():
            if label not in unique_labels:
                warnings.warn(f"Label {label} not found in target column")
        
        # Calculate number of samples for each class
        n_drift_samples = len(rows)
        target_counts = {}
        for label, proportion in target_distribution.items():
            target_counts[label] = int(n_drift_samples * proportion)
        
        # Adjust for rounding errors
        total_assigned = sum(target_counts.values())
        if total_assigned < n_drift_samples:
            # Add remaining samples to the most frequent class
            most_frequent = max(target_counts.keys(), key=lambda x: target_counts[x])
            target_counts[most_frequent] += n_drift_samples - total_assigned
        
        # Create new labels array
        new_labels = []
        for label, count in target_counts.items():
            new_labels.extend([label] * count)
        
        # Shuffle to randomize assignment
        np.random.shuffle(new_labels)
        
        # Assign new labels
        for i, row_idx in enumerate(rows):
            if i < len(new_labels):
                df_drift.loc[row_idx, target_col] = new_labels[i]
        
        return df_drift
    
    def inject_covariate_shift(self,
                               df: pd.DataFrame,
                               feature_cols: List[str],
                               shift_strength: float = 0.3,
                               feature_pairs: Optional[List[tuple]] = None,
                               correlation_changes: Optional[Dict[tuple, float]] = None,
                               start_index: Optional[int] = None,
                               block_index: Optional[int] = None,
                               block_column: Optional[str] = None) -> pd.DataFrame:
        """
        Inject covariate shift by changing feature correlations
        
        Args:
            df: Input dataframe
            feature_cols: List of feature columns to consider
            shift_strength: Default strength of correlation changes
            feature_pairs: Specific pairs of features to modify (e.g., [('f1', 'f2'), ('f3', 'f4')])
            correlation_changes: Specific correlation changes per pair (e.g., {('f1', 'f2'): 0.5})
            start_index: Index from which to start applying drift
            block_index: Specific block to apply drift
            block_column: Column name that identifies blocks
            
        Returns:
            DataFrame with modified feature correlations
        """
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)
        
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) < 2:
            warnings.warn("Need at least 2 numeric columns for covariate shift")
            return df_drift
        
        # Determine feature pairs to modify
        if feature_pairs is None:
            # Generate all possible pairs if not specified
            from itertools import combinations
            feature_pairs = list(combinations(numeric_cols, 2))
            # Limit to first few pairs to avoid excessive modifications
            feature_pairs = feature_pairs[:min(3, len(feature_pairs))]
        
        # Apply correlation changes to each pair
        for pair in feature_pairs:
            col1, col2 = pair
            
            if col1 not in numeric_cols or col2 not in numeric_cols:
                warnings.warn(f"Skipping pair ({col1}, {col2}) - not both numeric")
                continue
            
            # Get correlation change strength for this pair
            if correlation_changes and pair in correlation_changes:
                strength = correlation_changes[pair]
            elif correlation_changes and (col2, col1) in correlation_changes:
                strength = correlation_changes[(col2, col1)]
            else:
                strength = shift_strength
            
            # Apply covariate shift to this pair
            self._apply_pair_correlation_shift(df_drift, rows, col1, col2, strength)
        
        return df_drift
    
    def _apply_pair_correlation_shift(self, df_drift: pd.DataFrame, rows: pd.Index, 
                                    col1: str, col2: str, strength: float):
        """
        Apply correlation shift between two specific columns
        """
        if len(rows) == 0:
            return
            
        # Get original statistics
        mean1, std1 = df_drift.loc[rows, col1].mean(), df_drift.loc[rows, col1].std()
        mean2, std2 = df_drift.loc[rows, col2].mean(), df_drift.loc[rows, col2].std()
        
        if std1 == 0 or std2 == 0:
            warnings.warn(f"Zero standard deviation in pair ({col1}, {col2}), skipping")
            return
        
        # Normalize features
        norm1 = (df_drift.loc[rows, col1] - mean1) / std1
        norm2 = (df_drift.loc[rows, col2] - mean2) / std2
        
        # Apply correlation shift: add weighted influence of col1 to col2
        shifted_norm2 = norm2 + strength * norm1
        
        # Denormalize and assign back
        df_drift.loc[rows, col2] = shifted_norm2 * std2 + mean2

    def inject_advanced_covariate_shift(self,
                                       df: pd.DataFrame,
                                       feature_cols: List[str],
                                       correlation_matrix: Optional[np.ndarray] = None,
                                       shift_method: str = "linear",
                                       start_index: Optional[int] = None,
                                       block_index: Optional[int] = None,
                                       block_column: Optional[str] = None) -> pd.DataFrame:
        """
        Advanced covariate shift using correlation matrix transformations
        
        Args:
            df: Input dataframe
            feature_cols: List of numeric feature columns
            correlation_matrix: Target correlation matrix (if None, adds random correlations)
            shift_method: Method to apply shift ('linear', 'cholesky')
            start_index: Index from which to start applying drift
            block_index: Specific block to apply drift
            block_column: Column name that identifies blocks
            
        Returns:
            DataFrame with transformed correlations
        """
        df_drift = df.copy()
        rows = self._get_target_rows(df, start_index, block_index, block_column)
        
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) < 2:
            warnings.warn("Need at least 2 numeric columns for advanced covariate shift")
            return df_drift
        
        if len(rows) == 0:
            return df_drift
        
        # Extract data subset
        data_subset = df_drift.loc[rows, numeric_cols].values
        
        if shift_method == "cholesky" and correlation_matrix is not None:
            # Use Cholesky decomposition for precise correlation control
            data_subset = self._apply_cholesky_transformation(data_subset, correlation_matrix)
        else:
            # Default linear method with random correlation changes
            data_subset = self._apply_linear_correlation_shift(data_subset)
        
        # Assign back to dataframe
        df_drift.loc[rows, numeric_cols] = data_subset
        
        return df_drift
    
    def _apply_cholesky_transformation(self, data: np.ndarray, target_corr: np.ndarray) -> np.ndarray:
        """
        Apply Cholesky decomposition to achieve target correlation structure
        """
        # Standardize data
        data_std = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        # Current correlation
        current_corr = np.corrcoef(data_std.T)
        
        # Cholesky decomposition of target correlation
        try:
            L_target = np.linalg.cholesky(target_corr)
            L_current = np.linalg.cholesky(current_corr)
            L_current_inv = np.linalg.inv(L_current)
            
            # Transform data
            data_transformed = data_std @ L_current_inv.T @ L_target.T
            
            # Restore original scale
            original_mean = np.mean(data, axis=0)
            original_std = np.std(data, axis=0)
            data_transformed = data_transformed * original_std + original_mean
            
            return data_transformed
        except np.linalg.LinAlgError:
            warnings.warn("Cholesky decomposition failed, using original data")
            return data
    
    def _apply_linear_correlation_shift(self, data: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Apply linear correlation shifts between random pairs of features
        """
        data_transformed = data.copy()
        n_features = data.shape[1]
        
        if n_features < 2:
            return data_transformed
        
        # Generate random pairs and apply shifts
        from itertools import combinations
        pairs = list(combinations(range(n_features), 2))
        
        for i, j in pairs[:min(3, len(pairs))]:  # Limit modifications
            # Standardize columns
            col_i = (data_transformed[:, i] - np.mean(data_transformed[:, i])) / np.std(data_transformed[:, i])
            col_j = (data_transformed[:, j] - np.mean(data_transformed[:, j])) / np.std(data_transformed[:, j])
            
            # Apply correlation shift
            shifted_col_j = col_j + strength * col_i
            
            # Restore scale and assign back
            mean_j, std_j = np.mean(data[:, j]), np.std(data[:, j])
            data_transformed[:, j] = shifted_col_j * std_j + mean_j
        
        return data_transformed