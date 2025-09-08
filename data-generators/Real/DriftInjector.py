#!/usr/bin/env python3
"""
Drift Injector for Real Data - Injects various types of drift into real datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import LabelEncoder
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
    
    def inject_feature_drift(self, 
                           df: pd.DataFrame,
                           feature_cols: List[str],
                           drift_magnitude: float = 0.2,
                           drift_type: str = "gaussian_noise") -> pd.DataFrame:
        """
        Inject feature drift by modifying feature distributions
        
        Args:
            df: Input dataframe
            feature_cols: List of feature columns to modify
            drift_magnitude: Magnitude of drift (0.1 = 10% change)
            drift_type: Type of drift ('gaussian_noise', 'shift', 'scale')
            
        Returns:
            Modified dataframe with feature drift
        """
        df_drift = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in dataframe")
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                original_values = df_drift[col].values
                
                if drift_type == "gaussian_noise":
                    # Add Gaussian noise
                    noise = np.random.normal(0, drift_magnitude * np.std(original_values), len(original_values))
                    df_drift[col] = original_values + noise
                    
                elif drift_type == "shift":
                    # Shift mean
                    shift_amount = drift_magnitude * np.mean(original_values)
                    df_drift[col] = original_values + shift_amount
                    
                elif drift_type == "scale":
                    # Scale variance
                    mean_val = np.mean(original_values)
                    df_drift[col] = mean_val + (original_values - mean_val) * (1 + drift_magnitude)
                    
            else:
                # For categorical features, introduce label noise
                unique_vals = df_drift[col].unique()
                if len(unique_vals) > 1:
                    n_changes = int(len(df_drift) * drift_magnitude)
                    indices_to_change = np.random.choice(len(df_drift), n_changes, replace=False)
                    
                    for idx in indices_to_change:
                        current_val = df_drift.iloc[idx][col]
                        other_vals = [v for v in unique_vals if v != current_val]
                        if other_vals:
                            df_drift.iloc[idx, df_drift.columns.get_loc(col)] = np.random.choice(other_vals)
        
        return df_drift
    
    def inject_label_drift(self,
                          df: pd.DataFrame,
                          target_col: str,
                          drift_magnitude: float = 0.1) -> pd.DataFrame:
        """
        Inject label drift by flipping some labels
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            drift_magnitude: Fraction of labels to flip (0.1 = 10%)
            
        Returns:
            Modified dataframe with label drift
        """
        df_drift = df.copy()
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        unique_labels = df_drift[target_col].unique()
        
        if len(unique_labels) < 2:
            warnings.warn("Cannot inject label drift with less than 2 unique labels")
            return df_drift
        
        # Flip labels for a fraction of samples
        n_flips = int(len(df_drift) * drift_magnitude)
        indices_to_flip = np.random.choice(len(df_drift), n_flips, replace=False)
        
        for idx in indices_to_flip:
            current_label = df_drift.iloc[idx][target_col]
            other_labels = [l for l in unique_labels if l != current_label]
            df_drift.iloc[idx, df_drift.columns.get_loc(target_col)] = np.random.choice(other_labels)
        
        return df_drift
    
    def inject_covariate_shift(self,
                              df: pd.DataFrame,
                              feature_cols: List[str],
                              shift_strength: float = 0.3) -> pd.DataFrame:
        """
        Inject covariate shift by changing feature correlations
        
        Args:
            df: Input dataframe
            feature_cols: List of numeric feature columns
            shift_strength: Strength of covariate shift
            
        Returns:
            Modified dataframe with covariate shift
        """
        df_drift = df.copy()
        
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            warnings.warn("Need at least 2 numeric columns for covariate shift")
            return df_drift
        
        # Create correlation shift between first two numeric columns
        col1, col2 = numeric_cols[0], numeric_cols[1]
        
        # Normalize features
        mean1, std1 = df_drift[col1].mean(), df_drift[col1].std()
        mean2, std2 = df_drift[col2].mean(), df_drift[col2].std()
        
        norm1 = (df_drift[col1] - mean1) / std1
        norm2 = (df_drift[col2] - mean2) / std2
        
        # Apply covariate shift transformation
        shifted_norm2 = norm2 + shift_strength * norm1
        
        # Denormalize
        df_drift[col2] = shifted_norm2 * std2 + mean2
        
        return df_drift
    
    def inject_temporal_drift(self,
                             df: pd.DataFrame,
                             n_segments: int = 3,
                             drift_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inject temporal drift by creating segments with different characteristics
        
        Args:
            df: Input dataframe
            n_segments: Number of temporal segments
            drift_cols: Columns to apply drift to (if None, use all numeric)
            
        Returns:
            Modified dataframe with temporal drift and segment information
        """
        df_drift = df.copy()
        
        if drift_cols is None:
            drift_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create segment column
        segment_size = len(df) // n_segments
        segments = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            if i == n_segments - 1:  # Last segment gets remaining rows
                end_idx = len(df)
            else:
                end_idx = (i + 1) * segment_size
            
            segments.extend([i + 1] * (end_idx - start_idx))
        
        df_drift['segment'] = segments
        
        # Apply increasing drift to each segment
        for segment_id in range(1, n_segments + 1):
            segment_mask = df_drift['segment'] == segment_id
            drift_strength = (segment_id - 1) * 0.2  # Increasing drift
            
            segment_df = df_drift[segment_mask].copy()
            
            # Apply feature drift to this segment
            for col in drift_cols:
                if col in df_drift.columns and pd.api.types.is_numeric_dtype(df_drift[col]):
                    original_values = segment_df[col].values
                    noise = np.random.normal(0, drift_strength * np.std(original_values), len(original_values))
                    df_drift.loc[segment_mask, col] = original_values + noise
        
        return df_drift