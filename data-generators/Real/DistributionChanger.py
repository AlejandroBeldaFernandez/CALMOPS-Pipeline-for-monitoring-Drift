#!/usr/bin/env python3
"""
Distribution Changer for Real Data - Changes class distributions in real datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from collections import Counter
import warnings

class DistributionChanger:
    """
    Changes class distributions in real datasets through various methods
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the DistributionChanger
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def change_class_distribution(self,
                                 df: pd.DataFrame,
                                 target_col: str,
                                 target_distribution: Dict[Union[str, int], float],
                                 method: str = "undersample") -> pd.DataFrame:
        """
        Change class distribution to match target distribution
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            target_distribution: Target distribution as {class: ratio}
            method: Method to use ('undersample', 'oversample', 'mixed')
            
        Returns:
            Modified dataframe with new class distribution
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Validate target distribution sums to 1
        total_ratio = sum(target_distribution.values())
        if abs(total_ratio - 1.0) > 0.01:
            warnings.warn(f"Target distribution sums to {total_ratio}, normalizing to 1.0")
            target_distribution = {k: v/total_ratio for k, v in target_distribution.items()}
        
        # Get current distribution
        current_counts = df[target_col].value_counts().to_dict()
        total_samples = len(df)
        
        # Calculate target counts
        target_counts = {k: int(total_samples * ratio) for k, ratio in target_distribution.items()}
        
        if method == "undersample":
            return self._undersample_to_distribution(df, target_col, target_counts)
        elif method == "oversample":
            return self._oversample_to_distribution(df, target_col, target_counts)
        elif method == "mixed":
            return self._mixed_sampling_to_distribution(df, target_col, target_counts)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_imbalanced_version(self,
                                 df: pd.DataFrame,
                                 target_col: str,
                                 imbalance_ratio: float = 0.1,
                                 minority_class: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Create an imbalanced version of a balanced dataset
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            imbalance_ratio: Ratio for minority class (0.1 = 10%)
            minority_class: Which class should be minority (if None, use smallest current class)
            
        Returns:
            Imbalanced dataframe
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        class_counts = df[target_col].value_counts()
        classes = list(class_counts.index)
        
        if len(classes) != 2:
            warnings.warn(f"Expected 2 classes, found {len(classes)}. Using first 2 classes only.")
            classes = classes[:2]
        
        if minority_class is None:
            minority_class = class_counts.idxmin()
        
        majority_class = [c for c in classes if c != minority_class][0]
        
        # Create target distribution
        target_distribution = {
            minority_class: imbalance_ratio,
            majority_class: 1.0 - imbalance_ratio
        }
        
        return self.change_class_distribution(df, target_col, target_distribution, method="undersample")
    
    def balance_dataset(self,
                       df: pd.DataFrame,
                       target_col: str,
                       method: str = "undersample") -> pd.DataFrame:
        """
        Balance a dataset to have equal class distributions
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            method: Balancing method ('undersample', 'oversample')
            
        Returns:
            Balanced dataframe
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        classes = df[target_col].unique()
        equal_ratio = 1.0 / len(classes)
        
        target_distribution = {cls: equal_ratio for cls in classes}
        
        return self.change_class_distribution(df, target_col, target_distribution, method=method)
    
    def create_drift_sequence(self,
                             df: pd.DataFrame,
                             target_col: str,
                             n_steps: int = 5,
                             drift_strength: float = 0.3) -> pd.DataFrame:
        """
        Create a sequence of datasets with gradual distribution drift
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            n_steps: Number of drift steps
            drift_strength: How much to drift at each step
            
        Returns:
            Dataframe with step column indicating drift progression
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        classes = list(df[target_col].unique())
        if len(classes) != 2:
            raise ValueError("Drift sequence currently supports only binary classification")
        
        # Calculate initial distribution
        initial_counts = df[target_col].value_counts(normalize=True)
        initial_dist = {classes[0]: initial_counts[classes[0]], classes[1]: initial_counts[classes[1]]}
        
        # Create drift steps
        samples_per_step = len(df) // n_steps
        result_dfs = []
        
        for step in range(n_steps):
            start_idx = step * samples_per_step
            if step == n_steps - 1:  # Last step gets remaining rows
                end_idx = len(df)
            else:
                end_idx = (step + 1) * samples_per_step
            
            step_df = df.iloc[start_idx:end_idx].copy()
            
            # Calculate drift for this step
            drift_amount = step * drift_strength / n_steps
            
            # Modify distribution
            if drift_amount > 0:
                new_dist = {
                    classes[0]: max(0.1, min(0.9, initial_dist[classes[0]] - drift_amount)),
                    classes[1]: max(0.1, min(0.9, initial_dist[classes[1]] + drift_amount))
                }
                
                # Normalize
                total = sum(new_dist.values())
                new_dist = {k: v/total for k, v in new_dist.items()}
                
                step_df = self.change_class_distribution(step_df, target_col, new_dist, method="mixed")
            
            step_df['drift_step'] = step + 1
            result_dfs.append(step_df)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _undersample_to_distribution(self,
                                   df: pd.DataFrame,
                                   target_col: str,
                                   target_counts: Dict) -> pd.DataFrame:
        """Undersample to achieve target distribution"""
        result_dfs = []
        
        for class_label, target_count in target_counts.items():
            class_df = df[df[target_col] == class_label]
            
            if len(class_df) >= target_count:
                sampled_df = class_df.sample(n=target_count, random_state=self.random_state)
            else:
                warnings.warn(f"Class {class_label} has {len(class_df)} samples, need {target_count}")
                sampled_df = class_df  # Use all available samples
            
            result_dfs.append(sampled_df)
        
        result = pd.concat(result_dfs, ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
    def _oversample_to_distribution(self,
                                  df: pd.DataFrame,
                                  target_col: str,
                                  target_counts: Dict) -> pd.DataFrame:
        """Oversample to achieve target distribution"""
        result_dfs = []
        
        for class_label, target_count in target_counts.items():
            class_df = df[df[target_col] == class_label]
            
            if len(class_df) < target_count:
                # Oversample with replacement
                sampled_df = class_df.sample(n=target_count, replace=True, random_state=self.random_state)
            else:
                # Undersample
                sampled_df = class_df.sample(n=target_count, random_state=self.random_state)
            
            result_dfs.append(sampled_df)
        
        result = pd.concat(result_dfs, ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
    def _mixed_sampling_to_distribution(self,
                                      df: pd.DataFrame,
                                      target_col: str,
                                      target_counts: Dict) -> pd.DataFrame:
        """Use mixed sampling strategy to achieve target distribution"""
        current_counts = df[target_col].value_counts().to_dict()
        total_current = sum(current_counts.values())
        total_target = sum(target_counts.values())
        
        if total_target <= total_current:
            # Use undersampling
            return self._undersample_to_distribution(df, target_col, target_counts)
        else:
            # Use oversampling
            return self._oversample_to_distribution(df, target_col, target_counts)