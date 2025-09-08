#!/usr/bin/env python3
"""
Block Drift Generator for Real Data - Creates blocks with different distributions and drift
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from .DriftInjector import DriftInjector
from .DistributionChanger import DistributionChanger
from .RealReporter import RealReporter

class RealBlockDriftGenerator(RealReporter):
    """
    Generates blocks of real data with different distributions and drift patterns
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the RealBlockDriftGenerator
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.random_state = random_state
        self.drift_injector = DriftInjector(random_state=random_state)
        self.dist_changer = DistributionChanger(random_state=random_state)
    
    def generate_blocks_from_real_data(self,
                                     df: pd.DataFrame,
                                     target_col: str,
                                     output_path: str,
                                     filename: str,
                                     n_blocks: int,
                                     block_sizes: Optional[List[int]] = None,
                                     block_distributions: Optional[List[Dict]] = None,
                                     drift_configs: Optional[List[Dict]] = None) -> str:
        """
        Generate multiple blocks from real data with different characteristics
        
        Args:
            df: Original real dataset
            target_col: Name of target column
            output_path: Directory to save output
            filename: Output filename
            n_blocks: Number of blocks to generate
            block_sizes: Size of each block (if None, equal sizes)
            block_distributions: Distribution for each block
            drift_configs: Drift configuration for each block
            
        Returns:
            Path to generated file
        """
        os.makedirs(output_path, exist_ok=True)
        
        if block_sizes is None:
            total_size = len(df)
            base_size = total_size // n_blocks
            block_sizes = [base_size] * n_blocks
            # Add remaining samples to last block
            block_sizes[-1] += total_size - sum(block_sizes)
        
        if len(block_sizes) != n_blocks:
            raise ValueError(f"Number of block sizes ({len(block_sizes)}) must match n_blocks ({n_blocks})")
        
        # Generate each block
        block_dfs = []
        
        for block_id in range(n_blocks):
            print(f"Generating block {block_id + 1}/{n_blocks}...")
            
            block_size = block_sizes[block_id]
            
            # Sample base data for this block
            if block_size > len(df):
                # Oversample if needed
                block_df = df.sample(n=block_size, replace=True, random_state=self.random_state)
            else:
                block_df = df.sample(n=block_size, replace=False, random_state=self.random_state)
            
            block_df = block_df.copy().reset_index(drop=True)
            
            # Apply distribution changes if specified
            if block_distributions and block_id < len(block_distributions):
                dist_config = block_distributions[block_id]
                if dist_config:
                    block_df = self.dist_changer.change_class_distribution(
                        block_df, target_col, dist_config, method="mixed"
                    )
            
            # Apply drift if specified
            if drift_configs and block_id < len(drift_configs):
                drift_config = drift_configs[block_id]
                if drift_config:
                    block_df = self._apply_drift_config(block_df, target_col, drift_config)
            
            # Add block identifier
            block_df['block'] = block_id + 1
            block_dfs.append(block_df)
        
        # Combine all blocks
        final_df = pd.concat(block_dfs, ignore_index=True)
        
        # Save to file
        output_file = os.path.join(output_path, filename)
        final_df.to_csv(output_file, index=False)
        
        # Generate report
        self._report_blocks(final_df, target_col, n_blocks)
        
        print(f"Generated {n_blocks} blocks with {len(final_df)} total samples")
        print(f"Saved to: {output_file}")
        
        return output_file
    
    def generate_temporal_drift_blocks(self,
                                     df: pd.DataFrame,
                                     target_col: str,
                                     output_path: str,
                                     filename: str,
                                     n_blocks: int,
                                     drift_strength: float = 0.2,
                                     drift_type: str = "gradual") -> str:
        """
        Generate blocks with temporal drift patterns
        
        Args:
            df: Original dataset
            target_col: Target column name
            output_path: Output directory
            filename: Output filename
            n_blocks: Number of temporal blocks
            drift_strength: Strength of drift between blocks
            drift_type: Type of drift ('gradual', 'sudden', 'cyclic')
            
        Returns:
            Path to generated file
        """
        os.makedirs(output_path, exist_ok=True)
        
        block_size = len(df) // n_blocks
        block_dfs = []
        
        # Get initial class distribution
        initial_dist = df[target_col].value_counts(normalize=True).to_dict()
        classes = list(initial_dist.keys())
        
        for block_id in range(n_blocks):
            print(f"Generating temporal block {block_id + 1}/{n_blocks}...")
            
            start_idx = block_id * block_size
            if block_id == n_blocks - 1:
                end_idx = len(df)
            else:
                end_idx = (block_id + 1) * block_size
            
            block_df = df.iloc[start_idx:end_idx].copy()
            
            # Calculate drift based on block position and drift type
            if drift_type == "gradual":
                # Linear drift
                drift_factor = block_id * drift_strength / n_blocks
            elif drift_type == "sudden":
                # Sudden drift at midpoint
                drift_factor = drift_strength if block_id >= n_blocks // 2 else 0
            elif drift_type == "cyclic":
                # Cyclic drift
                drift_factor = drift_strength * np.sin(2 * np.pi * block_id / n_blocks)
            else:
                drift_factor = 0
            
            # Apply distribution drift
            if drift_factor != 0 and len(classes) == 2:
                new_dist = {
                    classes[0]: max(0.1, min(0.9, initial_dist[classes[0]] - drift_factor)),
                    classes[1]: max(0.1, min(0.9, initial_dist[classes[1]] + drift_factor))
                }
                # Normalize
                total = sum(new_dist.values())
                new_dist = {k: v/total for k, v in new_dist.items()}
                
                block_df = self.dist_changer.change_class_distribution(
                    block_df, target_col, new_dist, method="mixed"
                )
            
            # Add temporal information
            block_df['block'] = block_id + 1
            block_df['time_step'] = block_id + 1
            
            block_dfs.append(block_df)
        
        # Combine all blocks
        final_df = pd.concat(block_dfs, ignore_index=True)
        
        # Save to file
        output_file = os.path.join(output_path, filename)
        final_df.to_csv(output_file, index=False)
        
        # Generate report
        self._report_temporal_blocks(final_df, target_col, n_blocks, drift_type)
        
        print(f"Generated {n_blocks} temporal blocks with {drift_type} drift")
        print(f"Saved to: {output_file}")
        
        return output_file
    
    def _apply_drift_config(self, df: pd.DataFrame, target_col: str, config: Dict) -> pd.DataFrame:
        """Apply drift configuration to a dataframe"""
        result_df = df.copy()
        
        drift_type = config.get('type', 'none')
        
        if drift_type == 'feature':
            feature_cols = config.get('features', [])
            magnitude = config.get('magnitude', 0.2)
            method = config.get('method', 'gaussian_noise')
            
            result_df = self.drift_injector.inject_feature_drift(
                result_df, feature_cols, magnitude, method
            )
        
        elif drift_type == 'label':
            magnitude = config.get('magnitude', 0.1)
            result_df = self.drift_injector.inject_label_drift(
                result_df, target_col, magnitude
            )
        
        elif drift_type == 'covariate':
            feature_cols = config.get('features', [])
            strength = config.get('strength', 0.3)
            result_df = self.drift_injector.inject_covariate_shift(
                result_df, feature_cols, strength
            )
        
        return result_df
    
    def _report_blocks(self, df: pd.DataFrame, target_col: str, n_blocks: int):
        """Generate report for block-based data"""
        print(f"\nBLOCK GENERATION REPORT")
        print("=" * 40)
        print(f"Total samples: {len(df)}")
        print(f"Number of blocks: {n_blocks}")
        print(f"Features: {len(df.columns) - 2}")  # -2 for target and block columns
        
        # Block-wise statistics
        for block_id in range(1, n_blocks + 1):
            block_data = df[df['block'] == block_id]
            block_dist = block_data[target_col].value_counts().to_dict()
            print(f"Block {block_id}: {len(block_data)} samples, distribution: {block_dist}")
    
    def _report_temporal_blocks(self, df: pd.DataFrame, target_col: str, n_blocks: int, drift_type: str):
        """Generate report for temporal blocks"""
        print(f"\nTEMPORAL BLOCK REPORT - {drift_type.upper()} DRIFT")
        print("=" * 50)
        print(f"Total samples: {len(df)}")
        print(f"Number of temporal blocks: {n_blocks}")
        
        # Show distribution change over time
        for block_id in range(1, n_blocks + 1):
            block_data = df[df['block'] == block_id]
            block_dist = block_data[target_col].value_counts(normalize=True).round(3).to_dict()
            print(f"Time step {block_id}: distribution {block_dist}")