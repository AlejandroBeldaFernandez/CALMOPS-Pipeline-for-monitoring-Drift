"""
Real Block Data Generator with Enhanced Block-wise Processing
Implements block-based data generation and drift injection for real datasets
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import warnings
import logging
from datetime import datetime
import os

from .RealGenerator import RealGenerator
from ..DriftInjection.DriftInjector import DriftInjector
from .RealReporter import RealReporter


class RealBlockGenerator(RealGenerator):
    """
    Enhanced Real Block Data Generator with comprehensive block-wise processing
    
    Features:
    - Block-based data partitioning and generation
    - Progressive drift injection across blocks
    - Comprehensive block-wise statistics and validation
    - Dataset-level visualization (not per-block)
    - Enhanced reporting and quality assessment
    """
    
    def __init__(self,
                 dataset_path: str,
                 block_column: str,
                 synthesis_method: str = 'gmm',
                 target_column: Optional[str] = None,
                 categorical_columns: Optional[List[str]] = None,
                 auto_visualize: bool = True,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize RealBlockGenerator
        
        Parameters:
        -----------
        dataset_path : str
            Path to the original dataset
        block_column : str
            Name of the column that defines blocks
        synthesis_method : str
            Method for synthesis ('gmm', 'ctgan', 'copula', 'smote', 'resample')
        target_column : Optional[str]
            Name of target column
        categorical_columns : Optional[List[str]]
            List of categorical column names
        auto_visualize : bool
            Whether to generate automatic visualizations
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print detailed information
        """
        # Initialize parent class
        super().__init__(
            dataset_path=dataset_path,
            synthesis_method=synthesis_method,
            target_column=target_column,
            categorical_columns=categorical_columns,
            auto_visualize=auto_visualize,
            random_state=random_state,
            verbose=verbose
        )
        
        self.block_column = block_column
        
        # Validate block column
        if self.block_column not in self.original_data.columns:
            raise ValueError(f"Block column '{self.block_column}' not found in dataset")
        
        # Analyze blocks
        self.blocks = sorted(self.original_data[self.block_column].unique())
        self.n_blocks = len(self.blocks)
        
        if self.verbose:
            print(f"RealBlockGenerator initialized with {self.n_blocks} blocks")
            print(f"Blocks: {self.blocks}")
            print(f"Block sizes: {self.original_data[self.block_column].value_counts().sort_index().to_dict()}")
    
    def _generate_block_data(self, 
                           block_id: Any, 
                           n_samples: Optional[int] = None,
                           drift_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate synthetic data for a specific block"""
        # Get original block data
        block_data = self.original_data[self.original_data[self.block_column] == block_id].copy()
        block_data_no_block = block_data.drop(columns=[self.block_column])
        
        if len(block_data) == 0:
            raise ValueError(f"No data found for block: {block_id}")
        
        if n_samples is None:
            n_samples = len(block_data)
        
        self.logger.info(f"Generating {n_samples} samples for block {block_id}")
        
        # Generate synthetic data for this block
        if self.synthesis_method == 'gmm':
            synthetic_block = self._synthesize_gmm(block_data_no_block, n_samples)
        elif self.synthesis_method == 'ctgan':
            synthetic_block = self._synthesize_ctgan(block_data_no_block, n_samples)
        elif self.synthesis_method == 'copula':
            synthetic_block = self._synthesize_copula(block_data_no_block, n_samples)
        elif self.synthesis_method == 'smote':
            synthetic_block = self._synthesize_smote(block_data_no_block, n_samples)
        elif self.synthesis_method == 'resample':
            synthetic_block = self._synthesize_resample(block_data_no_block, n_samples)
        else:
            raise ValueError(f"Unknown synthesis method: {self.synthesis_method}")
        
        # Add block column back
        synthetic_block[self.block_column] = block_id
        
        # Apply drift if configured
        if drift_config:
            drift_injector = DriftInjector(**drift_config)
            synthetic_block = drift_injector.inject_drift(synthetic_block)
        
        return synthetic_block
    
    def generate_block_dataset(self,
                             samples_per_block: Optional[Union[int, Dict[Any, int]]] = None,
                             progressive_drift: bool = False,
                             drift_configs: Optional[Dict[Any, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Generate synthetic dataset with block-wise processing
        
        Parameters:
        -----------
        samples_per_block : Optional[Union[int, Dict[Any, int]]]
            Number of samples per block (uniform) or dict mapping block_id -> n_samples
        progressive_drift : bool
            Whether to apply progressive drift across blocks
        drift_configs : Optional[Dict[Any, Dict[str, Any]]]
            Dict mapping block_id -> drift_config for block-specific drift
        
        Returns:
        --------
        pd.DataFrame
            Complete synthetic dataset with all blocks
        """
        synthetic_blocks = []
        
        for i, block_id in enumerate(self.blocks):
            # Determine number of samples for this block
            if samples_per_block is None:
                n_samples = len(self.original_data[self.original_data[self.block_column] == block_id])
            elif isinstance(samples_per_block, int):
                n_samples = samples_per_block
            else:  # Dict
                n_samples = samples_per_block.get(block_id, 
                    len(self.original_data[self.original_data[self.block_column] == block_id]))
            
            # Determine drift configuration for this block
            drift_config = None
            if progressive_drift:
                # Apply increasing drift intensity
                drift_intensity = min(0.1 + (i * 0.1), 0.5)  # Cap at 50%
                drift_config = {
                    'drift_type': 'feature',
                    'affected_features': None,  # Auto-select
                    'drift_intensity': drift_intensity,
                    'random_state': self.random_state + i
                }
            elif drift_configs and block_id in drift_configs:
                drift_config = drift_configs[block_id]
            
            # Generate synthetic block
            synthetic_block = self._generate_block_data(
                block_id=block_id,
                n_samples=n_samples,
                drift_config=drift_config
            )
            
            synthetic_blocks.append(synthetic_block)
            
            if self.verbose:
                print(f"Generated block {block_id}: {len(synthetic_block)} samples")
        
        # Combine all blocks
        complete_dataset = pd.concat(synthetic_blocks, ignore_index=True)
        
        # Generate comprehensive report
        self._generate_block_report(complete_dataset)
        
        if self.verbose:
            print(f"Complete synthetic dataset generated: {complete_dataset.shape}")
            print(f"Block distribution: {complete_dataset[self.block_column].value_counts().sort_index().to_dict()}")
        
        return complete_dataset
    
    def _generate_block_report(self, synthetic_dataset: pd.DataFrame):
        """Generate comprehensive report for block dataset"""
        if not self.auto_visualize:
            return
        
        try:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output_real_block_generation_{self.synthesis_method}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate comprehensive report (dataset-level, not per-block)
            self.reporter.generate_comprehensive_report(
                real_df=self.original_data,
                synthetic_df=synthetic_dataset,
                generator_name=f"RealBlockGenerator_{self.synthesis_method}",
                output_dir=output_dir,
                target_column=self.target_column
            )
            
            self.logger.info(f"Block dataset report and visualizations saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate block report: {e}")
    
    def generate_drift_progression(self,
                                 samples_per_block: Optional[int] = None,
                                 start_drift_intensity: float = 0.0,
                                 end_drift_intensity: float = 0.5,
                                 drift_type: str = 'feature') -> pd.DataFrame:
        """
        Generate dataset with progressive drift across blocks
        
        Parameters:
        -----------
        samples_per_block : Optional[int]
            Number of samples per block
        start_drift_intensity : float
            Starting drift intensity (0.0 = no drift)
        end_drift_intensity : float
            Ending drift intensity
        drift_type : str
            Type of drift to apply ('feature', 'label', 'concept')
        
        Returns:
        --------
        pd.DataFrame
            Dataset with progressive drift
        """
        drift_configs = {}
        
        for i, block_id in enumerate(self.blocks):
            # Calculate drift intensity for this block
            if self.n_blocks > 1:
                intensity = start_drift_intensity + (
                    (end_drift_intensity - start_drift_intensity) * i / (self.n_blocks - 1)
                )
            else:
                intensity = start_drift_intensity
            
            if intensity > 0:
                drift_configs[block_id] = {
                    'drift_type': drift_type,
                    'affected_features': None,  # Auto-select
                    'drift_intensity': intensity,
                    'random_state': self.random_state + i
                }
        
        return self.generate_block_dataset(
            samples_per_block=samples_per_block,
            drift_configs=drift_configs
        )
    
    def analyze_block_statistics(self, synthetic_dataset: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistics for each block in the synthetic dataset"""
        block_stats = {}
        
        for block_id in self.blocks:
            # Original block data
            original_block = self.original_data[self.original_data[self.block_column] == block_id]
            synthetic_block = synthetic_dataset[synthetic_dataset[self.block_column] == block_id]
            
            # Basic statistics
            block_stats[block_id] = {
                'original_size': len(original_block),
                'synthetic_size': len(synthetic_block),
                'original_target_dist': None,
                'synthetic_target_dist': None
            }
            
            # Target distribution if available
            if self.target_column and self.target_column in original_block.columns:
                block_stats[block_id]['original_target_dist'] = original_block[self.target_column].value_counts().to_dict()
                block_stats[block_id]['synthetic_target_dist'] = synthetic_block[self.target_column].value_counts().to_dict()
            
            # Numeric feature statistics
            numeric_cols = original_block.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != self.block_column]
            
            if len(numeric_cols) > 0:
                block_stats[block_id]['original_numeric_means'] = original_block[numeric_cols].mean().to_dict()
                block_stats[block_id]['synthetic_numeric_means'] = synthetic_block[numeric_cols].mean().to_dict()
                block_stats[block_id]['original_numeric_stds'] = original_block[numeric_cols].std().to_dict()
                block_stats[block_id]['synthetic_numeric_stds'] = synthetic_block[numeric_cols].std().to_dict()
        
        return block_stats
    
    def get_block_info(self) -> Dict[str, Any]:
        """Get detailed information about blocks in the dataset"""
        block_info = {}
        
        for block_id in self.blocks:
            block_data = self.original_data[self.original_data[self.block_column] == block_id]
            
            block_info[block_id] = {
                'size': len(block_data),
                'percentage': len(block_data) / len(self.original_data) * 100,
                'target_distribution': None,
                'feature_means': None,
                'missing_values': block_data.isnull().sum().sum()
            }
            
            # Target distribution
            if self.target_column and self.target_column in block_data.columns:
                block_info[block_id]['target_distribution'] = block_data[self.target_column].value_counts().to_dict()
            
            # Feature means for numeric columns
            numeric_cols = block_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != self.block_column]
            if len(numeric_cols) > 0:
                block_info[block_id]['feature_means'] = block_data[numeric_cols].mean().to_dict()
        
        return block_info
    
    def save_block_dataset(self,
                          synthetic_dataset: pd.DataFrame,
                          output_path: str,
                          format: str = 'csv',
                          separate_blocks: bool = False) -> Union[str, List[str]]:
        """
        Save synthetic block dataset
        
        Parameters:
        -----------
        synthetic_dataset : pd.DataFrame
            The complete synthetic dataset
        output_path : str
            Output path (file or directory)
        format : str
            Output format ('csv', 'parquet', 'excel')
        separate_blocks : bool
            If True, save each block as a separate file
        
        Returns:
        --------
        Union[str, List[str]]
            Path(s) to saved file(s)
        """
        output_path = Path(output_path)
        
        if separate_blocks:
            # Save each block separately
            if output_path.is_file():
                output_dir = output_path.parent
                base_name = output_path.stem
                extension = output_path.suffix
            else:
                output_dir = output_path
                base_name = f"synthetic_block_{self.synthesis_method}"
                extension = f".{format}"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            
            for block_id in self.blocks:
                block_data = synthetic_dataset[synthetic_dataset[self.block_column] == block_id]
                block_path = output_dir / f"{base_name}_block_{block_id}{extension}"
                
                if format.lower() == 'csv':
                    block_data.to_csv(block_path, index=False)
                elif format.lower() == 'parquet':
                    block_data.to_parquet(block_path, index=False)
                elif format.lower() in ['xlsx', 'excel']:
                    block_data.to_excel(block_path, index=False)
                
                saved_paths.append(str(block_path))
                self.logger.info(f"Block {block_id} saved to: {block_path}")
            
            return saved_paths
        
        else:
            # Save complete dataset
            return self.save_synthetic_data(synthetic_dataset, output_path, format)