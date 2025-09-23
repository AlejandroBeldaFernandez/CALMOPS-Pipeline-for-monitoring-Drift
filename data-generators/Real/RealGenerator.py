"""
Real Data Generator with Enhanced Synthesis and Drift Injection
Enhanced with comprehensive statistics, improved visualizations, and multiple synthesis methods
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import warnings
import logging
from datetime import datetime
import os

from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from ..DriftInjection.DriftInjector import DriftInjector
from .RealReporter import RealReporter


class RealGenerator:
    """
    Enhanced Real Data Generator with multiple synthesis methods and comprehensive validation
    
    Features:
    - Multiple synthesis methods: GMM, CTGAN, Copula, SMOTE, Resample
    - Comprehensive statistical validation and quality assessment
    - Automatic visualization generation
    - Enhanced drift injection capabilities
    - Detailed reporting and logging
    """
    
    SUPPORTED_METHODS = ['gmm', 'ctgan', 'copula', 'smote', 'resample']
    
    def __init__(self, 
                 dataset_path: str,
                 synthesis_method: str = 'gmm',
                 target_column: Optional[str] = None,
                 categorical_columns: Optional[List[str]] = None,
                 auto_visualize: bool = True,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize RealGenerator with enhanced capabilities
        
        Parameters:
        -----------
        dataset_path : str
            Path to the original dataset
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
        self.dataset_path = Path(dataset_path)
        self.synthesis_method = synthesis_method.lower()
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.auto_visualize = auto_visualize
        self.random_state = random_state
        self.verbose = verbose
        
        # Validate synthesis method
        if self.synthesis_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported synthesis method: {self.synthesis_method}. "
                           f"Supported methods: {self.SUPPORTED_METHODS}")
        
        # Load original dataset
        self.original_data = self._load_dataset()
        self.reporter = RealReporter()
        
        # Setup logging
        self._setup_logging()
        
        if self.verbose:
            print(f"RealGenerator initialized with method: {self.synthesis_method}")
            print(f"Original dataset shape: {self.original_data.shape}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset from various formats"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        file_ext = self.dataset_path.suffix.lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(self.dataset_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(self.dataset_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(self.dataset_path)
            elif file_ext == '.json':
                return pd.read_json(self.dataset_path)
            elif file_ext == '.arff':
                from scipy.io.arff import loadarff
                data, meta = loadarff(self.dataset_path)
                return pd.DataFrame(data)
            else:
                # Default to CSV
                return pd.read_csv(self.dataset_path)
                
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def _prepare_metadata(self, data: pd.DataFrame) -> SingleTableMetadata:
        """Prepare metadata for SDV synthesizers"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        
        # Update categorical columns if specified
        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='categorical')
        
        return metadata
    
    def _synthesize_gmm(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using Gaussian Mixture Model"""
        try:
            # Prepare numeric data for GMM
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = [col for col in data.columns if col not in numeric_cols]
            
            if len(numeric_cols) == 0:
                raise ValueError("GMM requires at least one numeric column")
            
            # Fit GMM on numeric data
            gmm = GaussianMixture(
                n_components=min(10, len(data) // 50 + 1),
                random_state=self.random_state,
                max_iter=200
            )
            
            X_numeric = data[numeric_cols].values
            gmm.fit(X_numeric)
            
            # Generate synthetic numeric data
            X_synthetic = gmm.sample(n_samples)[0]
            synthetic_df = pd.DataFrame(X_synthetic, columns=numeric_cols)
            
            # Handle categorical columns by sampling
            for col in categorical_cols:
                if col in data.columns:
                    synthetic_df[col] = np.random.choice(
                        data[col].values, size=n_samples, replace=True
                    )
            
            return synthetic_df[data.columns]  # Maintain column order
            
        except Exception as e:
            self.logger.error(f"GMM synthesis failed: {e}")
            raise
    
    def _synthesize_ctgan(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using CTGAN"""
        try:
            metadata = self._prepare_metadata(data)
            
            synthesizer = CTGANSynthesizer(
                metadata=metadata,
                epochs=100,
                verbose=False
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                synthesizer.fit(data)
                synthetic_data = synthesizer.sample(num_rows=n_samples)
            
            return synthetic_data
            
        except Exception as e:
            self.logger.error(f"CTGAN synthesis failed: {e}")
            # Fallback to GMM
            self.logger.warning("Falling back to GMM method")
            return self._synthesize_gmm(data, n_samples)
    
    def _synthesize_copula(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using Gaussian Copula"""
        try:
            metadata = self._prepare_metadata(data)
            
            synthesizer = GaussianCopulaSynthesizer(
                metadata=metadata,
                default_distribution='beta'
            )
            
            synthesizer.fit(data)
            synthetic_data = synthesizer.sample(num_rows=n_samples)
            
            return synthetic_data
            
        except Exception as e:
            self.logger.error(f"Copula synthesis failed: {e}")
            # Fallback to GMM
            self.logger.warning("Falling back to GMM method")
            return self._synthesize_gmm(data, n_samples)
    
    def _synthesize_smote(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using SMOTE"""
        if self.target_column is None:
            raise ValueError("SMOTE requires a target column")
        
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        try:
            # Prepare data for SMOTE
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]
            
            # Only works with numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(X.columns):
                X = X[numeric_cols]
                self.logger.warning(f"SMOTE: Using only numeric columns: {list(numeric_cols)}")
            
            # Apply SMOTE
            smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, len(data)-1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Create synthetic dataframe
            synthetic_df = pd.DataFrame(X_resampled, columns=X.columns)
            synthetic_df[self.target_column] = y_resampled
            
            # Sample to desired size
            if len(synthetic_df) > n_samples:
                synthetic_df = synthetic_df.sample(n=n_samples, random_state=self.random_state)
            
            return synthetic_df
            
        except Exception as e:
            self.logger.error(f"SMOTE synthesis failed: {e}")
            # Fallback to GMM
            self.logger.warning("Falling back to GMM method")
            return self._synthesize_gmm(data, n_samples)
    
    def _synthesize_resample(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using resampling with noise"""
        try:
            # Bootstrap sampling with replacement
            synthetic_df = data.sample(n=n_samples, replace=True, random_state=self.random_state)
            
            # Add small amount of noise to numeric columns
            numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                noise_std = synthetic_df[col].std() * 0.05  # 5% noise
                noise = np.random.normal(0, noise_std, size=len(synthetic_df))
                synthetic_df[col] = synthetic_df[col] + noise
            
            # Reset index
            synthetic_df = synthetic_df.reset_index(drop=True)
            
            return synthetic_df
            
        except Exception as e:
            self.logger.error(f"Resample synthesis failed: {e}")
            raise
    
    def generate_synthetic_data(self, 
                              n_samples: Optional[int] = None,
                              apply_drift: bool = False,
                              drift_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate synthetic data using the specified method
        
        Parameters:
        -----------
        n_samples : Optional[int]
            Number of samples to generate (default: same as original)
        apply_drift : bool
            Whether to apply drift to the synthetic data
        drift_config : Optional[Dict[str, Any]]
            Configuration for drift injection
        
        Returns:
        --------
        pd.DataFrame
            Generated synthetic data
        """
        if n_samples is None:
            n_samples = len(self.original_data)
        
        self.logger.info(f"Generating {n_samples} synthetic samples using {self.synthesis_method}")
        
        # Generate synthetic data based on method
        if self.synthesis_method == 'gmm':
            synthetic_data = self._synthesize_gmm(self.original_data, n_samples)
        elif self.synthesis_method == 'ctgan':
            synthetic_data = self._synthesize_ctgan(self.original_data, n_samples)
        elif self.synthesis_method == 'copula':
            synthetic_data = self._synthesize_copula(self.original_data, n_samples)
        elif self.synthesis_method == 'smote':
            synthetic_data = self._synthesize_smote(self.original_data, n_samples)
        elif self.synthesis_method == 'resample':
            synthetic_data = self._synthesize_resample(self.original_data, n_samples)
        else:
            raise ValueError(f"Unknown synthesis method: {self.synthesis_method}")
        
        # Apply drift if requested
        if apply_drift and drift_config:
            drift_injector = DriftInjector(**drift_config)
            synthetic_data = drift_injector.inject_drift(synthetic_data)
            self.logger.info("Applied drift to synthetic data")
        
        # Generate report and visualizations
        self._generate_report(synthetic_data)
        
        return synthetic_data
    
    def _generate_report(self, synthetic_data: pd.DataFrame):
        """Generate comprehensive report comparing real and synthetic data"""
        if not self.auto_visualize:
            return
        
        try:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output_real_generation_{self.synthesis_method}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate comprehensive report
            self.reporter.generate_comprehensive_report(
                real_df=self.original_data,
                synthetic_df=synthetic_data,
                generator_name=f"RealGenerator_{self.synthesis_method}",
                output_dir=output_dir,
                target_column=self.target_column
            )
            
            self.logger.info(f"Report and visualizations saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the original dataset"""
        return {
            'dataset_path': str(self.dataset_path),
            'shape': self.original_data.shape,
            'columns': list(self.original_data.columns),
            'dtypes': self.original_data.dtypes.to_dict(),
            'missing_values': self.original_data.isnull().sum().to_dict(),
            'synthesis_method': self.synthesis_method,
            'target_column': self.target_column,
            'categorical_columns': self.categorical_columns
        }
    
    def save_synthetic_data(self, 
                          synthetic_data: pd.DataFrame, 
                          output_path: str,
                          format: str = 'csv') -> str:
        """Save synthetic data to file"""
        output_path = Path(output_path)
        
        try:
            if format.lower() == 'csv':
                synthetic_data.to_csv(output_path, index=False)
            elif format.lower() == 'parquet':
                synthetic_data.to_parquet(output_path, index=False)
            elif format.lower() in ['xlsx', 'excel']:
                synthetic_data.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Synthetic data saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save synthetic data: {e}")
            raise