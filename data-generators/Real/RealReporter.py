"""
Enhanced Real Data Reporter with Comprehensive Statistics and Visualizations
Provides detailed comparison between real and synthetic datasets with modern visualizations
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
from datetime import datetime
import os

# SDV imports for quality assessment
try:
    from sdv.evaluation.single_table import evaluate_quality
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    warnings.warn("SDV not available. Quality assessment will be limited.")

# Statistical tests
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score

from ..Visualization.AutoVisualizer import AutoVisualizer


class RealReporter:
    """
    Enhanced Real Data Reporter with comprehensive statistics and modern visualizations
    
    Features:
    - Comprehensive statistical analysis (matching SyntheticReporter)
    - SDV quality assessment for block datasets (complete dataset comparison)
    - Modern visualization with improved styling
    - Dataset-level plots (not per-block)
    - Enhanced reporting capabilities
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize RealReporter
        
        Parameters:
        -----------
        verbose : bool
            Whether to print detailed information
        """
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up matplotlib and seaborn styling
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def generate_comprehensive_report(self,
                                    real_df: pd.DataFrame,
                                    synthetic_df: pd.DataFrame,
                                    generator_name: str,
                                    output_dir: str,
                                    target_column: Optional[str] = None):
        """
        Generate comprehensive report comparing real and synthetic data
        
        Parameters:
        -----------
        real_df : pd.DataFrame
            Original real dataset
        synthetic_df : pd.DataFrame
            Generated synthetic dataset
        generator_name : str
            Name of the generator for labeling
        output_dir : str
            Directory to save outputs
        target_column : Optional[str]
            Name of target column
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE REAL DATA GENERATION REPORT")
        print(f"Generator: {generator_name}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Basic dataset information
        self._report_basic_info(real_df, synthetic_df)
        
        # Statistical analysis
        self._report_statistical_analysis(real_df, synthetic_df, target_column)
        
        # Feature analysis
        self._report_feature_analysis(real_df, synthetic_df)
        
        # Target analysis (if applicable)
        if target_column:
            self._report_target_analysis(real_df, synthetic_df, target_column)
        
        # SDV Quality assessment
        if SDV_AVAILABLE:
            self._report_sdv_quality_assessment(real_df, synthetic_df)
        
        # Generate visualizations using AutoVisualizer
        self._generate_visualizations(real_df, synthetic_df, generator_name, output_dir, target_column)
        
        print(f"\n{'='*80}")
        print(f"REPORT COMPLETE - Outputs saved to: {output_dir}")
        print(f"{'='*80}\n")
    
    def _report_basic_info(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        """Report basic dataset information with pandas-like statistics"""
        print(f"\n📊 DATASET OVERVIEW")
        print(f"{'-'*50}")
        
        # Real dataset info
        print(f"REAL DATASET:")
        print(f"Shape: {real_df.shape}")
        print(f"Size: {real_df.size:,} total elements")
        memory_usage = real_df.memory_usage(deep=True).sum()
        print(f"Memory usage: {memory_usage / 1024:.2f} KB")
        print(f"Columns: {list(real_df.columns)}")
        print(f"Data types: {real_df.dtypes.value_counts().to_dict()}")
        
        # Missing values
        missing_real = real_df.isnull().sum()
        if missing_real.sum() > 0:
            print(f"Missing values: {missing_real[missing_real > 0].to_dict()}")
        else:
            print("Missing values: None")
        
        print(f"\nSYNTHETIC DATASET:")
        print(f"Shape: {synthetic_df.shape}")
        print(f"Size: {synthetic_df.size:,} total elements")
        memory_usage = synthetic_df.memory_usage(deep=True).sum()
        print(f"Memory usage: {memory_usage / 1024:.2f} KB")
        
        # Missing values in synthetic
        missing_synthetic = synthetic_df.isnull().sum()
        if missing_synthetic.sum() > 0:
            print(f"Missing values: {missing_synthetic[missing_synthetic > 0].to_dict()}")
        else:
            print("Missing values: None")
        
        # Duplicate rows
        real_duplicates = real_df.duplicated().sum()
        synthetic_duplicates = synthetic_df.duplicated().sum()
        print(f"\nDuplicate rows:")
        print(f"Real: {real_duplicates} ({real_duplicates/len(real_df)*100:.2f}%)")
        print(f"Synthetic: {synthetic_duplicates} ({synthetic_duplicates/len(synthetic_df)*100:.2f}%)")
    
    def _report_statistical_analysis(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, target_column: Optional[str]):
        """Report comprehensive statistical analysis matching SyntheticReporter"""
        print(f"\n📈 STATISTICAL ANALYSIS")
        print(f"{'-'*50}")
        
        # Identify feature types
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = real_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from features if specified
        if target_column:
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
        
        print(f"Feature types:")
        print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
        print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
        if target_column:
            target_type = "numeric" if target_column in real_df.select_dtypes(include=[np.number]).columns else "categorical"
            print(f"  Target column: {target_column} ({target_type})")
        
        # Numeric features statistics (pandas describe)
        if numeric_cols:
            print(f"\nNUMERIC FEATURES STATISTICS:")
            print(f"Real dataset:")
            print(real_df[numeric_cols].describe())
            print(f"\nSynthetic dataset:")
            print(synthetic_df[numeric_cols].describe())
            
            # Statistical tests for numeric features
            print(f"\nSTATISTICAL TESTS (Numeric Features):")
            print(f"{'Feature':<20} {'KS-Test p-value':<15} {'Mean Diff':<12} {'Std Diff':<12}")
            print(f"{'-'*65}")
            
            for col in numeric_cols:
                if col in synthetic_df.columns:
                    try:
                        # Kolmogorov-Smirnov test
                        ks_stat, ks_p = stats.ks_2samp(real_df[col].dropna(), synthetic_df[col].dropna())
                        
                        # Mean and std differences
                        mean_diff = abs(real_df[col].mean() - synthetic_df[col].mean())
                        std_diff = abs(real_df[col].std() - synthetic_df[col].std())
                        
                        print(f"{col:<20} {ks_p:<15.4f} {mean_diff:<12.4f} {std_diff:<12.4f}")
                    except Exception as e:
                        print(f"{col:<20} {'ERROR':<15} {'ERROR':<12} {'ERROR':<12}")
        
        # Categorical features statistics
        if categorical_cols:
            print(f"\nCATEGORICAL FEATURES STATISTICS:")
            for col in categorical_cols:
                if col in synthetic_df.columns:
                    print(f"\n{col}:")
                    print(f"  Real unique values: {real_df[col].nunique()}")
                    print(f"  Synthetic unique values: {synthetic_df[col].nunique()}")
                    
                    # Top values comparison
                    real_top = real_df[col].value_counts().head(3)
                    synthetic_top = synthetic_df[col].value_counts().head(3)
                    
                    print(f"  Real top values: {real_top.to_dict()}")
                    print(f"  Synthetic top values: {synthetic_top.to_dict()}")
                    
                    # Chi-square test for categorical features
                    try:
                        common_values = set(real_df[col].unique()) & set(synthetic_df[col].unique())
                        if len(common_values) > 1:
                            real_counts = real_df[col].value_counts()
                            synthetic_counts = synthetic_df[col].value_counts()
                            
                            # Align counts
                            aligned_real = []
                            aligned_synthetic = []
                            for val in common_values:
                                aligned_real.append(real_counts.get(val, 0))
                                aligned_synthetic.append(synthetic_counts.get(val, 0))
                            
                            if sum(aligned_real) > 0 and sum(aligned_synthetic) > 0:
                                chi2, chi2_p = stats.chisquare(aligned_synthetic, aligned_real)
                                print(f"  Chi-square p-value: {chi2_p:.4f}")
                    except Exception as e:
                        print(f"  Chi-square test: ERROR")
    
    def _report_feature_analysis(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        """Report feature-level analysis"""
        print(f"\n🔍 FEATURE ANALYSIS")
        print(f"{'-'*50}")
        
        # Feature correlations
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            print(f"CORRELATION ANALYSIS:")
            
            real_corr = real_df[numeric_cols].corr()
            synthetic_corr = synthetic_df[numeric_cols].corr()
            
            # Calculate correlation differences
            corr_diff = abs(real_corr - synthetic_corr)
            avg_corr_diff = corr_diff.mean().mean()
            max_corr_diff = corr_diff.max().max()
            
            print(f"Average correlation difference: {avg_corr_diff:.4f}")
            print(f"Maximum correlation difference: {max_corr_diff:.4f}")
            
            # Top correlation differences
            corr_diff_flat = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)]
            if len(corr_diff_flat) > 0:
                print(f"Mean correlation preservation: {1 - avg_corr_diff:.4f}")
        
        # Feature ranges and distributions
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nFEATURE RANGES:")
            print(f"{'Feature':<20} {'Real Range':<20} {'Synthetic Range':<20} {'Range Similarity':<15}")
            print(f"{'-'*80}")
            
            for col in numeric_cols:
                if col in synthetic_df.columns:
                    real_range = (real_df[col].min(), real_df[col].max())
                    synthetic_range = (synthetic_df[col].min(), synthetic_df[col].max())
                    
                    # Calculate range similarity
                    real_span = real_range[1] - real_range[0]
                    synthetic_span = synthetic_range[1] - synthetic_range[0]
                    
                    if real_span > 0:
                        range_similarity = 1 - abs(real_span - synthetic_span) / real_span
                        range_similarity = max(0, min(1, range_similarity))
                    else:
                        range_similarity = 1.0 if synthetic_span == 0 else 0.0
                    
                    print(f"{col:<20} {str(real_range):<20} {str(synthetic_range):<20} {range_similarity:<15.4f}")
    
    def _report_target_analysis(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, target_column: str):
        """Report target variable analysis"""
        if target_column not in real_df.columns or target_column not in synthetic_df.columns:
            return
        
        print(f"\n🎯 TARGET ANALYSIS")
        print(f"{'-'*50}")
        
        print(f"Target column: {target_column}")
        
        # Check if target is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(real_df[target_column])
        
        if is_numeric:
            print(f"Target type: Numeric")
            print(f"\nTarget statistics:")
            print(f"Real - Mean: {real_df[target_column].mean():.4f}, Std: {real_df[target_column].std():.4f}")
            print(f"Synthetic - Mean: {synthetic_df[target_column].mean():.4f}, Std: {synthetic_df[target_column].std():.4f}")
            
            # Statistical test
            try:
                ks_stat, ks_p = stats.ks_2samp(real_df[target_column].dropna(), synthetic_df[target_column].dropna())
                print(f"KS-test p-value: {ks_p:.4f}")
            except:
                print("KS-test: ERROR")
        
        else:
            print(f"Target type: Categorical")
            print(f"\nTarget distribution:")
            
            real_dist = real_df[target_column].value_counts(normalize=True).sort_index()
            synthetic_dist = synthetic_df[target_column].value_counts(normalize=True).sort_index()
            
            print(f"Real distribution: {real_dist.to_dict()}")
            print(f"Synthetic distribution: {synthetic_dist.to_dict()}")
            
            # Jensen-Shannon divergence for distribution comparison
            try:
                common_labels = sorted(set(real_dist.index) & set(synthetic_dist.index))
                if len(common_labels) > 1:
                    real_probs = [real_dist.get(label, 0) for label in common_labels]
                    synthetic_probs = [synthetic_dist.get(label, 0) for label in common_labels]
                    
                    js_divergence = jensenshannon(real_probs, synthetic_probs)
                    print(f"Jensen-Shannon divergence: {js_divergence:.4f}")
                    print(f"Distribution similarity: {1 - js_divergence:.4f}")
            except Exception as e:
                print("Jensen-Shannon divergence: ERROR")
    
    def _report_sdv_quality_assessment(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        """Report SDV quality assessment - Fixed for block datasets"""
        print(f"\n🏆 SDV QUALITY ASSESSMENT")
        print(f"{'-'*50}")
        
        try:
            # Prepare data - remove block column if present for proper comparison
            common_cols = [col for col in synthetic_df.columns if col in real_df.columns and col != 'block']
            
            if len(common_cols) == 0:
                print("No common columns found for SDV assessment")
                return
            
            # Use only common columns (excluding block column)
            real_for_sdv = real_df[common_cols].copy()
            synthetic_for_sdv = synthetic_df[common_cols].copy()
            
            print(f"Comparing datasets using columns: {common_cols}")
            print(f"Real dataset shape: {real_for_sdv.shape}")
            print(f"Synthetic dataset shape: {synthetic_for_sdv.shape}")
            
            # Create metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_for_sdv)
            
            # Evaluate quality
            quality_report = evaluate_quality(
                real_data=real_for_sdv,
                synthetic_data=synthetic_for_sdv,
                metadata=metadata
            )
            
            print(f"\nQuality Scores:")
            print(f"Overall Quality Score: {quality_report.get_score():.4f}")
            
            # Detailed scores
            properties = quality_report.get_properties()
            for prop_name, prop_score in properties.items():
                print(f"{prop_name}: {prop_score:.4f}")
            
            # Details by column
            details = quality_report.get_details()
            if hasattr(details, 'items'):
                print(f"\nDetailed Analysis:")
                for metric_name, metric_details in details.items():
                    if isinstance(metric_details, dict):
                        print(f"\n{metric_name}:")
                        for col, score in metric_details.items():
                            if isinstance(score, (int, float)):
                                print(f"  {col}: {score:.4f}")
            
        except Exception as e:
            print(f"SDV quality assessment failed: {e}")
            print("This might be due to data format issues or SDV version compatibility")
    
    def _generate_visualizations(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                               generator_name: str, output_dir: str, target_column: Optional[str]):
        """Generate comprehensive visualizations using AutoVisualizer"""
        print(f"\n🎨 GENERATING VISUALIZATIONS")
        print(f"{'-'*50}")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Use AutoVisualizer for comprehensive plotting
            AutoVisualizer.generate_real_vs_synthetic_plots(
                real_df=real_df,
                synthetic_df=synthetic_df,
                generator_name=generator_name,
                output_dir=output_dir,
                target_column=target_column
            )
            
            print(f"Visualizations saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")
            print(f"Error generating visualizations: {e}")
    
    def compare_datasets_summary(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary comparison of datasets"""
        summary = {
            'shapes': {
                'real': real_df.shape,
                'synthetic': synthetic_df.shape
            },
            'missing_values': {
                'real': real_df.isnull().sum().sum(),
                'synthetic': synthetic_df.isnull().sum().sum()
            },
            'duplicates': {
                'real': real_df.duplicated().sum(),
                'synthetic': synthetic_df.duplicated().sum()
            },
            'dtypes_match': real_df.dtypes.equals(synthetic_df.dtypes),
            'columns_match': set(real_df.columns) == set(synthetic_df.columns)
        }
        
        # Numeric feature comparison
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_similarity'] = {}
            for col in numeric_cols:
                if col in synthetic_df.columns:
                    try:
                        # Simple correlation as similarity measure
                        combined_df = pd.DataFrame({
                            'real': real_df[col],
                            'synthetic': synthetic_df[col][:len(real_df)]  # Match lengths
                        }).dropna()
                        
                        if len(combined_df) > 1:
                            corr = combined_df['real'].corr(combined_df['synthetic'])
                            summary['numeric_similarity'][col] = corr if not pd.isna(corr) else 0.0
                    except:
                        summary['numeric_similarity'][col] = 0.0
        
        return summary