#!/usr/bin/env python3
"""
Real Data Visualizer for CALMOPS
=================================

Specialized visualization system for real data analysis and comparison.
Focuses on real vs synthetic data comparison, quality assessment, and drift analysis.

Author: CalmOps Team
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Matplotlib fallback for PNG export
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

# Suppress all common visualization warnings
plt.rcParams['axes.formatter.useoffset'] = False
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings('ignore', message='.*Using categorical units.*')
warnings.filterwarnings('ignore', message='.*strings that are all parsable.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='plotly')
warnings.filterwarnings('ignore', module='seaborn')

class RealVisualizer:
    """
    Specialized visualization system for real data comparison and analysis
    """

    @staticmethod
    def auto_analyze_and_visualize(data: List[Tuple], generator_name: str, 
                                  output_dir: str = "outputs",
                                  enable_plotly: bool = True) -> Dict[str, Any]:
        """
        Complete automatic analysis and visualization for real data comparison
        
        Args:
            data: List of (features_dict, target) tuples
            generator_name: Name for file outputs
            output_dir: Directory for outputs
            enable_plotly: Whether to generate interactive Plotly visualizations
            
        Returns:
            Dictionary with analysis results and visualization file paths
        """
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        df = RealVisualizer._convert_to_dataframe(data)
        
        # Get feature columns (excluding target)
        feature_cols = [col for col in df.columns if col != 'target']
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        # Calculate quality metrics
        quality_results = RealVisualizer._calculate_real_quality_score(df, feature_cols)
        
        # Create visualizations
        visualization_files = {}
        
        try:
            if enable_plotly:
                # Create Plotly visualizations
                visualization_files.update(
                    RealVisualizer._create_real_plotly_visualizations(
                        df, feature_cols, numeric_features, quality_results, 
                        generator_name, output_dir
                    )
                )
            else:
                # Create matplotlib visualizations
                visualization_files.update(
                    RealVisualizer._create_real_matplotlib_visualizations(
                        df, feature_cols, numeric_features, quality_results,
                        generator_name, output_dir
                    )
                )
            
            # Perform real data drift analysis
            drift_results = RealVisualizer._analyze_real_data_drift(df, numeric_features)
            
            # Block analysis if block column exists
            block_results = None
            if any(col in df.columns for col in ['block', 'chunk', 'Block', 'Chunk']):
                block_results = RealVisualizer._analyze_real_block_structure(df, output_dir, generator_name)
            
        except Exception as e:
            print(f"Real visualization error: {e}")
            visualization_files = {"error": str(e)}
        
        return {
            'quality_score': quality_results,
            'visualization_files': visualization_files,
            'drift_analysis': drift_results,
            'block_analysis': block_results,
            'dataset_info': {
                'samples': len(df),
                'features': len(feature_cols),
                'numeric_features': len(numeric_features),
                'data_type': 'real_comparison'
            }
        }

    @staticmethod
    def _convert_to_dataframe(data: List[Tuple]) -> pd.DataFrame:
        """Convert list of tuples to DataFrame"""
        records = []
        for features_dict, target in data:
            record = features_dict.copy()
            record['target'] = target
            records.append(record)
        return pd.DataFrame(records)

    @staticmethod
    def _calculate_real_quality_score(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Calculate real data quality metrics with focus on data authenticity"""
        
        # Data completeness
        completeness_ratio = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        data_completeness = completeness_ratio * 100
        
        # Feature coverage
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        feature_coverage = min(100, (len(numeric_cols) / max(1, len(feature_cols))) * 100)
        
        # Data consistency (check for reasonable ranges)
        data_consistency = 90  # Base score
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Check for extreme outliers
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = col_data[(col_data < q1 - 3*iqr) | (col_data > q3 + 3*iqr)]
                    outlier_ratio = len(outliers) / len(col_data)
                    if outlier_ratio > 0.1:  # More than 10% outliers
                        data_consistency -= 10
        
        # Target distribution quality
        target_quality = 100
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            # Check for extreme imbalance
            if len(target_counts) > 1:
                imbalance_ratio = target_counts.max() / target_counts.min()
                if imbalance_ratio > 10:  # Very imbalanced
                    target_quality = 60
                elif imbalance_ratio > 5:
                    target_quality = 80
        
        # Overall real data quality score
        overall_score = (data_completeness + feature_coverage + data_consistency + target_quality) / 4
        
        return {
            'overall_score': round(overall_score, 1),
            'data_completeness': round(data_completeness, 1),
            'feature_coverage': round(feature_coverage, 1),
            'data_consistency': round(data_consistency, 1),
            'target_quality': round(target_quality, 1),
            'details': {
                'total_samples': len(df),
                'total_features': len(feature_cols),
                'missing_values': df.isnull().sum().sum(),
                'numeric_features': len(numeric_cols),
                'outlier_analysis': 'completed'
            }
        }

    @staticmethod
    def _create_real_plotly_visualizations(df: pd.DataFrame, feature_cols: List[str],
                                         numeric_features: List[str], quality_results: Dict,
                                         generator_name: str, output_dir: str) -> Dict[str, str]:
        """Create Plotly visualizations specific to real data analysis"""
        
        viz_files = {}
        
        # Real data quality breakdown
        quality_file = RealVisualizer._plot_real_quality_breakdown_plotly(
            quality_results, generator_name, output_dir
        )
        if quality_file:
            viz_files['real_quality_breakdown'] = quality_file
        
        # Feature authenticity analysis
        for feature in feature_cols[:5]:  # Limit to first 5 features
            if feature in numeric_features:
                auth_file = RealVisualizer._plot_feature_authenticity_plotly(
                    df, feature, generator_name, output_dir
                )
                if auth_file:
                    viz_files[f'authenticity_{feature}'] = auth_file
        
        # Data integrity heatmap
        if len(numeric_features) > 1:
            integrity_file = RealVisualizer._plot_data_integrity_heatmap_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if integrity_file:
                viz_files['data_integrity'] = integrity_file
        
        # Real vs synthetic comparison (if both datasets available)
        comparison_file = RealVisualizer._plot_real_synthetic_comparison_plotly(
            df, numeric_features, generator_name, output_dir
        )
        if comparison_file:
            viz_files['real_synthetic_comparison'] = comparison_file
        
        # Statistical validation
        if len(numeric_features) > 0:
            validation_file = RealVisualizer._plot_statistical_validation_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if validation_file:
                viz_files['statistical_validation'] = validation_file
        
        # NUEVOS PLOTS EXPLORATORIOS
        
        # Feature distributions (histogramas individuales)
        for feature in numeric_features[:6]:  # Primeras 6 features
            dist_file = RealVisualizer._plot_feature_distribution_exploratory_plotly(
                df, feature, generator_name, output_dir
            )
            if dist_file:
                viz_files[f'distribution_exploratory_{feature}'] = dist_file
        
        # Box plots summary
        if len(numeric_features) > 0:
            boxplot_file = RealVisualizer._plot_boxplots_summary_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if boxplot_file:
                viz_files['boxplots_summary'] = boxplot_file
        
        # Correlation matrix exploratory
        if len(numeric_features) > 1:
            corr_exp_file = RealVisualizer._plot_correlation_exploratory_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if corr_exp_file:
                viz_files['correlation_exploratory'] = corr_exp_file
        
        # PCA exploratory
        if len(numeric_features) >= 2:
            pca_exp_file = RealVisualizer._plot_pca_exploratory_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if pca_exp_file:
                viz_files['pca_exploratory'] = pca_exp_file
        
        # Outlier analysis
        if len(numeric_features) > 0:
            outlier_file = RealVisualizer._plot_outlier_analysis_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if outlier_file:
                viz_files['outlier_analysis'] = outlier_file
        
        # Target analysis (si existe target)
        if 'target' in df.columns:
            target_file = RealVisualizer._plot_target_analysis_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if target_file:
                viz_files['target_analysis'] = target_file
        
        # PLOTS ADICIONALES DE SYNTHETICVISUALIZER
        
        # Histogramas detallados (del SyntheticVisualizer)
        for feature in numeric_features[:4]:  # Primeras 4 features
            histogram_detailed_file = RealVisualizer._plot_histogram_detailed_synth_style_plotly(
                df, feature, generator_name, output_dir
            )
            if histogram_detailed_file:
                viz_files[f'histogram_detailed_synth_{feature}'] = histogram_detailed_file
        
        # Violin plots (del SyntheticVisualizer)
        if len(numeric_features) > 0:
            violin_synth_file = RealVisualizer._plot_violin_plots_synth_style_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if violin_synth_file:
                viz_files['violin_plots_synth'] = violin_synth_file
        
        # Pairwise distributions (del SyntheticVisualizer)
        if len(numeric_features) >= 2:
            pairwise_synth_file = RealVisualizer._plot_pairwise_distributions_synth_style_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if pairwise_synth_file:
                viz_files['pairwise_distributions_synth'] = pairwise_synth_file
        
        # Target categorical analysis (del SyntheticVisualizer)
        if 'target' in df.columns:
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'target' in categorical_features:
                categorical_features.remove('target')
                
            target_categorical_synth_file = RealVisualizer._plot_target_categorical_synth_style_plotly(
                df, categorical_features, generator_name, output_dir
            )
            if target_categorical_synth_file:
                viz_files['target_categorical_synth'] = target_categorical_synth_file
        
        return viz_files

    @staticmethod
    def _create_real_matplotlib_visualizations(df: pd.DataFrame, feature_cols: List[str],
                                             numeric_features: List[str], quality_results: Dict,
                                             generator_name: str, output_dir: str) -> Dict[str, str]:
        """Create matplotlib visualizations for real data"""
        
        viz_files = {}
        
        try:
            plt.style.use('default')
            
            # Create comprehensive real data analysis
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Real Data Analysis - {generator_name}', fontsize=16)
            
            # Quality breakdown
            ax = axes[0, 0]
            metrics = ['Completeness', 'Coverage', 'Consistency', 'Target Quality']
            values = [quality_results['data_completeness'], quality_results['feature_coverage'],
                     quality_results['data_consistency'], quality_results['target_quality']]
            bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylim(0, 100)
            ax.set_title('Real Data Quality Metrics')
            ax.set_ylabel('Score')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}', ha='center', va='bottom')
            
            # Feature distributions (first 4 features)
            for i, feature in enumerate(feature_cols[:4]):
                if i < 4 and feature in numeric_features:
                    row, col = (0, 1) if i < 2 else (1, 0), (i % 2 + 1) if i < 2 else (i % 2)
                    if row == 0 and col == 2:
                        continue  # Skip this position
                    
                    ax = axes[row, col + (1 if row == 0 else 0)]
                    data = df[feature].dropna()
                    ax.hist(data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                    ax.set_title(f'{feature} Distribution')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Frequency')
                    
                    # Add statistical info
                    mean_val = data.mean()
                    std_val = data.std()
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    ax.legend()
            
            # Correlation analysis
            if len(numeric_features) > 1:
                ax = axes[0, 2]
                corr_matrix = df[numeric_features].corr()
                im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric_features)))
                ax.set_yticks(range(len(numeric_features)))
                ax.set_xticklabels([f[:8] for f in numeric_features], rotation=45, ha='right')
                ax.set_yticklabels([f[:8] for f in numeric_features])
                ax.set_title('Feature Correlations')
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Target distribution
            if 'target' in df.columns:
                ax = axes[1, 2]
                target_counts = df['target'].value_counts()
                colors = plt.cm.Set3(np.linspace(0, 1, len(target_counts)))
                wedges, texts, autotexts = ax.pie(target_counts.values,
                                                 labels=[f'Class {label}' for label in target_counts.index],
                                                 autopct='%1.1f%%',
                                                 colors=colors)
                ax.set_title('Target Distribution')
            
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(output_dir, f'{generator_name}_real_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['comprehensive_real_analysis'] = output_file
            
        except Exception as e:
            print(f"Real matplotlib visualization error: {e}")
        
        return viz_files

    @staticmethod
    def _analyze_real_data_drift(df: pd.DataFrame, numeric_features: List[str]) -> Dict[str, Any]:
        """Analyze drift patterns in real data"""
        
        drift_results = {
            'data_stability': {},
            'feature_consistency': {},
            'drift_indicators': [],
            'overall_drift_score': 0.0
        }
        
        if len(numeric_features) == 0:
            return drift_results
        
        # Analyze feature stability
        try:
            for feature in numeric_features[:5]:  # Check first 5 features
                data = df[feature].dropna()
                if len(data) > 10:
                    # Check for distribution normality
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
                        
                        # Check for outliers
                        q1, q3 = data.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
                        outlier_ratio = len(outliers) / len(data)
                        
                        # Stability metrics
                        cv = data.std() / abs(data.mean()) if data.mean() != 0 else float('inf')
                        
                        drift_results['data_stability'][feature] = {
                            'normality_p_value': shapiro_p,
                            'outlier_ratio': outlier_ratio,
                            'coefficient_variation': cv,
                            'is_stable': outlier_ratio < 0.05 and cv < 1.0
                        }
                        
                        # Flag potential drift indicators
                        if outlier_ratio > 0.1:
                            drift_results['drift_indicators'].append(f"{feature}: High outlier ratio ({outlier_ratio:.1%})")
                        
                        if cv > 2.0:
                            drift_results['drift_indicators'].append(f"{feature}: High variability (CV={cv:.2f})")
                            
                    except Exception as e:
                        drift_results['data_stability'][feature] = {'error': str(e)}
        
        except Exception as e:
            drift_results['error'] = str(e)
        
        # Calculate overall drift score
        stable_features = sum(1 for f in drift_results['data_stability'].values() 
                            if isinstance(f, dict) and f.get('is_stable', False))
        total_features = len(drift_results['data_stability'])
        if total_features > 0:
            drift_results['overall_drift_score'] = 1 - (stable_features / total_features)
        
        return drift_results

    @staticmethod
    def _analyze_real_block_structure(df: pd.DataFrame, output_dir: str, generator_name: str) -> Dict[str, Any]:
        """Analyze block structure in real data"""
        
        # Find block column
        block_col = None
        for col_name in ['block', 'chunk', 'Block', 'Chunk']:
            if col_name in df.columns:
                block_col = col_name
                break
        
        if not block_col:
            return None
        
        unique_blocks = sorted(df[block_col].unique())
        block_results = {
            'block_column': block_col,
            'num_blocks': len(unique_blocks),
            'block_sizes': {},
            'block_quality': {},
            'inter_block_consistency': []
        }
        
        # Analyze each block for real data quality
        for block_id in unique_blocks:
            block_data = df[df[block_col] == block_id]
            
            # Block size
            block_results['block_sizes'][str(block_id)] = len(block_data)
            
            # Block quality metrics
            missing_ratio = block_data.isnull().sum().sum() / (len(block_data) * len(block_data.columns))
            duplicate_ratio = block_data.duplicated().sum() / len(block_data)
            
            block_results['block_quality'][str(block_id)] = {
                'missing_ratio': missing_ratio,
                'duplicate_ratio': duplicate_ratio,
                'quality_score': max(0, 100 - missing_ratio*100 - duplicate_ratio*50)
            }
        
        # Check consistency between blocks
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        if block_col in numeric_features:
            numeric_features.remove(block_col)
        
        if len(unique_blocks) > 1 and len(numeric_features) > 0:
            for i in range(len(unique_blocks) - 1):
                current_block = unique_blocks[i]
                next_block = unique_blocks[i + 1]
                
                current_data = df[df[block_col] == current_block]
                next_data = df[df[block_col] == next_block]
                
                # Check feature consistency between blocks
                consistency_score = 0
                feature_count = 0
                
                for feature in numeric_features[:3]:  # Check first 3 numeric features
                    if feature in current_data.columns and feature in next_data.columns:
                        try:
                            # Compare means
                            current_mean = current_data[feature].mean()
                            next_mean = next_data[feature].mean()
                            
                            # Calculate relative difference
                            if current_mean != 0:
                                rel_diff = abs(next_mean - current_mean) / abs(current_mean)
                                consistency_score += max(0, 1 - rel_diff)
                            else:
                                consistency_score += 1 if next_mean == 0 else 0
                            
                            feature_count += 1
                        except:
                            pass
                
                if feature_count > 0:
                    avg_consistency = consistency_score / feature_count
                    block_results['inter_block_consistency'].append({
                        'from_block': current_block,
                        'to_block': next_block,
                        'consistency_score': avg_consistency,
                        'is_consistent': avg_consistency > 0.8
                    })
        
        return block_results

    # Plotly visualization methods for real data
    @staticmethod
    def _plot_real_quality_breakdown_plotly(quality_results: Dict, generator_name: str, 
                                          output_dir: str) -> Optional[str]:
        """Create interactive real data quality breakdown"""
        try:
            metrics = ['Data Completeness', 'Feature Coverage', 'Data Consistency', 'Target Quality']
            values = [quality_results['data_completeness'], quality_results['feature_coverage'],
                     quality_results['data_consistency'], quality_results['target_quality']]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            fig = go.Figure(data=[
                go.Bar(x=metrics, y=values, marker_color=colors,
                      text=[f'{v:.1f}' for v in values],
                      textposition='auto')
            ])
            
            fig.update_layout(
                title=f'Real Data Quality Assessment - {generator_name}',
                xaxis_title='Quality Dimensions',
                yaxis_title='Score (0-100)',
                yaxis=dict(range=[0, 100]),
                template='plotly_white'
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_real_quality', output_dir)
            
        except Exception as e:
            print(f"Real quality plot error: {e}")
            return None

    @staticmethod
    def _plot_feature_authenticity_plotly(df: pd.DataFrame, feature: str, generator_name: str,
                                        output_dir: str) -> Optional[str]:
        """Create feature authenticity analysis plot"""
        try:
            data = df[feature].dropna()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Distribution', 'Outlier Analysis', 'Normality Check', 'Statistics'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter'}, {'type': 'scatter'}]]
            )
            
            # Distribution
            fig.add_trace(
                go.Histogram(x=data, name='Distribution', showlegend=False, marker_color='lightcoral'),
                row=1, col=1
            )
            
            # Outlier analysis (box plot)
            fig.add_trace(
                go.Box(y=data, name='Outliers', showlegend=False, marker_color='orange'),
                row=1, col=2
            )
            
            # Normality check (Q-Q plot)
            try:
                qq_data = stats.probplot(data, dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                             mode='markers', name='Q-Q', showlegend=False, marker_color='blue'),
                    row=2, col=1
                )
                # Reference line
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][0],
                             mode='lines', name='Normal', showlegend=False,
                             line=dict(color='red', dash='dash')),
                    row=2, col=1
                )
            except:
                pass
            
         
            
            fig.update_layout(
                title=f'Feature Authenticity Analysis: {feature} - {generator_name}',
                template='plotly_white',
                height=600
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_{feature}_authenticity', output_dir)
            
        except Exception as e:
            print(f"Feature authenticity plot error: {e}")
            return None

    @staticmethod
    def _plot_data_integrity_heatmap_plotly(df: pd.DataFrame, numeric_features: List[str],
                                          generator_name: str, output_dir: str) -> Optional[str]:
        """Create data integrity heatmap"""
        try:
            # Calculate integrity metrics
            integrity_matrix = np.zeros((len(numeric_features), len(numeric_features)))
            
            for i, feat1 in enumerate(numeric_features):
                for j, feat2 in enumerate(numeric_features):
                    if i == j:
                        # Self-integrity: based on missing values and outliers
                        data = df[feat1].dropna()
                        q1, q3 = data.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
                        integrity_score = 1 - (len(outliers) / len(data))
                        integrity_matrix[i, j] = integrity_score
                    else:
                        # Cross-integrity: correlation strength
                        try:
                            corr = abs(df[feat1].corr(df[feat2]))
                            integrity_matrix[i, j] = corr if not np.isnan(corr) else 0
                        except:
                            integrity_matrix[i, j] = 0
            
            fig = go.Figure(data=go.Heatmap(
                z=integrity_matrix,
                x=numeric_features,
                y=numeric_features,
                colorscale='Viridis',
                text=np.round(integrity_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f'Data Integrity Matrix - {generator_name}',
                template='plotly_white',
                width=max(500, len(numeric_features) * 50),
                height=max(500, len(numeric_features) * 50)
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_integrity', output_dir)
            
        except Exception as e:
            print(f"Integrity heatmap error: {e}")
            return None

    @staticmethod
    def _plot_real_synthetic_comparison_plotly(df: pd.DataFrame, numeric_features: List[str],
                                             generator_name: str, output_dir: str) -> Optional[str]:
        """Create real vs synthetic comparison plot"""
        try:
            if len(numeric_features) < 2:
                return None
            
            # Create comparison scatter plots for first few features
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f'{feat} Analysis' for feat in numeric_features[:4]]
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for i, feature in enumerate(numeric_features[:4]):
                if i < len(positions):
                    row, col = positions[i]
                    data = df[feature].dropna()
                    
                    # Histogram
                    fig.add_trace(
                        go.Histogram(x=data, name=f'{feature}', showlegend=False,
                                   marker_color='lightblue', opacity=0.7),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=f'Real Data Feature Analysis - {generator_name}',
                template='plotly_white',
                height=600
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_real_comparison', output_dir)
            
        except Exception as e:
            print(f"Real comparison plot error: {e}")
            return None

    @staticmethod
    def _plot_statistical_validation_plotly(df: pd.DataFrame, numeric_features: List[str],
                                           generator_name: str, output_dir: str) -> Optional[str]:
        """Create statistical validation plot"""
        try:
            # Calculate validation metrics
            validation_data = []
            for feature in numeric_features[:6]:  # Limit to 6 features
                data = df[feature].dropna()
                if len(data) > 0:
                    validation_data.append({
                        'Feature': feature,
                        'Mean': data.mean(),
                        'Std': data.std(),
                        'Skewness': abs(data.skew()),
                        'Kurtosis': abs(data.kurtosis()),
                        'CV': data.std() / abs(data.mean()) if data.mean() != 0 else 0
                    })
            
            if not validation_data:
                return None
            
            validation_df = pd.DataFrame(validation_data)
            
            # Create radar chart for validation metrics
            fig = go.Figure()
            
            for i, row in validation_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[min(row['CV'], 2), row['Skewness'], row['Kurtosis'], 
                       min(row['Std'], 10), min(abs(row['Mean']), 10)],
                    theta=['CV', 'Skewness', 'Kurtosis', 'Std', 'Mean'],
                    fill='toself',
                    name=row['Feature']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                title=f'Statistical Validation - {generator_name}',
                template='plotly_white'
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_validation', output_dir)
            
        except Exception as e:
            print(f"Statistical validation plot error: {e}")
            return None

    @staticmethod
    def _save_plotly_file(fig, base_filename: str, output_dir: str) -> Optional[str]:
        """Save Plotly figure as HTML with PNG fallback"""
        try:
            html_file = os.path.join(output_dir, f'{base_filename}.html')
            fig.write_html(html_file)
            
            # Try to create PNG
            try:
                png_file = os.path.join(output_dir, f'{base_filename}.png')
                fig.write_image(png_file, width=800, height=600)
            except:
                # Create matplotlib fallback
                RealVisualizer._create_matplotlib_fallback(fig, base_filename, output_dir)
            
            return html_file
            
        except Exception as e:
            print(f"Real Plotly save error: {e}")
            return None

    @staticmethod
    def _create_matplotlib_fallback(plotly_fig, base_filename: str, output_dir: str) -> Optional[str]:
        """Create matplotlib fallback for plotly figures"""
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Interactive real data visualization:\n{base_filename}.html', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            plt.title(f'Real Data Analysis: {base_filename}')
            plt.axis('off')
            
            png_file = os.path.join(output_dir, f'{base_filename}.png')
            plt.savefig(png_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return png_file
            
        except Exception as e:
            print(f"Real matplotlib fallback error: {e}")
            return None

    # NUEVOS MÉTODOS EXPLORATORIOS
    
    @staticmethod
    def _plot_feature_distribution_exploratory_plotly(df: pd.DataFrame, feature: str, 
                                                     generator_name: str, output_dir: str) -> Optional[str]:
        """Histograma exploratorio detallado de una feature"""
        try:
            fig = go.Figure()
            
            data = df[feature].dropna()
            
            # Histograma principal
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=40,
                name=feature,
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Líneas estadísticas
            mean_val = data.mean()
            median_val = data.median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.3f}")
            fig.add_vline(x=median_val, line_dash="dot", line_color="green", 
                         annotation_text=f"Median: {median_val:.3f}")
            
            fig.update_layout(
                title=f"Exploratory Distribution: {feature}",
                xaxis_title=feature,
                yaxis_title="Frequency",
                template='plotly_white',
                height=400
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_dist_exp_{feature}', output_dir)
            
        except Exception as e:
            print(f"Distribution exploratory error: {e}")
            return None

    @staticmethod
    def _plot_boxplots_summary_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                    generator_name: str, output_dir: str) -> Optional[str]:
        """Box plots de todas las features numéricas"""
        try:
            fig = go.Figure()
            
            for feature in numeric_features:
                fig.add_trace(go.Box(
                    y=df[feature].dropna(),
                    name=feature,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title="Box Plots Summary - All Numeric Features",
                yaxis_title="Value",
                template='plotly_white',
                height=500
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_boxplots', output_dir)
            
        except Exception as e:
            print(f"Box plots error: {e}")
            return None

    @staticmethod
    def _plot_correlation_exploratory_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                           generator_name: str, output_dir: str) -> Optional[str]:
        """Matriz de correlación exploratoria"""
        try:
            corr_matrix = df[numeric_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Exploratory Correlation Matrix",
                template='plotly_white',
                height=500
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_corr_exp', output_dir)
            
        except Exception as e:
            print(f"Correlation exploratory error: {e}")
            return None

    @staticmethod
    def _plot_pca_exploratory_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                   generator_name: str, output_dir: str) -> Optional[str]:
        """PCA exploratorio con análisis de componentes"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            X = df[numeric_features].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['PCA 2D Projection', 'Explained Variance']
            )
            
            # PCA 2D
            if 'target' in df.columns:
                target_data = df.loc[X.index, 'target']
                unique_targets = target_data.unique()
                
                for target in unique_targets:
                    mask = target_data == target
                    fig.add_trace(go.Scatter(
                        x=X_pca[mask, 0],
                        y=X_pca[mask, 1],
                        mode='markers',
                        name=f'Target {target}',
                        showlegend=True
                    ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    mode='markers',
                    name='Data Points'
                ), row=1, col=1)
            
            # Explained variance
            fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                y=pca.explained_variance_ratio_,
                name='Explained Variance',
                showlegend=False
            ), row=1, col=2)
            
            fig.update_layout(
                title="PCA Exploratory Analysis",
                template='plotly_white',
                height=500
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_pca_exp', output_dir)
            
        except Exception as e:
            print(f"PCA exploratory error: {e}")
            return None

    @staticmethod
    def _plot_outlier_analysis_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                    generator_name: str, output_dir: str) -> Optional[str]:
        """Análisis de outliers usando IQR"""
        try:
            outlier_counts = []
            feature_names = []
            
            for feature in numeric_features:
                data = df[feature].dropna()
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(data)) * 100
                
                outlier_counts.append(outlier_percentage)
                feature_names.append(feature)
            
            fig = go.Figure(data=[
                go.Bar(x=feature_names, y=outlier_counts, 
                      marker_color='orange',
                      text=[f'{x:.1f}%' for x in outlier_counts],
                      textposition='auto')
            ])
            
            fig.update_layout(
                title="Outlier Analysis (% of outliers per feature)",
                xaxis_title="Features",
                yaxis_title="Outlier Percentage (%)",
                template='plotly_white',
                height=400
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_outliers', output_dir)
            
        except Exception as e:
            print(f"Outlier analysis error: {e}")
            return None

    @staticmethod
    def _plot_target_analysis_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                   generator_name: str, output_dir: str) -> Optional[str]:
        """Análisis del target y su relación con features"""
        try:
            if 'target' not in df.columns:
                return None
                
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Target Distribution', 'Target vs Feature Means', 
                               'Target Balance', 'Feature Importance (approx)']
            )
            
            # Target distribution
            target_counts = df['target'].value_counts()
            fig.add_trace(go.Bar(
                x=[f'Class {x}' for x in target_counts.index],
                y=target_counts.values,
                name='Target Distribution',
                showlegend=False
            ), row=1, col=1)
            
            # Target vs feature means
            if len(numeric_features) > 0:
                means_by_target = []
                feature_subset = numeric_features[:5]  # Primeras 5 features
                
                for target_val in sorted(df['target'].unique()):
                    target_data = df[df['target'] == target_val]
                    means = [target_data[feat].mean() for feat in feature_subset]
                    
                    fig.add_trace(go.Bar(
                        x=feature_subset,
                        y=means,
                        name=f'Target {target_val}',
                        showlegend=True
                    ), row=1, col=2)
            
            # Target balance pie
            fig.add_trace(go.Pie(
                labels=[f'Class {x}' for x in target_counts.index],
                values=target_counts.values,
                name='Balance',
                showlegend=False
            ), row=2, col=1)
            
            # Feature importance approximation (correlation with target)
            if len(numeric_features) > 0:
                correlations = []
                for feat in numeric_features[:8]:  # Primeras 8 features
                    try:
                        corr = abs(df[feat].corr(df['target']))
                        correlations.append(corr if not np.isnan(corr) else 0)
                    except:
                        correlations.append(0)
                
                fig.add_trace(go.Bar(
                    x=numeric_features[:8],
                    y=correlations,
                    name='Abs Correlation',
                    showlegend=False,
                    marker_color='green'
                ), row=2, col=2)
            
            fig.update_layout(
                title="Target Analysis",
                template='plotly_white',
                height=700
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_target_analysis', output_dir)
            
        except Exception as e:
            print(f"Target analysis error: {e}")
            return None

    # NUEVOS MÉTODOS ESTILO SYNTHETICVISUALIZER
    
    @staticmethod
    def _plot_histogram_detailed_synth_style_plotly(df: pd.DataFrame, feature: str, 
                                                   generator_name: str, output_dir: str) -> Optional[str]:
        """Histograma detallado estilo SyntheticVisualizer con mean/median/mode"""
        try:
            data = df[feature].dropna()
            
            fig = go.Figure()
            
            # Histograma principal
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=50,
                name=feature,
                marker_color='lightgreen',
                opacity=0.7,
                hovertemplate=f'<b>{feature}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
            
            # Estadísticas (estilo SyntheticVisualizer)
            mean_val = data.mean()
            median_val = data.median()
            mode_val = data.mode().iloc[0] if not data.mode().empty else mean_val
            std_val = data.std()
            
            # Líneas estadísticas
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.3f}")
            fig.add_vline(x=median_val, line_dash="dot", line_color="green", 
                         annotation_text=f"Median: {median_val:.3f}")
            fig.add_vline(x=mode_val, line_dash="dashdot", line_color="purple", 
                         annotation_text=f"Mode: {mode_val:.3f}")
            
            # Área de desviación estándar (estilo SyntheticVisualizer)
            fig.add_vrect(
                x0=mean_val - std_val, x1=mean_val + std_val,
                fillcolor="yellow", opacity=0.2,
                annotation_text=f"±1σ", annotation_position="top left"
            )
            
            fig.update_layout(
                title=f"Real Data Detailed Histogram: {feature} (μ={mean_val:.3f}, σ={std_val:.3f})",
                xaxis_title=feature,
                yaxis_title="Frequency",
                template='plotly_white',
                height=450
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_hist_synth_{feature}', output_dir)
            
        except Exception as e:
            print(f"Histogram synth style error: {e}")
            return None

    @staticmethod
    def _plot_violin_plots_synth_style_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                            generator_name: str, output_dir: str) -> Optional[str]:
        """Violin plots estilo SyntheticVisualizer"""
        try:
            fig = go.Figure()
            
            for feature in numeric_features:
                data = df[feature].dropna()
                
                fig.add_trace(go.Violin(
                    y=data,
                    name=feature,
                    box_visible=True,
                    meanline_visible=True,
                    hovertemplate=f'<b>{feature}</b><br>Value: %{{y:.3f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Real Data Violin Plots (SyntheticVisualizer Style)",
                yaxis_title="Value",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_violin_synth', output_dir)
            
        except Exception as e:
            print(f"Violin synth style error: {e}")
            return None

    @staticmethod
    def _plot_pairwise_distributions_synth_style_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                                      generator_name: str, output_dir: str) -> Optional[str]:
        """Pairwise distributions estilo SyntheticVisualizer"""
        try:
            features_subset = numeric_features[:4]  # Primeras 4 features
            n_features = len(features_subset)
            
            if n_features < 2:
                return None
            
            # Crear scatter matrix estilo SyntheticVisualizer
            fig = make_subplots(
                rows=n_features, cols=n_features,
                subplot_titles=[f'{f1} vs {f2}' for f1 in features_subset for f2 in features_subset]
            )
            
            for i, feat1 in enumerate(features_subset):
                for j, feat2 in enumerate(features_subset):
                    row, col = i + 1, j + 1
                    
                    if i == j:
                        # Diagonal: histograma
                        fig.add_trace(go.Histogram(
                            x=df[feat1].dropna(),
                            name=f'{feat1}_hist',
                            showlegend=False,
                            marker_color='lightgreen'
                        ), row=row, col=col)
                    else:
                        # Off-diagonal: scatter plot
                        if 'target' in df.columns:
                            # Colorear por target (estilo SyntheticVisualizer)
                            fig.add_trace(go.Scatter(
                                x=df[feat2],
                                y=df[feat1],
                                mode='markers',
                                name=f'{feat1}_vs_{feat2}',
                                showlegend=False,
                                marker=dict(
                                    color=df['target'],
                                    colorscale='Viridis',
                                    size=4,
                                    opacity=0.6
                                )
                            ), row=row, col=col)
                        else:
                            fig.add_trace(go.Scatter(
                                x=df[feat2],
                                y=df[feat1],
                                mode='markers',
                                name=f'{feat1}_vs_{feat2}',
                                showlegend=False,
                                marker=dict(size=4, opacity=0.6, color='lightgreen')
                            ), row=row, col=col)
            
            fig.update_layout(
                title="Real Data Pairwise Distributions (SyntheticVisualizer Style)",
                template='plotly_white',
                height=200 * n_features
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_pairwise_synth', output_dir)
            
        except Exception as e:
            print(f"Pairwise synth style error: {e}")
            return None

    @staticmethod
    def _plot_target_categorical_synth_style_plotly(df: pd.DataFrame, categorical_features: List[str], 
                                                  generator_name: str, output_dir: str) -> Optional[str]:
        """Target categorical analysis estilo SyntheticVisualizer"""
        try:
            if 'target' not in df.columns:
                return None
                
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Target Distribution', 'Target Mode Analysis', 
                               'Target vs Categorical', 'Target Entropy'],
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Target distribution
            target_counts = df['target'].value_counts()
            mode_target = target_counts.index[0]  # Moda del target
            
            fig.add_trace(go.Bar(
                x=[f'Class {x}' for x in target_counts.index],
                y=target_counts.values,
                name='Target Distribution',
                showlegend=False,
                marker_color='lightgreen',
                text=[f'Mode: {mode_target}' if i == 0 else '' for i in range(len(target_counts))],
                textposition='auto'
            ), row=1, col=1)
            
            # Target pie chart
            fig.add_trace(go.Pie(
                labels=[f'Class {x}' for x in target_counts.index],
                values=target_counts.values,
                name='Target Balance',
                showlegend=False
            ), row=1, col=2)
            
            # Target vs primera variable categórica (si existe)
            if categorical_features:
                cat_feat = categorical_features[0]
                cross_tab = pd.crosstab(df[cat_feat], df['target'])
                
                for target_val in cross_tab.columns:
                    fig.add_trace(go.Bar(
                        x=cross_tab.index,
                        y=cross_tab[target_val],
                        name=f'Target {target_val}',
                        showlegend=True
                    ), row=2, col=1)
                    
                fig.update_xaxes(title_text=cat_feat, row=2, col=1)
            
            # Target entropy/diversity (estilo SyntheticVisualizer)
            target_probs = target_counts / len(df)
            entropy = -sum(p * np.log2(p) for p in target_probs if p > 0)
            
            fig.add_trace(go.Bar(
                x=['Entropy', 'Classes', 'Mode Frequency'],
                y=[entropy, len(target_counts), target_counts.max()/len(df)],
                name='Target Metrics',
                showlegend=False,
                marker_color='orange'
            ), row=2, col=2)
            
            fig.update_layout(
                title=f"Real Data Target Categorical Analysis (SyntheticVisualizer Style) - Mode: Class {mode_target}",
                template='plotly_white',
                height=600
            )
            
            return RealVisualizer._save_plotly_file(fig, f'{generator_name}_target_categorical_synth', output_dir)
            
        except Exception as e:
            print(f"Target categorical synth style error: {e}")
            return None