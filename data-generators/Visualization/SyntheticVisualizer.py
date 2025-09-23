#!/usr/bin/env python3
"""
Synthetic Data Visualizer for CALMOPS
=====================================

Specialized visualization system for synthetic data generation and analysis.
Includes drift detection, temporal analysis, and block-based visualizations.

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
from sklearn.ensemble import RandomForestClassifier
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

class SyntheticVisualizer:
    """
    Specialized visualization system for synthetic data with drift analysis
    """

    @staticmethod
    def auto_analyze_and_visualize(data: List[Tuple], generator_name: str, 
                                  output_dir: str = "outputs",
                                  enable_plotly: bool = True) -> Dict[str, Any]:
        """
        Complete automatic analysis and visualization for synthetic data
        
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
        df = SyntheticVisualizer._convert_to_dataframe(data)
        
        # Get feature columns (excluding target)
        feature_cols = [col for col in df.columns if col != 'target']
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        # Calculate quality metrics
        quality_results = SyntheticVisualizer._calculate_quality_score(df, feature_cols)
        
        # Create visualizations
        visualization_files = {}
        
        try:
            if enable_plotly:
                # Create Plotly visualizations
                visualization_files.update(
                    SyntheticVisualizer._create_plotly_visualizations(
                        df, feature_cols, numeric_features, quality_results, 
                        generator_name, output_dir
                    )
                )
            else:
                # Create matplotlib visualizations
                visualization_files.update(
                    SyntheticVisualizer._create_matplotlib_visualizations(
                        df, feature_cols, numeric_features, quality_results,
                        generator_name, output_dir
                    )
                )
            
            # Perform drift analysis if applicable
            drift_results = SyntheticVisualizer._analyze_synthetic_drift(df, numeric_features)
            
            # Block analysis if block column exists
            block_results = None
            if any(col in df.columns for col in ['block', 'chunk', 'Block', 'Chunk']):
                block_results = SyntheticVisualizer._analyze_block_structure(df, output_dir, generator_name)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            visualization_files = {"error": str(e)}
        
        return {
            'quality_score': quality_results,
            'visualization_files': visualization_files,
            'drift_analysis': drift_results,
            'block_analysis': block_results,
            'dataset_info': {
                'samples': len(df),
                'features': len(feature_cols),
                'numeric_features': len(numeric_features)
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
    def _calculate_quality_score(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Calculate synthetic data quality metrics"""
        
        # Statistical validity
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        statistical_validity = 80  # Base score
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                # Check for reasonable distribution
                if df[col].std() == 0:
                    statistical_validity -= 10  # Penalty for constant features
                elif df[col].isnull().sum() > len(df) * 0.1:
                    statistical_validity -= 5   # Penalty for too many nulls
        
        # Feature diversity
        feature_diversity = min(100, len(feature_cols) * 10)  # 10 points per feature, max 100
        
        # Data completeness
        completeness_ratio = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        data_completeness = completeness_ratio * 100
        
        # Class balance (if target exists)
        class_balance = 100
        if 'target' in df.columns:
            class_counts = df['target'].value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1.0
            class_balance = max(0, 100 - (imbalance_ratio - 1) * 20)  # Penalty for imbalance
        
        # Overall score
        overall_score = (statistical_validity + feature_diversity + data_completeness + class_balance) / 4
        
        return {
            'overall_score': round(overall_score, 1),
            'statistical_validity': round(statistical_validity, 1),
            'feature_diversity': round(feature_diversity, 1),
            'data_completeness': round(data_completeness, 1),
            'class_balance': round(class_balance, 1),
            'details': {
                'total_samples': len(df),
                'total_features': len(feature_cols),
                'missing_values': df.isnull().sum().sum(),
                'numeric_features': len(numeric_cols)
            }
        }

    @staticmethod
    def _create_plotly_visualizations(df: pd.DataFrame, feature_cols: List[str],
                                    numeric_features: List[str], quality_results: Dict,
                                    generator_name: str, output_dir: str) -> Dict[str, str]:
        """Create Plotly interactive visualizations"""
        
        viz_files = {}
        
        # Quality breakdown
        quality_file = SyntheticVisualizer._plot_quality_breakdown_plotly(
            quality_results, generator_name, output_dir
        )
        if quality_file:
            viz_files['quality_breakdown'] = quality_file
        
        # ALWAYS create detailed variable evolution plots  
        detailed_evolution_file = SyntheticVisualizer.create_detailed_variable_evolution_plotly(
            df, feature_cols, generator_name, output_dir
        )
        if detailed_evolution_file:
            viz_files['detailed_variable_evolution'] = detailed_evolution_file
        
        # Feature distributions
        for feature in feature_cols[:6]:  # Limit to first 6 features
            if feature in numeric_features:
                dist_file = SyntheticVisualizer._plot_feature_distribution_plotly(
                    df, feature, generator_name, output_dir
                )
                if dist_file:
                    viz_files[f'distribution_{feature}'] = dist_file
        
        # Correlation heatmap
        if len(numeric_features) > 1:
            corr_file = SyntheticVisualizer._plot_correlation_heatmap_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if corr_file:
                viz_files['correlation_heatmap'] = corr_file
        
        # PCA projection
        if len(numeric_features) >= 2:
            pca_file = SyntheticVisualizer._plot_pca_projection_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if pca_file:
                viz_files['pca_projection'] = pca_file
        
        # Statistical summary
        if len(numeric_features) > 0:
            stats_file = SyntheticVisualizer._plot_statistical_summary_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if stats_file:
                viz_files['statistical_summary'] = stats_file
        
        # NUEVOS PLOTS ADICIONALES
        
        # Box plots detallados
        if len(numeric_features) > 0:
            boxplot_detailed_file = SyntheticVisualizer._plot_boxplots_detailed_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if boxplot_detailed_file:
                viz_files['boxplots_detailed'] = boxplot_detailed_file
        
        # Histogramas comparativos
        for feature in numeric_features[:4]:  # Primeras 4 features
            histogram_file = SyntheticVisualizer._plot_histogram_detailed_plotly(
                df, feature, generator_name, output_dir
            )
            if histogram_file:
                viz_files[f'histogram_detailed_{feature}'] = histogram_file
        
        # Análisis de variables categóricas (si existen)
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'target' in categorical_features:
            categorical_features.remove('target')
        
        if categorical_features:
            categorical_file = SyntheticVisualizer._plot_categorical_analysis_plotly(
                df, categorical_features, generator_name, output_dir
            )
            if categorical_file:
                viz_files['categorical_analysis'] = categorical_file
        
        # Violin plots (combinación de box plot y density)
        if len(numeric_features) > 0:
            violin_file = SyntheticVisualizer._plot_violin_plots_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if violin_file:
                viz_files['violin_plots'] = violin_file
        
        # Distribución conjunta (pairwise)
        if len(numeric_features) >= 2:
            pairwise_file = SyntheticVisualizer._plot_pairwise_distributions_plotly(
                df, numeric_features, generator_name, output_dir
            )
            if pairwise_file:
                viz_files['pairwise_distributions'] = pairwise_file
        
        # Análisis de target categórico (moda, frecuencias)
        if 'target' in df.columns:
            target_categorical_file = SyntheticVisualizer._plot_target_categorical_plotly(
                df, categorical_features, generator_name, output_dir
            )
            if target_categorical_file:
                viz_files['target_categorical'] = target_categorical_file
        
        return viz_files

    @staticmethod
    def _create_matplotlib_visualizations(df: pd.DataFrame, feature_cols: List[str],
                                        numeric_features: List[str], quality_results: Dict,
                                        generator_name: str, output_dir: str) -> Dict[str, str]:
        """Create matplotlib static visualizations"""
        
        viz_files = {}
        
        # Create comprehensive matplotlib plot
        try:
            plt.style.use('default')
            
            # Determine grid size
            n_plots = min(6, len(feature_cols)) + 3  # Features + quality + correlation + PCA
            n_cols = 3
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            plot_idx = 0
            
            # Quality breakdown
            if plot_idx < len(axes.flat):
                ax = axes.flat[plot_idx]
                metrics = ['Statistical\nValidity', 'Feature\nDiversity', 'Data\nCompleteness', 'Class\nBalance']
                values = [quality_results['statistical_validity'], quality_results['feature_diversity'],
                         quality_results['data_completeness'], quality_results['class_balance']]
                bars = ax.bar(metrics, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'])
                ax.set_ylim(0, 100)
                ax.set_title(f'Quality Breakdown - {generator_name}')
                ax.set_ylabel('Score')
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom')
                plot_idx += 1
            
            # Feature distributions
            for feature in feature_cols[:4]:  # First 4 features
                if plot_idx < len(axes.flat) and feature in numeric_features:
                    ax = axes.flat[plot_idx]
                    ax.hist(df[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_title(f'{feature} Distribution')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Frequency')
                    plot_idx += 1
            
            # Correlation heatmap
            if plot_idx < len(axes.flat) and len(numeric_features) > 1:
                ax = axes.flat[plot_idx]
                corr_matrix = df[numeric_features].corr()
                im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric_features)))
                ax.set_yticks(range(len(numeric_features)))
                ax.set_xticklabels(numeric_features, rotation=45, ha='right')
                ax.set_yticklabels(numeric_features)
                ax.set_title('Feature Correlations')
                plt.colorbar(im, ax=ax, shrink=0.8)
                plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, len(axes.flat)):
                axes.flat[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(output_dir, f'{generator_name}_synthetic_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['comprehensive_analysis'] = output_file
            
        except Exception as e:
            print(f"Matplotlib visualization error: {e}")
        
        return viz_files

    @staticmethod
    def _analyze_synthetic_drift(df: pd.DataFrame, numeric_features: List[str]) -> Dict[str, Any]:
        """Analyze drift patterns in synthetic data"""
        
        drift_results = {
            'temporal_drift': False,
            'feature_stability': {},
            'drift_score': 0.0
        }
        
        if len(numeric_features) == 0:
            return drift_results
        
        # Check for temporal patterns (if data has sequence)
        try:
            # Simple temporal drift detection
            for feature in numeric_features[:3]:  # Check first 3 features
                data = df[feature].dropna()
                if len(data) > 10:
                    # Split into first and last halves
                    mid_point = len(data) // 2
                    first_half = data.iloc[:mid_point]
                    second_half = data.iloc[mid_point:]
                    
                    # Statistical test for difference
                    try:
                        stat, p_value = stats.ks_2samp(first_half, second_half)
                        drift_results['feature_stability'][feature] = {
                            'ks_statistic': stat,
                            'p_value': p_value,
                            'has_drift': p_value < 0.05
                        }
                        
                        if p_value < 0.05:
                            drift_results['temporal_drift'] = True
                    except:
                        pass
        except Exception as e:
            drift_results['error'] = str(e)
        
        # Calculate overall drift score
        drift_features = sum(1 for f in drift_results['feature_stability'].values() if f.get('has_drift', False))
        total_features = len(drift_results['feature_stability'])
        if total_features > 0:
            drift_results['drift_score'] = drift_features / total_features
        
        return drift_results

    @staticmethod
    def _analyze_block_structure(df: pd.DataFrame, output_dir: str, generator_name: str) -> Dict[str, Any]:
        """Analyze block structure in synthetic data"""
        
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
            'block_distributions': {},
            'inter_block_drift': []
        }
        
        # Analyze each block
        for block_id in unique_blocks:
            block_data = df[df[block_col] == block_id]
            block_results['block_sizes'][str(block_id)] = len(block_data)
            
            if 'target' in block_data.columns:
                block_dist = block_data['target'].value_counts(normalize=True).to_dict()
                block_results['block_distributions'][str(block_id)] = block_dist
        
        # Calculate inter-block drift
        if 'target' in df.columns and len(unique_blocks) > 1:
            for i in range(len(unique_blocks) - 1):
                current_block = unique_blocks[i]
                next_block = unique_blocks[i + 1]
                
                current_data = df[df[block_col] == current_block]
                next_data = df[df[block_col] == next_block]
                
                current_dist = current_data['target'].value_counts(normalize=True)
                next_dist = next_data['target'].value_counts(normalize=True)
                
                # Calculate Total Variation Distance
                all_classes = set(current_dist.index) | set(next_dist.index)
                tvd = 0.5 * sum(abs(current_dist.get(cls, 0) - next_dist.get(cls, 0)) for cls in all_classes)
                
                block_results['inter_block_drift'].append({
                    'from_block': current_block,
                    'to_block': next_block,
                    'tvd': tvd,
                    'significant': tvd > 0.1
                })
        
        return block_results

    # Plotly visualization methods
    @staticmethod
    def _plot_quality_breakdown_plotly(quality_results: Dict, generator_name: str, 
                                     output_dir: str) -> Optional[str]:
        """Create interactive quality breakdown plot"""
        try:
            metrics = ['Statistical Validity', 'Feature Diversity', 'Data Completeness', 'Class Balance']
            values = [quality_results['statistical_validity'], quality_results['feature_diversity'],
                     quality_results['data_completeness'], quality_results['class_balance']]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']
            
            fig = go.Figure(data=[
                go.Bar(x=metrics, y=values, marker_color=colors,
                      text=[f'{v:.1f}' for v in values],
                      textposition='auto')
            ])
            
            fig.update_layout(
                title=f'Synthetic Data Quality Breakdown - {generator_name}',
                xaxis_title='Quality Metrics',
                yaxis_title='Score (0-100)',
                yaxis=dict(range=[0, 100]),
                template='plotly_white'
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_quality', output_dir)
            
        except Exception as e:
            print(f"Quality plot error: {e}")
            return None

    @staticmethod
    def _plot_feature_distribution_plotly(df: pd.DataFrame, feature: str, generator_name: str,
                                        output_dir: str) -> Optional[str]:
        """Create interactive feature distribution plot"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Violin Plot'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter'}, {'type': 'scatter'}]]
            )
            
            data = df[feature].dropna()
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=data, name='Distribution', showlegend=False),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=data, name='Box Plot', showlegend=False),
                row=1, col=2
            )
            
            # Q-Q plot
            try:
                qq_data = stats.probplot(data, dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                             mode='markers', name='Q-Q Plot', showlegend=False),
                    row=2, col=1
                )
                # Add reference line
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][0],
                             mode='lines', name='Reference', showlegend=False,
                             line=dict(color='red', dash='dash')),
                    row=2, col=1
                )
            except:
                pass
            
            # Violin plot
            fig.add_trace(
                go.Violin(y=data, name='Violin Plot', showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Feature Analysis: {feature} - {generator_name}',
                template='plotly_white',
                height=600
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_{feature}_dist', output_dir)
            
        except Exception as e:
            print(f"Feature distribution plot error: {e}")
            return None

    @staticmethod
    def _plot_correlation_heatmap_plotly(df: pd.DataFrame, numeric_features: List[str],
                                       generator_name: str, output_dir: str) -> Optional[str]:
        """Create interactive correlation heatmap"""
        try:
            corr_matrix = df[numeric_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=numeric_features,
                y=numeric_features,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f'Feature Correlations - {generator_name}',
                template='plotly_white',
                width=max(500, len(numeric_features) * 50),
                height=max(500, len(numeric_features) * 50)
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_correlations', output_dir)
            
        except Exception as e:
            print(f"Correlation plot error: {e}")
            return None

    @staticmethod
    def _plot_pca_projection_plotly(df: pd.DataFrame, numeric_features: List[str],
                                  generator_name: str, output_dir: str) -> Optional[str]:
        """Create interactive PCA projection"""
        try:
            # Prepare data
            data = df[numeric_features].dropna()
            if len(data) < 5:
                return None
            
            # Standardize and apply PCA
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            pca = PCA(n_components=min(3, len(numeric_features)))
            pca_result = pca.fit_transform(data_scaled)
            
            # Create DataFrame for plotting
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
            
            # Add target if available
            if 'target' in df.columns:
                pca_df['target'] = df['target'].iloc[:len(pca_df)].values
                color_col = 'target'
            else:
                color_col = None
            
            # Create plot
            if pca_result.shape[1] >= 2:
                fig = px.scatter(pca_df, x='PC1', y='PC2', color=color_col,
                               title=f'PCA Projection - {generator_name}',
                               labels={
                                   'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
                               })
                
                # Add explained variance info
                total_variance = sum(pca.explained_variance_ratio_[:2])
                fig.add_annotation(
                    text=f'Total Variance Explained: {total_variance:.1%}',
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, showarrow=False,
                    bgcolor="white", bordercolor="black"
                )
                
                fig.update_layout(template='plotly_white')
                
                return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_pca', output_dir)
            
        except Exception as e:
            print(f"PCA plot error: {e}")
            return None

    @staticmethod
    def _plot_statistical_summary_plotly(df: pd.DataFrame, numeric_features: List[str],
                                       generator_name: str, output_dir: str) -> Optional[str]:
        """Create interactive statistical summary"""
        try:
            # Calculate statistics
            stats_data = []
            for feature in numeric_features[:8]:  # Limit to 8 features
                data = df[feature].dropna()
                if len(data) > 0:
                    stats_data.append({
                        'Feature': feature,
                        'Mean': data.mean(),
                        'Std': data.std(),
                        'Min': data.min(),
                        'Max': data.max(),
                        'Skewness': data.skew(),
                        'Kurtosis': data.kurtosis()
                    })
            
            if not stats_data:
                return None
            
            stats_df = pd.DataFrame(stats_data)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Mean & Std', 'Min & Max', 'Skewness', 'Kurtosis'),
                specs=[[{'secondary_y': True}, {'secondary_y': True}],
                       [{'secondary_y': False}, {'secondary_y': False}]]
            )
            
            # Mean & Std (with independent scales)
            fig.add_trace(
                go.Bar(x=stats_df['Feature'], y=stats_df['Mean'], 
                      name='Mean', marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=stats_df['Feature'], y=stats_df['Std'], 
                          mode='lines+markers', name='Std', 
                          marker_color='red', yaxis='y2'),
                row=1, col=1, secondary_y=True
            )
            
            # Min & Max
            fig.add_trace(
                go.Scatter(x=stats_df['Feature'], y=stats_df['Min'], 
                          mode='lines+markers', name='Min', marker_color='green'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=stats_df['Feature'], y=stats_df['Max'], 
                          mode='lines+markers', name='Max', marker_color='orange'),
                row=1, col=2
            )
            
            # Skewness
            fig.add_trace(
                go.Bar(x=stats_df['Feature'], y=stats_df['Skewness'], 
                      name='Skewness', marker_color='purple'),
                row=2, col=1
            )
            
            # Kurtosis
            fig.add_trace(
                go.Bar(x=stats_df['Feature'], y=stats_df['Kurtosis'], 
                      name='Kurtosis', marker_color='brown'),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Statistical Summary - {generator_name}',
                template='plotly_white',
                height=600,
                showlegend=True
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Mean", row=1, col=1)
            fig.update_yaxes(title_text="Std", secondary_y=True, row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=2)
            fig.update_yaxes(title_text="Skewness", row=2, col=1)
            fig.update_yaxes(title_text="Kurtosis", row=2, col=2)
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_statistics', output_dir)
            
        except Exception as e:
            print(f"Statistical summary plot error: {e}")
            return None

    @staticmethod
    def _save_plotly_file(fig, base_filename: str, output_dir: str) -> Optional[str]:
        """Save Plotly figure as HTML"""
        try:
            html_file = os.path.join(output_dir, f'{base_filename}.html')
            fig.write_html(html_file)
            
            # Also try to create PNG fallback
            try:
                png_file = os.path.join(output_dir, f'{base_filename}.png')
                fig.write_image(png_file, width=800, height=600)
            except:
                # Create matplotlib fallback
                SyntheticVisualizer._create_matplotlib_fallback(fig, base_filename, output_dir)
            
            return html_file
            
        except Exception as e:
            print(f"Plotly save error: {e}")
            return None

    @staticmethod
    def _create_matplotlib_fallback(plotly_fig, base_filename: str, output_dir: str) -> Optional[str]:
        """Create matplotlib fallback for plotly figures"""
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Interactive visualization available in:\n{base_filename}.html', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title(f'Visualization: {base_filename}')
            plt.axis('off')
            
            png_file = os.path.join(output_dir, f'{base_filename}.png')
            plt.savefig(png_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return png_file
            
        except Exception as e:
            print(f"Matplotlib fallback error: {e}")
            return None
    
    @staticmethod
    def create_detailed_variable_evolution_plotly(df: pd.DataFrame, feature_cols: List[str], 
                                                generator_name: str, output_dir: str) -> Optional[str]:
        """Create detailed Plotly visualization showing evolution of each numeric variable"""
        try:
            # Check if we have block data
            has_blocks = 'block' in df.columns
                
            # Filter numeric features
            numeric_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
            
            if not numeric_cols:
                print("Warning: No numeric features found for detailed evolution plot")
                return None
            
            # Add instance index
            df_viz = df.copy()
            df_viz['instance_idx'] = range(len(df_viz))
            
            # Create subplots
            n_vars = len(numeric_cols)
            title_suffix = "Instances & Blocks" if has_blocks else "Instances (Sequential)"
            subplot_titles = [f'{var} - Evolution Across {title_suffix}' for var in numeric_cols]
            
            fig = make_subplots(
                rows=n_vars, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                specs=[[{"secondary_y": True}] for _ in range(n_vars)]
            )
            
            # Get unique blocks and colors
            unique_blocks = sorted(df['block'].unique()) if has_blocks else []
            colors_discrete = px.colors.qualitative.Set3
            
            for i, var in enumerate(numeric_cols):
                row_idx = i + 1
                
                # Raw data line
                fig.add_trace(
                    go.Scatter(
                        x=df_viz['instance_idx'],
                        y=df_viz[var],
                        mode='lines',
                        name=f'{var} - Raw Values',
                        line=dict(color='steelblue', width=2),
                        showlegend=(i == 0),  # Show legend only for first variable
                        legendgroup='raw'
                    ),
                    row=row_idx, col=1
                )
                
                # Rolling mean
                if len(df_viz) > 10:
                    window = max(5, len(df_viz) // 20)
                    rolling_mean = df_viz[var].rolling(window=window, center=True).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df_viz['instance_idx'],
                            y=rolling_mean,
                            mode='lines',
                            name=f'{var} - Trend',
                            line=dict(color='red', width=3),
                            showlegend=(i == 0),
                            legendgroup='trend'
                        ),
                        row=row_idx, col=1
                    )
                
                # Block boundaries and statistics
                if has_blocks:
                    block_start_idx = 0
                    for j, block in enumerate(unique_blocks):
                        block_data = df_viz[df_viz['block'] == block]
                        block_end_idx = block_start_idx + len(block_data)
                        block_color = colors_discrete[j % len(colors_discrete)]
                        
                        # Block mean line
                        block_mean = block_data[var].mean()
                        fig.add_shape(
                            type="line",
                            x0=block_start_idx, x1=block_end_idx-1,
                            y0=block_mean, y1=block_mean,
                            line=dict(color=block_color, width=3, dash="dash"),
                            row=row_idx, col=1
                        )
                        
                        # Block boundary (vertical line)
                        if j > 0:
                            fig.add_shape(
                                type="line",
                                x0=block_start_idx, x1=block_start_idx,
                                y0=df_viz[var].min(), y1=df_viz[var].max(),
                                line=dict(color="gray", width=1, dash="dot"),
                                row=row_idx, col=1
                            )
                        
                        # Block annotation
                        mid_x = (block_start_idx + block_end_idx - 1) / 2
                        fig.add_annotation(
                            x=mid_x,
                            y=block_mean,
                            text=f"B{block}<br>μ={block_mean:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=block_color,
                            bgcolor="white",
                            bordercolor=block_color,
                            borderwidth=1,
                            font=dict(size=10),
                            row=row_idx, col=1
                        )
                        
                        block_start_idx = block_end_idx
                
                # Update axes
                fig.update_xaxes(title_text="Instance Index", row=row_idx, col=1)
                fig.update_yaxes(title_text=f"{var} Value", row=row_idx, col=1)
            
            # Update layout
            fig.update_layout(
                height=400 * n_vars,
                title_text=f"{generator_name} - Detailed Variable Evolution",
                showlegend=True,
                font=dict(size=12)
            )
            
            # Save file
            base_filename = f'{generator_name}_detailed_variable_evolution_plotly'
            html_file = os.path.join(output_dir, f'{base_filename}.html')
            fig.write_html(html_file)
            
            # Create matplotlib fallback
            SyntheticVisualizer._create_matplotlib_fallback(fig, base_filename, output_dir)
            
            print(f"✓ Generated detailed Plotly variable evolution plot with {n_vars} numeric variables")
            return html_file
            
        except Exception as e:
            print(f"Warning: Detailed Plotly variable evolution plot failed: {e}")
            return None

    # NUEVOS MÉTODOS PARA PLOTS ADICIONALES
    
    @staticmethod
    def _plot_boxplots_detailed_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                     generator_name: str, output_dir: str) -> Optional[str]:
        """Box plots detallados con estadísticas"""
        try:
            fig = go.Figure()
            
            for feature in numeric_features:
                data = df[feature].dropna()
                
                fig.add_trace(go.Box(
                    y=data,
                    name=feature,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    hovertemplate=f'<b>{feature}</b><br>Value: %{{y:.3f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Detailed Box Plots with Outliers",
                yaxis_title="Value",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_boxplots_detailed', output_dir)
            
        except Exception as e:
            print(f"Detailed box plots error: {e}")
            return None

    @staticmethod
    def _plot_histogram_detailed_plotly(df: pd.DataFrame, feature: str, 
                                      generator_name: str, output_dir: str) -> Optional[str]:
        """Histograma detallado con estadísticas superpuestas"""
        try:
            data = df[feature].dropna()
            
            fig = go.Figure()
            
            # Histograma principal
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=50,
                name=feature,
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate=f'<b>{feature}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
            
            # Estadísticas
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
            
            # Área de desviación estándar
            fig.add_vrect(
                x0=mean_val - std_val, x1=mean_val + std_val,
                fillcolor="yellow", opacity=0.2,
                annotation_text=f"±1σ", annotation_position="top left"
            )
            
            fig.update_layout(
                title=f"Detailed Histogram: {feature} (μ={mean_val:.3f}, σ={std_val:.3f})",
                xaxis_title=feature,
                yaxis_title="Frequency",
                template='plotly_white',
                height=450
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_hist_detailed_{feature}', output_dir)
            
        except Exception as e:
            print(f"Detailed histogram error: {e}")
            return None

    @staticmethod
    def _plot_categorical_analysis_plotly(df: pd.DataFrame, categorical_features: List[str], 
                                        generator_name: str, output_dir: str) -> Optional[str]:
        """Análisis de variables categóricas: frecuencias y moda"""
        try:
            if not categorical_features:
                return None
                
            n_features = len(categorical_features)
            n_cols = 2
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f'{feat} Distribution' for feat in categorical_features]
            )
            
            for i, feature in enumerate(categorical_features):
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                
                # Contar frecuencias
                value_counts = df[feature].value_counts()
                mode_value = value_counts.index[0]  # Moda
                
                fig.add_trace(go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name=feature,
                    showlegend=False,
                    marker_color='lightcoral',
                    hovertemplate=f'<b>{feature}</b><br>Category: %{{x}}<br>Count: %{{y}}<br>Mode: {mode_value}<extra></extra>'
                ), row=row, col=col)
            
            fig.update_layout(
                title="Categorical Variables Analysis (Mode & Frequencies)",
                template='plotly_white',
                height=300 * n_rows
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_categorical', output_dir)
            
        except Exception as e:
            print(f"Categorical analysis error: {e}")
            return None

    @staticmethod
    def _plot_violin_plots_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                generator_name: str, output_dir: str) -> Optional[str]:
        """Violin plots: combinación de box plot y density"""
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
                title="Violin Plots (Distribution Shape + Box Plot)",
                yaxis_title="Value",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_violin', output_dir)
            
        except Exception as e:
            print(f"Violin plots error: {e}")
            return None

    @staticmethod
    def _plot_pairwise_distributions_plotly(df: pd.DataFrame, numeric_features: List[str], 
                                          generator_name: str, output_dir: str) -> Optional[str]:
        """Distribuciones conjuntas pairwise (scatter plots)"""
        try:
            # Limitar a primeras 4 features para no sobrecargar
            features_subset = numeric_features[:4]
            n_features = len(features_subset)
            
            if n_features < 2:
                return None
            
            # Crear scatter matrix
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
                            marker_color='lightblue'
                        ), row=row, col=col)
                    else:
                        # Off-diagonal: scatter plot
                        if 'target' in df.columns:
                            # Colorear por target si existe
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
                                marker=dict(size=4, opacity=0.6)
                            ), row=row, col=col)
            
            fig.update_layout(
                title="Pairwise Feature Distributions",
                template='plotly_white',
                height=200 * n_features
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_pairwise', output_dir)
            
        except Exception as e:
            print(f"Pairwise distributions error: {e}")
            return None

    @staticmethod
    def _plot_target_categorical_plotly(df: pd.DataFrame, categorical_features: List[str], 
                                      generator_name: str, output_dir: str) -> Optional[str]:
        """Análisis del target: moda, frecuencias y relación con categóricas"""
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
                marker_color='lightblue',
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
            
            # Target entropy/diversity
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
                title=f"Target Categorical Analysis (Mode: Class {mode_target})",
                template='plotly_white',
                height=600
            )
            
            return SyntheticVisualizer._save_plotly_file(fig, f'{generator_name}_target_categorical', output_dir)
            
        except Exception as e:
            print(f"Target categorical error: {e}")
            return None