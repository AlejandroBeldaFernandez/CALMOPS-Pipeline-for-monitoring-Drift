#!/usr/bin/env python3
"""
Auto-Visualizer for CALMOPS - Matplotlib-based Visualization System
====================================================================

This module provides automatic visualization capabilities using Matplotlib
for simple, clean plots showing feature evolution and data patterns.

Author: CalmOps Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from typing import List, Tuple, Dict, Any, Optional
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Non-interactive backend

# Set improved matplotlib and seaborn styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#666666',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#e0e0e0',
    'grid.alpha': 0.6,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
})

class AutoVisualizer:
    """
    Automatic visualization system using Matplotlib for simple, clean plots
    """
    
    @staticmethod
    def auto_analyze_and_visualize(data: List[Tuple], generator_name: str, 
                                 output_dir: str = "auto_viz_output") -> Dict[str, Any]:
        """
        Automatically analyze and visualize synthetic data using Matplotlib
        
        Args:
            data: List of (features_dict, target) tuples from generator
            generator_name: Name of the generator
            output_dir: Directory to save visualizations
            
        Returns:
            Analysis results dictionary
        """
        
        print(f"\nAUTO-VISUALIZATION: {generator_name}")
        print("-" * 50)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame  
        df = AutoVisualizer._convert_to_dataframe(data)
        feature_cols = [col for col in df.columns if col != 'target']
        
        print(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
        
        # Quality Analysis
        quality_results = AutoVisualizer._calculate_quality_score(df, feature_cols)
        
        # Check if this is block data
        has_blocks = 'block' in df.columns
        
        # Create visualizations
        viz_files = AutoVisualizer._create_matplotlib_plots(
            df, feature_cols, generator_name, quality_results, output_dir, has_blocks
        )
        
        print(f"Generated {len(viz_files)} plots")
        
        return {
            'dataset_info': {
                'samples': len(df),
                'features': len(feature_cols),
                'generator': generator_name
            },
            'quality_score': quality_results,
            'visualization_files': viz_files
        }
    
    @staticmethod
    def _convert_to_dataframe(data: List[Tuple]) -> pd.DataFrame:
        """Convert list of tuples to DataFrame"""
        if not data:
            return pd.DataFrame()
        
        # Extract features and targets
        all_features = []
        all_targets = []
        
        for features_dict, target in data:
            all_features.append(features_dict)
            all_targets.append(target)
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        if all_targets[0] is not None:
            df['target'] = all_targets
        
        return df
    
    @staticmethod
    def _calculate_quality_score(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Calculate basic quality score"""
        if len(df) == 0:
            return {'overall_score': 0, 'grade': 'F'}
        
        score = 85  # Base score
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= missing_pct * 2
        
        # Check for duplicates
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        score -= duplicate_pct
        
        # Grade assignment
        if score >= 90:
            grade = 'A+'
        elif score >= 85:
            grade = 'A'
        elif score >= 80:
            grade = 'B+'
        elif score >= 75:
            grade = 'B'
        elif score >= 70:
            grade = 'C+'
        elif score >= 65:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'overall_score': max(0, score),
            'grade': grade,
            'missing_percentage': missing_pct,
            'duplicate_percentage': duplicate_pct
        }
    
    @staticmethod
    def _create_matplotlib_plots(df: pd.DataFrame, feature_cols: List[str], 
                                generator_name: str, quality_results: Dict, 
                                output_dir: str, has_blocks: bool = False) -> Dict[str, str]:
        """Create matplotlib plots showing feature evolution and patterns"""
        
        viz_files = {}
        
        # 1. Feature Evolution Plot (block-based if applicable)
        if len(feature_cols) > 0:
            if has_blocks:
                evolution_file = AutoVisualizer._plot_block_feature_evolution(
                    df, feature_cols, generator_name, output_dir
                )
            else:
                evolution_file = AutoVisualizer._plot_feature_evolution(
                    df, feature_cols, generator_name, output_dir
                )
            if evolution_file:
                viz_files['feature_evolution'] = evolution_file
        
        # 2. Target Distribution Plot (block-based if applicable)
        if 'target' in df.columns:
            if has_blocks:
                target_file = AutoVisualizer._plot_block_target_distribution(
                    df, generator_name, output_dir
                )
            else:
                target_file = AutoVisualizer._plot_target_distribution(
                    df, generator_name, output_dir
                )
            if target_file:
                viz_files['target_distribution'] = target_file
        
        # 3. Feature Correlation Plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        if len(numeric_cols) > 1:
            corr_file = AutoVisualizer._plot_correlation_matrix(
                df, numeric_cols, generator_name, output_dir
            )
            if corr_file:
                viz_files['correlation'] = corr_file
        
        # 4. PCA Plot
        if len(numeric_cols) >= 2:
            pca_file = AutoVisualizer._plot_pca_analysis(
                df, numeric_cols, generator_name, output_dir
            )
            if pca_file:
                viz_files['pca'] = pca_file
        
        # 5. Categorical Features Plot
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'target' in categorical_cols:
            categorical_cols.remove('target')
        
        if len(categorical_cols) > 0:
            cat_file = AutoVisualizer._plot_categorical_features(
                df, categorical_cols, generator_name, output_dir
            )
            if cat_file:
                viz_files['categorical_features'] = cat_file
                
            # Add mode-based categorical evolution plot
            mode_file = AutoVisualizer._plot_categorical_mode_evolution(
                df, categorical_cols, generator_name, output_dir, has_blocks
            )
            if mode_file:
                viz_files['categorical_mode_evolution'] = mode_file
        
        # 6. Box Plots for Numeric Variables
        if len(feature_cols) > 0:
            box_file = AutoVisualizer._plot_boxplots(
                df, feature_cols, generator_name, output_dir, has_blocks
            )
            if box_file:
                viz_files['boxplots'] = box_file
        
        # 7. Histograms for Numeric Variables
        if len(feature_cols) > 0:
            hist_file = AutoVisualizer._plot_histograms(
                df, feature_cols, generator_name, output_dir, has_blocks
            )
            if hist_file:
                viz_files['histograms'] = hist_file
        
        return viz_files

    @staticmethod
    def compare_datasets_and_visualize(original_data, synthetic_data, 
                                     generator_name: str, output_dir: str) -> Dict[str, str]:
        """Compare original and synthetic datasets and create comparison plots"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            viz_files = {}
            
            # Convert data to DataFrame if needed
            if isinstance(original_data, list):
                orig_df = AutoVisualizer._tuples_to_dataframe(original_data)
            else:
                orig_df = original_data.copy()
                
            if isinstance(synthetic_data, list):
                synth_df = AutoVisualizer._tuples_to_dataframe(synthetic_data)
            else:
                synth_df = synthetic_data.copy()
            
            # Get feature columns (excluding target)
            feature_cols = [col for col in orig_df.columns if col != 'target']
            numeric_cols = orig_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = orig_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 1. Dataset Comparison Summary Plot
            summary_file = AutoVisualizer._plot_dataset_comparison_summary(
                orig_df, synth_df, generator_name, output_dir
            )
            if summary_file:
                viz_files['dataset_comparison_summary'] = summary_file
            
            # 2. Numeric Features Distribution Comparison
            if len(numeric_cols) > 0:
                numeric_comp_file = AutoVisualizer._plot_numeric_comparison(
                    orig_df, synth_df, numeric_cols, generator_name, output_dir
                )
                if numeric_comp_file:
                    viz_files['numeric_comparison'] = numeric_comp_file
            
            # 3. Categorical Features Distribution Comparison  
            if len(categorical_cols) > 0:
                categorical_comp_file = AutoVisualizer._plot_categorical_comparison(
                    orig_df, synth_df, categorical_cols, generator_name, output_dir
                )
                if categorical_comp_file:
                    viz_files['categorical_comparison'] = categorical_comp_file
            
            # 4. Target Distribution Comparison
            if 'target' in orig_df.columns and 'target' in synth_df.columns:
                target_comp_file = AutoVisualizer._plot_target_comparison(
                    orig_df, synth_df, generator_name, output_dir
                )
                if target_comp_file:
                    viz_files['target_comparison'] = target_comp_file
            
            return viz_files
            
        except Exception as e:
            print(f"Warning: Dataset comparison plots failed: {e}")
            return {}
    
    @staticmethod
    def _plot_feature_evolution(df: pd.DataFrame, feature_cols: List[str], 
                               generator_name: str, output_dir: str) -> str:
        """Plot feature evolution over instances with improved styling"""
        try:
            # Select up to 6 features for visualization
            selected_features = feature_cols[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(f'{generator_name} - Feature Evolution Over Time', fontsize=14, fontweight='bold')
            axes = axes.flatten()
            
            # Define a color palette
            colors = sns.color_palette("husl", len(selected_features))
            
            for i, feature in enumerate(selected_features):
                if i >= 6:
                    break
                    
                ax = axes[i]
                color = colors[i]
                
                if df[feature].dtype in ['int64', 'float64']:
                    # Numeric feature - show evolution with smoothing
                    x_vals = range(len(df))
                    y_vals = df[feature].values
                    
                    # Plot original data
                    ax.plot(x_vals, y_vals, color=color, alpha=0.6, linewidth=1.2, label='Data')
                    
                    # Add smoothed trend line if enough data points
                    if len(df) > 50:
                        window = max(5, len(df) // 20)
                        smoothed = df[feature].rolling(window=window, center=True).mean()
                        ax.plot(x_vals, smoothed, color='darkred', linewidth=2.5, alpha=0.8, label='Trend')
                        ax.legend()
                    
                    ax.set_title(f'{feature}', fontweight='bold', pad=10)
                    ax.set_xlabel('Instance Index')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.4)
                    
                    # Add statistics annotation
                    mean_val = df[feature].mean()
                    std_val = df[feature].std()
                    ax.text(0.02, 0.98, f'μ: {mean_val:.3f}\nσ: {std_val:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                else:
                    # Categorical feature - show mode progression
                    window_size = max(10, len(df) // 15)
                    modes = []
                    positions = []
                    
                    for start in range(0, len(df) - window_size + 1, window_size//2):
                        end = min(start + window_size, len(df))
                        window_data = df[feature].iloc[start:end]
                        mode_val = window_data.mode().iloc[0] if len(window_data.mode()) > 0 else 'Unknown'
                        modes.append(str(mode_val))
                        positions.append(start + (end - start) // 2)
                    
                    # Create line plot for mode evolution
                    unique_modes = sorted(list(set(modes)))
                    mode_colors = sns.color_palette("Set2", len(unique_modes))
                    mode_to_num = {mode: i for i, mode in enumerate(unique_modes)}
                    
                    y_vals = [mode_to_num[mode] for mode in modes]
                    ax.plot(positions, y_vals, 'o-', color=color, linewidth=2, markersize=6, alpha=0.8)
                    
                    ax.set_title(f'{feature} (Mode Evolution)', fontweight='bold', pad=10)
                    ax.set_xlabel('Instance Index')
                    ax.set_ylabel('Mode Category')
                    ax.set_yticks(range(len(unique_modes)))
                    ax.set_yticklabels(unique_modes)
                    ax.grid(True, alpha=0.4)
            
            # Hide unused subplots
            for i in range(len(selected_features), 6):
                axes[i].axis('off')
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_feature_evolution.png')
            plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Feature evolution plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    @staticmethod
    def _plot_target_distribution(df: pd.DataFrame, generator_name: str, output_dir: str) -> str:
        """Plot target distribution with improved styling"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'{generator_name} - Target Distribution Analysis', fontsize=14, fontweight='bold')
            
            target_counts = df['target'].value_counts().sort_index()
            colors = sns.color_palette("viridis", len(target_counts))
            
            # Left plot: Bar chart
            if len(target_counts) <= 15:
                bars = ax1.bar(range(len(target_counts)), target_counts.values, 
                              color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
                ax1.set_xticks(range(len(target_counts)))
                ax1.set_xticklabels(target_counts.index, rotation=45 if len(target_counts) > 5 else 0)
                
                # Add value labels on bars
                for bar, value in zip(bars, target_counts.values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(target_counts.values),
                            f'{value}\n({value/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=9)
            else:
                # Line plot for many categories
                ax1.plot(target_counts.index, target_counts.values, 'o-', 
                        color=colors[0], linewidth=2.5, markersize=6, alpha=0.8)
            
            ax1.set_title('Class Counts', fontweight='bold', pad=15)
            ax1.set_xlabel('Target Classes')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.4)
            
            # Right plot: Pie chart (only if not too many classes)
            if len(target_counts) <= 8:
                wedges, texts, autotexts = ax2.pie(target_counts.values, labels=target_counts.index,
                                                  colors=colors, autopct='%1.1f%%', startangle=90,
                                                  explode=[0.05] * len(target_counts))
                ax2.set_title('Class Proportions', fontweight='bold', pad=15)
                
                # Improve text styling
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                # Show statistics instead
                ax2.axis('off')
                stats_text = f"""Class Statistics:
                
Total Classes: {len(target_counts)}
Total Samples: {len(df)}
Most Frequent: {target_counts.index[0]} ({target_counts.iloc[0]} samples)
Least Frequent: {target_counts.index[-1]} ({target_counts.iloc[-1]} samples)
                
Balance Ratio: {target_counts.min() / target_counts.max():.3f}
Gini Index: {1 - sum((target_counts / len(df)) ** 2):.3f}"""
                
                ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=11,
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                ax2.set_title('Distribution Statistics', fontweight='bold', pad=15)
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_target_distribution.png')
            plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Target distribution plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    @staticmethod
    def _plot_block_feature_evolution(df: pd.DataFrame, feature_cols: List[str], 
                                     generator_name: str, output_dir: str) -> str:
        """Plot feature evolution aggregated by blocks"""
        try:
            if 'block' not in df.columns:
                return AutoVisualizer._plot_feature_evolution(df, feature_cols, generator_name, output_dir)
            
            # Select up to 6 features for visualization
            selected_features = feature_cols[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(f'{generator_name} - Feature Evolution by Blocks', fontsize=14, fontweight='bold')
            axes = axes.flatten()
            
            # Get block statistics and map to sequential integers
            original_blocks = sorted(df['block'].unique())
            block_mapping = {orig: i+1 for i, orig in enumerate(original_blocks)}
            blocks = list(range(1, len(original_blocks) + 1))  # 1, 2, 3, ...
            colors = sns.color_palette("husl", len(blocks))
            
            for i, feature in enumerate(selected_features):
                if i >= 6:
                    break
                    
                ax = axes[i]
                
                if df[feature].dtype in ['int64', 'float64']:
                    # Numeric feature - show block statistics
                    block_stats = df.groupby('block')[feature].agg(['mean', 'std', 'min', 'max']).reset_index()
                    # Map original block IDs to sequential integers
                    block_stats['block_seq'] = block_stats['block'].map(block_mapping)
                    
                    # Plot mean with error bars (std)
                    ax.errorbar(block_stats['block_seq'], block_stats['mean'], yerr=block_stats['std'], 
                               fmt='o-', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                               color=colors[i % len(colors)], alpha=0.8, label='Mean ± Std')
                    
                    # Add min/max range as shaded area
                    ax.fill_between(block_stats['block_seq'], block_stats['min'], block_stats['max'], 
                                   alpha=0.2, color=colors[i % len(colors)], label='Min-Max Range')
                    
                    ax.set_title(f'{feature}', fontweight='bold', pad=10)
                    ax.set_xlabel('Block')
                    ax.set_ylabel('Value')
                    ax.set_xticks(blocks)
                    ax.grid(True, alpha=0.4)
                    ax.legend()
                    
                    # Add trend annotation
                    if len(blocks) > 1:
                        trend = 'increasing' if block_stats['mean'].iloc[-1] > block_stats['mean'].iloc[0] else 'decreasing'
                        trend_color = 'green' if trend == 'increasing' else 'red'
                        ax.text(0.02, 0.98, f'Trend: {trend}', transform=ax.transAxes, 
                               verticalalignment='top', color=trend_color, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                else:
                    # Categorical feature - show mode distribution across blocks
                    block_modes = []
                    block_diversity = []
                    
                    for orig_block in original_blocks:
                        block_data = df[df['block'] == orig_block][feature]
                        mode_val = block_data.mode().iloc[0] if len(block_data.mode()) > 0 else 'Unknown'
                        diversity = len(block_data.unique()) / len(block_data) if len(block_data) > 0 else 0
                        
                        block_modes.append(str(mode_val))
                        block_diversity.append(diversity)
                    
                    # Create dual y-axis plot
                    ax2 = ax.twinx()
                    
                    # Plot mode changes (categorical)
                    unique_modes = sorted(list(set(block_modes)))
                    mode_to_num = {mode: i for i, mode in enumerate(unique_modes)}
                    mode_nums = [mode_to_num[mode] for mode in block_modes]
                    
                    ax.plot(blocks, mode_nums, 'o-', color=colors[i % len(colors)], 
                           linewidth=2.5, markersize=8, alpha=0.8, label='Mode')
                    ax.set_ylabel('Dominant Mode')
                    ax.set_yticks(range(len(unique_modes)))
                    ax.set_yticklabels(unique_modes)
                    
                    # Plot diversity (numeric)
                    ax2.bar(blocks, block_diversity, alpha=0.3, color='gray', label='Diversity')
                    ax2.set_ylabel('Value Diversity', color='gray')
                    ax2.tick_params(axis='y', labelcolor='gray')
                    
                    ax.set_title(f'{feature} (Mode & Diversity)', fontweight='bold', pad=10)
                    ax.set_xlabel('Block')
                    ax.set_xticks(blocks)
                    ax.grid(True, alpha=0.4)
                    
                    # Combine legends
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Hide unused subplots
            for i in range(len(selected_features), 6):
                axes[i].axis('off')
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_block_feature_evolution.png')
            plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Block feature evolution plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    @staticmethod
    def _plot_block_target_distribution(df: pd.DataFrame, generator_name: str, output_dir: str) -> str:
        """Plot target distribution across blocks"""
        try:
            if 'block' not in df.columns or 'target' not in df.columns:
                return AutoVisualizer._plot_target_distribution(df, generator_name, output_dir)
            
            original_blocks = sorted(df['block'].unique())
            block_mapping = {orig: i+1 for i, orig in enumerate(original_blocks)}
            blocks = list(range(1, len(original_blocks) + 1))  # 1, 2, 3, ...
            targets = sorted(df['target'].unique())
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{generator_name} - Target Distribution Across Blocks', fontsize=14, fontweight='bold')
            
            # 1. Stacked bar chart showing distribution per block
            ax1 = axes[0, 0]
            block_target_counts = df.groupby(['block', 'target']).size().unstack(fill_value=0)
            # Map original block IDs to sequential integers
            block_target_counts.index = [block_mapping[idx] for idx in block_target_counts.index]
            
            colors = sns.color_palette("Set3", len(targets))
            bottom = np.zeros(len(blocks))
            
            for i, target in enumerate(targets):
                if target in block_target_counts.columns:
                    values = block_target_counts[target].values
                    ax1.bar(block_target_counts.index, values, bottom=bottom, label=f'Class {target}', 
                           color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)
                    bottom += values
            
            ax1.set_title('Class Distribution per Block', fontweight='bold', pad=15)
            ax1.set_xlabel('Block')
            ax1.set_ylabel('Sample Count')
            ax1.set_xticks(blocks)
            ax1.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.4)
            
            # 2. Proportional distribution (percentage)
            ax2 = axes[0, 1]
            block_target_pcts = df.groupby(['block', 'target']).size().unstack(fill_value=0)
            block_target_pcts = block_target_pcts.div(block_target_pcts.sum(axis=1), axis=0) * 100
            # Map original block IDs to sequential integers
            block_target_pcts.index = [block_mapping[idx] for idx in block_target_pcts.index]
            
            bottom = np.zeros(len(block_target_pcts))
            for i, target in enumerate(targets):
                if target in block_target_pcts.columns:
                    values = block_target_pcts[target].values
                    ax2.bar(block_target_pcts.index, values, bottom=bottom, label=f'Class {target}', 
                           color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)
                    bottom += values
            
            ax2.set_title('Class Proportions per Block (%)', fontweight='bold', pad=15)
            ax2.set_xlabel('Block')
            ax2.set_ylabel('Percentage')
            ax2.set_xticks(blocks)
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.4)
            
            # 3. Class evolution trends
            ax3 = axes[1, 0]
            for i, target in enumerate(targets):
                if target in block_target_counts.columns:
                    values = block_target_counts[target].values
                    ax3.plot(block_target_counts.index, values, 'o-', linewidth=2.5, markersize=6, 
                            color=colors[i], alpha=0.8, label=f'Class {target}')
            
            ax3.set_title('Class Count Trends', fontweight='bold', pad=15)
            ax3.set_xlabel('Block')
            ax3.set_ylabel('Sample Count')
            ax3.set_xticks(blocks)
            ax3.legend(title='Classes')
            ax3.grid(True, alpha=0.4)
            
            # 4. Class balance analysis
            ax4 = axes[1, 1]
            
            # Calculate balance metrics per block
            balance_metrics = []
            gini_indices = []
            
            for orig_block in original_blocks:
                block_data = df[df['block'] == orig_block]['target']
                counts = block_data.value_counts()
                
                # Balance ratio (min/max)
                balance_ratio = counts.min() / counts.max() if len(counts) > 1 else 1.0
                balance_metrics.append(balance_ratio)
                
                # Gini index
                proportions = counts / len(block_data)
                gini = 1 - sum(proportions ** 2)
                gini_indices.append(gini)
            
            # Dual y-axis for balance metrics
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(blocks, balance_metrics, 'o-', linewidth=2.5, markersize=6, 
                            color='blue', alpha=0.8, label='Balance Ratio (min/max)')
            line2 = ax4_twin.plot(blocks, gini_indices, 's-', linewidth=2.5, markersize=6, 
                                 color='red', alpha=0.8, label='Gini Index')
            
            ax4.set_title('Class Balance Analysis', fontweight='bold', pad=15)
            ax4.set_xlabel('Block')
            ax4.set_ylabel('Balance Ratio', color='blue')
            ax4.set_xticks(blocks)
            ax4_twin.set_ylabel('Gini Index', color='red')
            
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_twin.tick_params(axis='y', labelcolor='red')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='center right')
            
            ax4.grid(True, alpha=0.4)
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_block_target_distribution.png')
            plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Block target distribution plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    @staticmethod
    def _plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str], 
                                generator_name: str, output_dir: str) -> str:
        """Plot beautiful correlation matrix with modern styling"""
        try:
            if len(numeric_cols) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create single beautiful heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.suptitle(f'{generator_name} - Feature Correlation Matrix', fontsize=16, fontweight='bold', y=0.95)
            
            # Create a mask for the upper triangle (optional)
            n_features = len(corr_matrix)
            show_full_matrix = n_features <= 8  # Show full matrix for smaller datasets
            
            if show_full_matrix:
                mask = None
                title_text = 'Feature Correlation Matrix'
            else:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                title_text = 'Feature Correlation Matrix (Lower Triangle)'
            
            # Choose colormap and styling
            cmap = 'RdBu_r'  # Red-white-blue diverging colormap
            
            # Create the heatmap with beautiful styling
            heatmap = sns.heatmap(
                corr_matrix, 
                mask=mask,
                annot=True, 
                cmap=cmap, 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={
                    'shrink': 0.8, 
                    'label': 'Correlation Coefficient',
                    'aspect': 20,
                    'pad': 0.02
                },
                ax=ax, 
                linewidths=1.0,
                linecolor='white',
                annot_kws={
                    'size': 11 if n_features <= 6 else 9 if n_features <= 10 else 7,
                    'weight': 'bold'
                },
                vmin=-1, 
                vmax=1
            )
            
            # Customize the heatmap
            ax.set_title(title_text, fontweight='bold', pad=20, fontsize=14)
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
            
            # Add feature count and statistics as text
            n_pairs = len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2
            
            # Calculate statistics excluding diagonal
            off_diagonal_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            mean_abs_corr = np.abs(off_diagonal_corr).mean()
            max_abs_corr = np.abs(off_diagonal_corr).max()
            strong_corr_count = np.sum(np.abs(off_diagonal_corr) > 0.7)
            moderate_corr_count = np.sum((np.abs(off_diagonal_corr) > 0.5) & (np.abs(off_diagonal_corr) <= 0.7))
 
            
            # Position the text box
            if show_full_matrix:
                bbox_x, bbox_y = 1.02, 0.98
            else:
                bbox_x, bbox_y = 0.02, 0.98

            
            # Improve colorbar
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
            

            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_correlation.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Correlation plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    @staticmethod
    def _plot_pca_analysis(df: pd.DataFrame, numeric_cols: List[str], 
                          generator_name: str, output_dir: str) -> str:
        """Plot PCA analysis with improved styling and insights"""
        try:
            if len(numeric_cols) < 2:
                return None
            
            # Prepare data
            X = df[numeric_cols].fillna(df[numeric_cols].mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA with more components for analysis
            n_components = min(len(numeric_cols), 6)  # Up to 6 components
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{generator_name} - Principal Component Analysis', fontsize=14, fontweight='bold')
            
            # 1. 2D PCA scatter plot
            ax1 = axes[0, 0]
            
            if 'target' in df.columns:
                # Color by target
                unique_targets = sorted(df['target'].unique())
                colors = sns.color_palette("Set2", len(unique_targets))
                
                for i, target in enumerate(unique_targets):
                    mask = df['target'] == target
                    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=[colors[i]], label=f'Class {target}', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
                ax1.legend(title='Target Classes', frameon=True, fancybox=True, shadow=True)
            else:
                ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50, c=sns.color_palette()[0], edgecolors='white', linewidth=0.5)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} of variance)', fontweight='bold')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} of variance)', fontweight='bold')
            ax1.set_title('2D PCA Projection', fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.4)
            
            # Add ellipses for target classes if available
            if 'target' in df.columns and len(unique_targets) <= 5:
                from matplotlib.patches import Ellipse
                for i, target in enumerate(unique_targets):
                    mask = df['target'] == target
                    if np.sum(mask) > 2:  # Need at least 3 points for ellipse
                        data_points = X_pca[mask, :2]
                        mean = np.mean(data_points, axis=0)
                        cov = np.cov(data_points.T)
                        
                        # Calculate ellipse parameters
                        eigenvals, eigenvecs = np.linalg.eig(cov)
                        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                        width, height = 2 * 2 * np.sqrt(eigenvals)  # 2 standard deviations
                        
                        ellipse = Ellipse(mean, width, height, angle=angle, 
                                        facecolor=colors[i], alpha=0.1, edgecolor=colors[i], linewidth=2)
                        ax1.add_patch(ellipse)
            
            # 2. Explained variance plot
            ax2 = axes[0, 1]
            
            cumulative_var = np.cumsum(pca.explained_variance_ratio_)
            
            # Bar plot of individual variance
            bars = ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                          pca.explained_variance_ratio_, alpha=0.7, color=sns.color_palette("viridis", len(pca.explained_variance_ratio_)))
            
            # Line plot of cumulative variance
            ax2_twin = ax2.twinx()
            ax2_twin.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                         'ro-', linewidth=2.5, markersize=6, color='red', label='Cumulative')
            
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Explained Variance Ratio', color='blue')
            ax2_twin.set_ylabel('Cumulative Variance', color='red')
            ax2.set_title('Variance Explained by Components', fontweight='bold', pad=15)
            
            # Add percentage labels
            for i, (bar, var_ratio) in enumerate(zip(bars, pca.explained_variance_ratio_)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{var_ratio:.2%}', ha='center', va='bottom', fontsize=9)
            
            ax2.grid(True, alpha=0.4)
            ax2_twin.legend(loc='lower right')
            
            # 3. Feature contributions (loadings)
            ax3 = axes[1, 0]
            
            # Calculate loadings (feature contributions to PC1 and PC2)
            loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
            
            # Create biplot
            for i, feature in enumerate(numeric_cols):
                ax3.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                         head_width=0.03, head_length=0.03, fc='red', ec='red', alpha=0.8)
                ax3.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, 
                        fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax3.set_xlabel('PC1 Loading')
            ax3.set_ylabel('PC2 Loading')
            ax3.set_title('Feature Contributions (Loadings)', fontweight='bold', pad=15)
            ax3.grid(True, alpha=0.4)
            ax3.axhline(y=0, color='k', linewidth=0.5)
            ax3.axvline(x=0, color='k', linewidth=0.5)
            
            # Set equal aspect ratio and limits
            max_range = max(np.abs(loadings).max() * 1.2, 0.1)
            ax3.set_xlim(-max_range, max_range)
            ax3.set_ylim(-max_range, max_range)
            ax3.set_aspect('equal')
            
            # 4. PCA Statistics and Interpretation
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Calculate additional statistics
            total_var_2pc = cumulative_var[1] if len(cumulative_var) > 1 else cumulative_var[0]
            kaiser_criterion = np.sum(pca.explained_variance_ > 1.0) if len(pca.explained_variance_) > 0 else 0
            
            stats_text = f"""PCA ANALYSIS SUMMARY
            
Dataset Dimensions:
  • Original Features: {len(numeric_cols)}
  • Samples: {len(df):,}
  • Components Computed: {n_components}
            
Variance Analysis:
  • PC1 Variance: {pca.explained_variance_ratio_[0]:.2%}
  • PC2 Variance: {pca.explained_variance_ratio_[1]:.2%} 
  • First 2 PCs Total: {total_var_2pc:.2%}
  • Kaiser Criterion (λ>1): {kaiser_criterion} components
            
Interpretation:
  • Dimensionality Reduction: {'Good' if total_var_2pc > 0.6 else 'Moderate' if total_var_2pc > 0.4 else 'Poor'}
  • Data Complexity: {'Low' if pca.explained_variance_ratio_[0] > 0.5 else 'Medium' if pca.explained_variance_ratio_[0] > 0.3 else 'High'}
            
Top Contributing Features (PC1):"""
            
            # Find top contributing features to PC1
            pc1_contributions = np.abs(pca.components_[0])
            top_features_idx = np.argsort(pc1_contributions)[-3:][::-1]  # Top 3
            
            for i, idx in enumerate(top_features_idx):
                feature_name = numeric_cols[idx]
                contribution = pc1_contributions[idx]
                stats_text += f"\n  {i+1}. {feature_name}: {contribution:.3f}"
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            ax4.set_title('Analysis Summary', fontweight='bold', pad=15)
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_pca.png')
            plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: PCA plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    @staticmethod
    def _plot_categorical_features(df: pd.DataFrame, categorical_cols: List[str], 
                                  generator_name: str, output_dir: str) -> str:
        """Plot categorical features analysis with improved styling"""
        try:
            if not categorical_cols:
                return None
                
            n_features = min(len(categorical_cols), 4)
            selected_features = categorical_cols[:n_features]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{generator_name} - Categorical Features Analysis', fontsize=14, fontweight='bold')
            axes = axes.flatten()
            
            colors = sns.color_palette("Set3", 12)  # Rich color palette
            
            for i, feature in enumerate(selected_features):
                if i >= 4:
                    break
                    
                ax = axes[i]
                
                value_counts = df[feature].value_counts()
                total_count = len(df)
                
                if len(value_counts) <= 12:
                    # Bar plot with percentages
                    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                                 color=colors[:len(value_counts)], alpha=0.8, 
                                 edgecolor='white', linewidth=1.5)
                    
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    
                    # Add value and percentage labels
                    for bar, value in zip(bars, value_counts.values):
                        percentage = (value / total_count) * 100
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + 0.01*max(value_counts.values),
                               f'{value}\\n({percentage:.1f}%)', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                else:
                    # Show only top 10 + "Others"
                    top_values = value_counts.head(10)
                    others_count = value_counts.iloc[10:].sum()
                    
                    # Create combined data
                    combined_values = list(top_values.values) + [others_count]
                    combined_labels = list(top_values.index) + ['Others']
                    
                    bars = ax.bar(range(len(combined_values)), combined_values, 
                                 color=colors[:len(combined_values)], alpha=0.8,
                                 edgecolor='white', linewidth=1.5)
                    
                    ax.set_xticks(range(len(combined_values)))
                    ax.set_xticklabels(combined_labels, rotation=45, ha='right')
                    
                    # Add value and percentage labels
                    for bar, value in zip(bars, combined_values):
                        percentage = (value / total_count) * 100
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + 0.01*max(combined_values),
                               f'{value}\\n({percentage:.1f}%)', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Calculate and display statistics
                unique_count = len(value_counts)
                mode_value = value_counts.index[0]
                mode_count = value_counts.iloc[0]
                mode_percentage = (mode_count / total_count) * 100
                
                # Diversity metrics
                entropy = -np.sum((value_counts / total_count) * np.log2(value_counts / total_count + 1e-10))
                max_entropy = np.log2(unique_count) if unique_count > 1 else 1
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                ax.set_title(f'{feature}\\nUnique: {unique_count}, Mode: {mode_value} ({mode_percentage:.1f}%)\\nDiversity: {normalized_entropy:.2f}', 
                           fontweight='bold', pad=15)
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.4)
                
                # Add statistics box
                stats_text = f'Total: {total_count}\\nEntropy: {entropy:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Hide unused subplots
            for i in range(len(selected_features), 4):
                axes[i].axis('off')
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_categorical_features.png')
            plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Warning: Categorical features plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    @staticmethod
    def _plot_detailed_variable_evolution(df: pd.DataFrame, feature_cols: List[str], 
                                        generator_name: str, output_dir: str) -> str:
        """Plot detailed evolution of each numeric variable across all instances (with or without blocks)"""
        try:
            # Check if we have blocks
            has_blocks = 'block' in df.columns
                
            # Filter only numeric features
            numeric_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
            
            if not numeric_cols:
                print("Warning: No numeric features found for detailed evolution plot")
                return None
                
            # Create a comprehensive figure showing all numeric variables
            n_vars = len(numeric_cols)
            n_cols = 2  # Two columns layout
            n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            title_suffix = "All Instances & Blocks" if has_blocks else "All Instances (Sequential)"
            fig.suptitle(f'{generator_name} - Detailed Variable Evolution ({title_suffix})', 
                        fontsize=16, fontweight='bold')
            
            # Handle single subplot case
            if n_vars == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_vars > 1 else [axes]
            else:
                axes = axes.flatten()
            
            # Get unique blocks and colors (only if blocks exist)
            if has_blocks:
                unique_blocks = sorted(df['block'].unique())
                block_colors = sns.color_palette("husl", len(unique_blocks))
            else:
                unique_blocks = []
                block_colors = []
            
            # Add instance index for continuous evolution tracking
            df_viz = df.copy()
            df_viz['instance_idx'] = range(len(df_viz))
            
            for i, var in enumerate(numeric_cols):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Plot 1: Raw evolution with block boundaries highlighted
                y_values = df_viz[var].values
                x_values = df_viz['instance_idx'].values
                
                # Plot the continuous line
                ax.plot(x_values, y_values, color='steelblue', alpha=0.7, linewidth=1.5, 
                       label='Raw Values', zorder=1)
                
                # Add rolling mean for trend
                if len(df_viz) > 10:
                    window = max(5, len(df_viz) // 20)
                    rolling_mean = df_viz[var].rolling(window=window, center=True).mean()
                    ax.plot(x_values, rolling_mean, color='darkred', linewidth=3, 
                           alpha=0.8, label='Trend', zorder=2)
                
                # Add block-specific or instance-based analysis
                if has_blocks:
                    # Highlight block boundaries and add block-specific statistics
                    block_start_idx = 0
                    for j, block in enumerate(unique_blocks):
                        block_data = df_viz[df_viz['block'] == block]
                        block_end_idx = block_start_idx + len(block_data)
                        
                        # Vertical line for block boundary (except first)
                        if j > 0:
                            ax.axvline(x=block_start_idx, color='gray', linestyle='--', 
                                      alpha=0.6, linewidth=1)
                        
                        # Add block statistics as colored regions
                        block_mean = block_data[var].mean()
                        block_std = block_data[var].std()
                        block_x_values = block_data['instance_idx'].values
                        
                        # Confidence interval shading
                        if len(block_x_values) > 1:
                            ax.fill_between(block_x_values, 
                                          block_mean - block_std, 
                                          block_mean + block_std,
                                          alpha=0.2, color=block_colors[j], 
                                          label=f'Block {block} (μ±σ)' if j < 5 else '')
                        
                        # Block mean line
                        ax.hlines(y=block_mean, xmin=block_start_idx, xmax=block_end_idx-1,
                                 colors=block_colors[j], linewidth=2, alpha=0.8, zorder=3)
                        
                        # Add block label
                        mid_x = (block_start_idx + block_end_idx - 1) / 2
                        ax.text(mid_x, ax.get_ylim()[1] * 0.95, f'B{block}', 
                               ha='center', va='top', fontweight='bold', 
                               color=block_colors[j], fontsize=10,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor=block_colors[j], alpha=0.8))
                        
                        block_start_idx = block_end_idx
                else:
                    # For sequential data without blocks, add quartile markers
                    total_instances = len(df_viz)
                    quartile_positions = [total_instances//4, total_instances//2, 3*total_instances//4]
                    quartile_labels = ['Q1', 'Q2 (Median)', 'Q3']
                    
                    for pos, label in zip(quartile_positions, quartile_labels):
                        if pos < total_instances:
                            ax.axvline(x=pos, color='lightblue', linestyle=':', 
                                      alpha=0.7, linewidth=1)
                            ax.text(pos, ax.get_ylim()[1] * 0.95, label, 
                                   ha='center', va='top', fontsize=9, 
                                   color='navy', alpha=0.8)
                
                # Statistical annotations
                overall_mean = df_viz[var].mean()
                overall_std = df_viz[var].std()
                min_val = df_viz[var].min()
                max_val = df_viz[var].max()
                
                # Add horizontal reference lines
                ax.axhline(y=overall_mean, color='black', linestyle=':', alpha=0.6, 
                          linewidth=1, label='Overall Mean')
                
                # Calculate trend (blocks or sequential)
                if has_blocks and len(unique_blocks) > 1:
                    # Calculate trend across blocks
                    block_means = [df_viz[df_viz['block'] == block][var].mean() 
                                  for block in unique_blocks]
                    from scipy import stats as scipy_stats
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        range(len(block_means)), block_means)
                    trend_direction = "↗" if slope > 0 else "↘" if slope < 0 else "→"
                    trend_strength = "Strong" if abs(r_value) > 0.7 else "Moderate" if abs(r_value) > 0.4 else "Weak"
                elif not has_blocks and len(df_viz) > 10:
                    # Calculate trend across sequential instances
                    from scipy import stats as scipy_stats
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        x_values, y_values)
                    trend_direction = "↗" if slope > 0 else "↘" if slope < 0 else "→"
                    trend_strength = "Strong" if abs(r_value) > 0.7 else "Moderate" if abs(r_value) > 0.4 else "Weak"
                else:
                    trend_direction = "→"
                    trend_strength = "No trend"
                    r_value = 0
                
                # Title with summary statistics
                ax.set_title(f'{var}\n'
                           f'Trend: {trend_strength} {trend_direction} (R²={r_value**2:.3f}) | '
                           f'Range: [{min_val:.2f}, {max_val:.2f}] | σ: {overall_std:.3f}',
                           fontweight='bold', pad=15, fontsize=11)
                
                ax.set_xlabel('Instance Index (Sequential)')
                ax.set_ylabel(f'{var} Value')
                ax.grid(True, alpha=0.3)
                
                # Legend (limit to avoid clutter)
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 8:  # Limit legend entries
                    ax.legend(handles[:8], labels[:8], loc='upper right', fontsize=8)
                else:
                    ax.legend(loc='upper right', fontsize=8)
                
                # Add summary statistics box
                if has_blocks:
                    stats_text = f'μ: {overall_mean:.3f}\nσ: {overall_std:.3f}\nBlocks: {len(unique_blocks)}'
                else:
                    stats_text = f'μ: {overall_mean:.3f}\nσ: {overall_std:.3f}\nInstances: {len(df_viz)}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9, fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                               alpha=0.8, edgecolor='orange'))
            
            # Hide unused subplots
            for j in range(n_vars, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            filename = os.path.join(output_dir, f'{generator_name}_detailed_variable_evolution.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Generated detailed variable evolution plot with {n_vars} numeric variables")
            return filename
            
        except Exception as e:
            print(f"Warning: Detailed variable evolution plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    @staticmethod
    def _plot_categorical_mode_evolution(df: pd.DataFrame, categorical_cols: List[str], 
                                       generator_name: str, output_dir: str, has_blocks: bool) -> str:
        """Plot mode evolution of categorical variables over time/blocks"""
        try:
            if not categorical_cols:
                return None
            
            # Select up to 4 categorical features for visualization
            selected_features = categorical_cols[:4]
            
            # Create subplots
            n_cols = 2
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
            
            if has_blocks:
                fig.suptitle(f'{generator_name} - Categorical Mode Evolution by Blocks', fontsize=14, fontweight='bold')
            else:
                fig.suptitle(f'{generator_name} - Categorical Mode Evolution Over Time', fontsize=14, fontweight='bold')
            
            # Ensure axes is always a list
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            colors = sns.color_palette("husl", max(10, len(selected_features)))
            
            for i, feature in enumerate(selected_features):
                ax = axes[i]
                
                if has_blocks and 'block' in df.columns:
                    # Block-based mode evolution
                    blocks = sorted(df['block'].unique())
                    modes = []
                    mode_counts = []
                    total_counts = []
                    
                    for block in blocks:
                        block_data = df[df['block'] == block][feature]
                        if len(block_data) > 0:
                            value_counts = block_data.value_counts()
                            mode = value_counts.index[0] if len(value_counts) > 0 else "N/A"
                            mode_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
                            total_count = len(block_data)
                        else:
                            mode = "N/A"
                            mode_count = 0
                            total_count = 0
                            
                        modes.append(str(mode))
                        mode_counts.append(mode_count)
                        total_counts.append(total_count)
                    
                    # Plot mode frequency by block
                    block_labels = [f"Block {b}" for b in blocks]
                    percentages = [mc/tc*100 if tc > 0 else 0 for mc, tc in zip(mode_counts, total_counts)]
                    
                    bars = ax.bar(block_labels, percentages, color=colors[i % len(colors)], alpha=0.7)
                    
                    # Add mode labels on bars
                    for bar, mode, percentage in zip(bars, modes, percentages):
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + max(percentages) * 0.02,
                               f'{mode}\\n({percentage:.1f}%)', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    ax.set_title(f'{feature} - Mode per Block', fontweight='bold', pad=15)
                    ax.set_ylabel('Mode Frequency (%)')
                    ax.set_ylim(0, max(percentages) * 1.2 if percentages else 100)
                    
                else:
                    # Sequential mode evolution (rolling window approach)
                    window_size = max(50, len(df) // 20)  # Adaptive window size
                    x_positions = []
                    mode_values = []
                    mode_percentages = []
                    
                    for start_idx in range(0, len(df) - window_size + 1, window_size // 2):
                        end_idx = start_idx + window_size
                        window_data = df.iloc[start_idx:end_idx][feature]
                        
                        if len(window_data) > 0:
                            value_counts = window_data.value_counts()
                            mode = value_counts.index[0] if len(value_counts) > 0 else "N/A"
                            mode_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
                            mode_percentage = (mode_count / len(window_data)) * 100
                        else:
                            mode = "N/A"
                            mode_percentage = 0
                        
                        x_positions.append(start_idx + window_size // 2)
                        mode_values.append(str(mode))
                        mode_percentages.append(mode_percentage)
                    
                    # Plot mode percentages over time
                    ax.plot(x_positions, mode_percentages, marker='o', linewidth=2.5, 
                           markersize=6, color=colors[i % len(colors)], alpha=0.8)
                    
                    # Add mode labels at key points
                    for j in range(0, len(x_positions), max(1, len(x_positions) // 5)):
                        ax.annotate(mode_values[j], 
                                   (x_positions[j], mode_percentages[j]),
                                   xytext=(0, 15), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
                    
                    ax.set_title(f'{feature} - Mode Evolution', fontweight='bold', pad=15)
                    ax.set_xlabel('Instance')
                    ax.set_ylabel('Mode Frequency (%)')
                    ax.set_ylim(0, 100)
                
                ax.grid(True, alpha=0.4)
                ax.tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for j in range(len(selected_features), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            if has_blocks:
                filename = f"{generator_name}_categorical_mode_evolution_blocks.png"
            else:
                filename = f"{generator_name}_categorical_mode_evolution.png"
                
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Warning: Categorical mode evolution plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    @staticmethod
    def _plot_boxplots(df: pd.DataFrame, feature_cols: List[str], 
                      generator_name: str, output_dir: str, has_blocks: bool) -> str:
        """Create box plots for numeric variables showing distribution patterns"""
        try:
            # Select numeric columns only
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return None
            
            # Select up to 8 numeric features for visualization
            selected_features = numeric_cols[:8]
            
            # Create subplots
            n_cols = 4
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            
            if has_blocks:
                fig.suptitle(f'{generator_name} - Box Plots by Blocks', fontsize=16, fontweight='bold', y=0.98)
            else:
                fig.suptitle(f'{generator_name} - Box Plots Distribution', fontsize=16, fontweight='bold', y=0.98)
            
            # Ensure axes is always a list for consistent indexing
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            colors = sns.color_palette("Set2", max(8, len(selected_features)))
            
            for i, feature in enumerate(selected_features):
                ax = axes[i]
                
                if has_blocks and 'block' in df.columns:
                    # Box plot comparing distributions across blocks
                    blocks = sorted(df['block'].unique())
                    
                    # Prepare data for each block
                    box_data = []
                    labels = []
                    
                    for block in blocks:
                        block_data = df[df['block'] == block][feature].dropna()
                        if len(block_data) > 0:
                            box_data.append(block_data.values)
                            labels.append(f'Block {block}')
                    
                    if box_data:
                        # Create box plot
                        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                                       showmeans=True, meanline=False, 
                                       notch=True, whis=1.5)
                        
                        # Color each box differently
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        # Style the plot elements
                        for element in ['whiskers', 'fliers', 'medians', 'caps']:
                            plt.setp(bp[element], color='darkblue', linewidth=1.5)
                        plt.setp(bp['means'], color='red', marker='D', markersize=6)
                        
                        # Add statistics annotations
                        for j, block_data in enumerate(box_data):
                            stats_text = f'μ={np.mean(block_data):.2f}\\nσ={np.std(block_data):.2f}'
                            ax.text(j+1, np.max(block_data) * 1.1, stats_text, 
                                   ha='center', va='bottom', fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        
                        ax.set_title(f'{feature} by Blocks', fontweight='bold', pad=15)
                        ax.set_ylabel('Value')
                        ax.grid(True, alpha=0.3)
                        
                else:
                    # Single box plot for overall distribution
                    feature_data = df[feature].dropna()
                    
                    if len(feature_data) > 0:
                        bp = ax.boxplot([feature_data.values], labels=[feature], 
                                       patch_artist=True, showmeans=True, meanline=False,
                                       notch=True, whis=1.5)
                        
                        # Color the box
                        bp['boxes'][0].set_facecolor(colors[i % len(colors)])
                        bp['boxes'][0].set_alpha(0.7)
                        
                        # Style elements
                        for element in ['whiskers', 'fliers', 'medians', 'caps']:
                            plt.setp(bp[element], color='darkblue', linewidth=1.5)
                        plt.setp(bp['means'], color='red', marker='D', markersize=6)
                        
                        # Add statistics
                        stats = {
                            'Mean': np.mean(feature_data),
                            'Median': np.median(feature_data),
                            'Std': np.std(feature_data),
                            'IQR': np.percentile(feature_data, 75) - np.percentile(feature_data, 25),
                            'Outliers': len(feature_data[(feature_data < np.percentile(feature_data, 25) - 1.5 * (np.percentile(feature_data, 75) - np.percentile(feature_data, 25))) | 
                                                        (feature_data > np.percentile(feature_data, 75) + 1.5 * (np.percentile(feature_data, 75) - np.percentile(feature_data, 25)))])
                        }
                        
                        stats_text = f"μ={stats['Mean']:.2f}\\nMed={stats['Median']:.2f}\\nσ={stats['Std']:.2f}\\nIQR={stats['IQR']:.2f}\\nOutliers={stats['Outliers']}"
                        ax.text(1.3, np.max(feature_data) * 0.9, stats_text, 
                               ha='left', va='top', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
                        
                        ax.set_title(f'{feature} Distribution', fontweight='bold', pad=15)
                        ax.set_ylabel('Value')
                        ax.grid(True, alpha=0.3)
                    
                # Rotate x-axis labels if needed
                ax.tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for j in range(len(selected_features), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            if has_blocks:
                filename = f"{generator_name}_boxplots_blocks.png"
            else:
                filename = f"{generator_name}_boxplots.png"
                
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Generated box plots for {len(selected_features)} numeric variables")
            return filepath
            
        except Exception as e:
            print(f"Warning: Box plots failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    @staticmethod
    def _plot_histograms(df: pd.DataFrame, feature_cols: List[str], 
                        generator_name: str, output_dir: str, has_blocks: bool) -> str:
        """Create histograms for numeric variables showing distribution shapes"""
        try:
            # Select numeric columns only
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return None
            
            # Select up to 8 numeric features for visualization
            selected_features = numeric_cols[:8]
            
            # Create subplots
            n_cols = 4
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            
            if has_blocks:
                fig.suptitle(f'{generator_name} - Histograms by Blocks', fontsize=16, fontweight='bold', y=0.98)
            else:
                fig.suptitle(f'{generator_name} - Distribution Histograms', fontsize=16, fontweight='bold', y=0.98)
            
            # Ensure axes is always a list for consistent indexing
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            colors = sns.color_palette("viridis", max(10, len(selected_features)))
            
            for i, feature in enumerate(selected_features):
                ax = axes[i]
                feature_data = df[feature].dropna()
                
                if len(feature_data) == 0:
                    ax.set_visible(False)
                    continue
                
                if has_blocks and 'block' in df.columns:
                    # Overlapping histograms for each block
                    blocks = sorted(df['block'].unique())
                    block_colors = sns.color_palette("Set1", len(blocks))
                    
                    # Calculate global range for consistent x-axis
                    all_data = df[feature].dropna()
                    if len(all_data) > 0:
                        data_range = (all_data.min(), all_data.max())
                        bins = np.linspace(data_range[0], data_range[1], min(30, max(10, len(all_data) // 20)))
                    else:
                        bins = 20
                    
                    # Plot histogram for each block
                    for j, block in enumerate(blocks):
                        block_data = df[df['block'] == block][feature].dropna()
                        if len(block_data) > 0:
                            ax.hist(block_data.values, bins=bins, alpha=0.6, 
                                   label=f'Block {block}', color=block_colors[j % len(block_colors)],
                                   density=True, edgecolor='white', linewidth=0.5)
                    
                    # Add statistics for each block
                    stats_text = []
                    for j, block in enumerate(blocks):
                        block_data = df[df['block'] == block][feature].dropna()
                        if len(block_data) > 0:
                            mean_val = np.mean(block_data)
                            std_val = np.std(block_data)
                            stats_text.append(f'Block {block}: μ={mean_val:.2f}, σ={std_val:.2f}')
                    
                    if stats_text:
                        ax.text(0.02, 0.98, '\\n'.join(stats_text), transform=ax.transAxes,
                               verticalalignment='top', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
                    
                    ax.legend(loc='upper right', fontsize=9)
                    ax.set_title(f'{feature} - Distribution by Blocks', fontweight='bold', pad=15)
                    
                else:
                    # Single histogram for overall distribution
                    # Calculate optimal number of bins
                    n_bins = min(50, max(10, int(np.sqrt(len(feature_data)))))
                    
                    # Create histogram
                    n, bins, patches = ax.hist(feature_data.values, bins=n_bins, 
                                             color=colors[i % len(colors)], alpha=0.7,
                                             density=True, edgecolor='white', linewidth=0.5)
                    
                    # Add normal distribution overlay for comparison
                    mean_val = np.mean(feature_data)
                    std_val = np.std(feature_data)
                    x_norm = np.linspace(feature_data.min(), feature_data.max(), 100)
                    y_norm = ((1 / (np.sqrt(2 * np.pi) * std_val)) * 
                             np.exp(-0.5 * ((x_norm - mean_val) / std_val) ** 2))
                    ax.plot(x_norm, y_norm, 'r--', linewidth=2, alpha=0.8, label='Normal fit')
                    
                    # Calculate distribution statistics
                    from scipy.stats import skew, kurtosis
                    skewness = skew(feature_data)
                    kurt = kurtosis(feature_data)
                    
                    # Add comprehensive statistics
                    stats = {
                        'Mean': mean_val,
                        'Median': np.median(feature_data),
                        'Mode': float(feature_data.mode().iloc[0]) if len(feature_data.mode()) > 0 else mean_val,
                        'Std': std_val,
                        'Skew': skewness,
                        'Kurt': kurt,
                        'Min': feature_data.min(),
                        'Max': feature_data.max()
                    }
                    
                    stats_text = f"μ={stats['Mean']:.2f}\\nMed={stats['Median']:.2f}\\nMod={stats['Mode']:.2f}\\nσ={stats['Std']:.2f}\\nSkew={stats['Skew']:.2f}\\nKurt={stats['Kurt']:.2f}"
                    
                    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
                    
                    # Add interpretation text
                    interpretation = []
                    if abs(skewness) < 0.5:
                        interpretation.append("Symmetric")
                    elif skewness > 0.5:
                        interpretation.append("Right-skewed")
                    elif skewness < -0.5:
                        interpretation.append("Left-skewed")
                    
                    if kurt > 3:
                        interpretation.append("Heavy-tailed")
                    elif kurt < 3:
                        interpretation.append("Light-tailed")
                    else:
                        interpretation.append("Normal-tailed")
                    
                    if interpretation:
                        ax.text(0.02, 0.98, ' | '.join(interpretation), transform=ax.transAxes,
                               verticalalignment='top', fontsize=8, style='italic',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
                    
                    ax.legend(loc='upper right', fontsize=9)
                    ax.set_title(f'{feature} - Distribution Shape', fontweight='bold', pad=15)
                
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                
                # Color bars based on height for better visual appeal
                if not has_blocks or not 'block' in df.columns:
                    # Gradient coloring for single histogram
                    cm = plt.cm.viridis
                    for patch, height in zip(patches, n):
                        patch.set_facecolor(cm(height / max(n)))
            
            # Hide empty subplots
            for j in range(len(selected_features), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            if has_blocks:
                filename = f"{generator_name}_histograms_blocks.png"
            else:
                filename = f"{generator_name}_histograms.png"
                
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Generated histograms for {len(selected_features)} numeric variables")
            return filepath
            
        except Exception as e:
            print(f"Warning: Histograms failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    @staticmethod
    def _plot_dataset_comparison_summary(orig_df: pd.DataFrame, synth_df: pd.DataFrame, 
                                       generator_name: str, output_dir: str) -> str:
        """Create a summary comparison plot between original and synthetic datasets"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{generator_name} - Dataset Comparison Summary', fontsize=16, fontweight='bold')
            
            # 1. Dataset size comparison
            ax1 = axes[0, 0]
            datasets = ['Original', 'Synthetic']
            sizes = [len(orig_df), len(synth_df)]
            colors = ['#2E86AB', '#A23B72']
            bars = ax1.bar(datasets, sizes, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            ax1.set_title('Dataset Sizes', fontweight='bold', pad=15)
            ax1.set_ylabel('Number of Samples')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                        f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 2. Feature types comparison
            ax2 = axes[0, 1]
            orig_numeric = len(orig_df.select_dtypes(include=[np.number]).columns)
            orig_categorical = len(orig_df.select_dtypes(include=['object', 'category']).columns)
            synth_numeric = len(synth_df.select_dtypes(include=[np.number]).columns) 
            synth_categorical = len(synth_df.select_dtypes(include=['object', 'category']).columns)
            
            x = np.arange(len(datasets))
            width = 0.35
            
            ax2.bar(x - width/2, [orig_numeric, synth_numeric], width, label='Numeric', 
                   color='#F18F01', alpha=0.8)
            ax2.bar(x + width/2, [orig_categorical, synth_categorical], width, label='Categorical', 
                   color='#C73E1D', alpha=0.8)
            
            ax2.set_title('Feature Types', fontweight='bold', pad=15)
            ax2.set_ylabel('Number of Features')
            ax2.set_xticks(x)
            ax2.set_xticklabels(datasets)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Missing values comparison
            ax3 = axes[1, 0]
            orig_missing = orig_df.isnull().sum().sum()
            synth_missing = synth_df.isnull().sum().sum()
            missing_vals = [orig_missing, synth_missing]
            
            bars = ax3.bar(datasets, missing_vals, color=['#FF6B35', '#004E89'], alpha=0.8)
            ax3.set_title('Missing Values', fontweight='bold', pad=15)
            ax3.set_ylabel('Total Missing Values')
            ax3.grid(True, alpha=0.3)
            
            for bar, missing in zip(bars, missing_vals):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(missing_vals)*0.01,
                        f'{missing:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 4. Memory usage comparison
            ax4 = axes[1, 1]
            orig_memory = orig_df.memory_usage(deep=True).sum() / 1024  # KB
            synth_memory = synth_df.memory_usage(deep=True).sum() / 1024  # KB
            memory_usage = [orig_memory, synth_memory]
            
            bars = ax4.bar(datasets, memory_usage, color=['#7209B7', '#2D3748'], alpha=0.8)
            ax4.set_title('Memory Usage', fontweight='bold', pad=15)
            ax4.set_ylabel('Memory Usage (KB)')
            ax4.grid(True, alpha=0.3)
            
            for bar, memory in zip(bars, memory_usage):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(memory_usage)*0.01,
                        f'{memory:.1f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            filename = f"{generator_name}_dataset_comparison_summary.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"Warning: Dataset comparison summary plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    @staticmethod
    def _plot_numeric_comparison(orig_df: pd.DataFrame, synth_df: pd.DataFrame, 
                               numeric_cols: List[str], generator_name: str, output_dir: str) -> str:
        """Compare numeric features between original and synthetic datasets"""
        try:
            # Select up to 6 numeric features for comparison
            selected_features = numeric_cols[:6]
            
            n_cols = 3
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            
            fig.suptitle(f'{generator_name} - Numeric Features Comparison', fontsize=16, fontweight='bold')
            
            # Ensure axes is always a list
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            colors = ['#2E86AB', '#A23B72']  # Blue for original, Pink for synthetic
            
            for i, feature in enumerate(selected_features):
                ax = axes[i]
                
                # Get data for both datasets
                orig_data = orig_df[feature].dropna()
                synth_data = synth_df[feature].dropna()
                
                if len(orig_data) > 0 and len(synth_data) > 0:
                    # Calculate bins for consistent comparison
                    all_data = pd.concat([orig_data, synth_data])
                    bins = np.linspace(all_data.min(), all_data.max(), 30)
                    
                    # Plot overlapping histograms
                    ax.hist(orig_data.values, bins=bins, alpha=0.7, label='Original', 
                           color=colors[0], density=True, edgecolor='white', linewidth=0.5)
                    ax.hist(synth_data.values, bins=bins, alpha=0.7, label='Synthetic', 
                           color=colors[1], density=True, edgecolor='white', linewidth=0.5)
                    
                    # Add statistics
                    orig_mean, orig_std = np.mean(orig_data), np.std(orig_data)
                    synth_mean, synth_std = np.mean(synth_data), np.std(synth_data)
                    
                    stats_text = f"Original: μ={orig_mean:.2f}, σ={orig_std:.2f}\\nSynthetic: μ={synth_mean:.2f}, σ={synth_std:.2f}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
                    
                    ax.set_title(f'{feature}', fontweight='bold', pad=15)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
            # Hide empty subplots
            for j in range(len(selected_features), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            filename = f"{generator_name}_numeric_comparison.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Generated numeric comparison plot for {len(selected_features)} features")
            return filepath
            
        except Exception as e:
            print(f"Warning: Numeric comparison plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    @staticmethod
    def _plot_categorical_comparison(orig_df: pd.DataFrame, synth_df: pd.DataFrame, 
                                   categorical_cols: List[str], generator_name: str, output_dir: str) -> str:
        """Compare categorical features between original and synthetic datasets"""
        try:
            # Select up to 4 categorical features for comparison
            selected_features = categorical_cols[:4]
            
            n_cols = 2
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
            
            fig.suptitle(f'{generator_name} - Categorical Features Comparison', fontsize=16, fontweight='bold')
            
            # Ensure axes is always a list
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            colors = ['#2E86AB', '#A23B72']
            
            for i, feature in enumerate(selected_features):
                ax = axes[i]
                
                # Get value counts for both datasets
                orig_counts = orig_df[feature].value_counts().sort_index()
                synth_counts = synth_df[feature].value_counts().sort_index()
                
                # Get all unique values
                all_values = sorted(set(orig_counts.index) | set(synth_counts.index))
                
                # Ensure both series have the same index
                orig_counts = orig_counts.reindex(all_values, fill_value=0)
                synth_counts = synth_counts.reindex(all_values, fill_value=0)
                
                # Convert to percentages
                orig_pct = (orig_counts / orig_counts.sum()) * 100
                synth_pct = (synth_counts / synth_counts.sum()) * 100
                
                # Plot side by side bars
                x = np.arange(len(all_values))
                width = 0.35
                
                ax.bar(x - width/2, orig_pct.values, width, label='Original', 
                      color=colors[0], alpha=0.8, edgecolor='white', linewidth=1)
                ax.bar(x + width/2, synth_pct.values, width, label='Synthetic', 
                      color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
                
                ax.set_title(f'{feature}', fontweight='bold', pad=15)
                ax.set_xlabel('Category')
                ax.set_ylabel('Percentage (%)')
                ax.set_xticks(x)
                ax.set_xticklabels(all_values, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add percentage labels on bars
                for j, (orig_val, synth_val) in enumerate(zip(orig_pct.values, synth_pct.values)):
                    if orig_val > 0:
                        ax.text(j - width/2, orig_val + max(orig_pct.values) * 0.01, 
                               f'{orig_val:.1f}%', ha='center', va='bottom', fontsize=8)
                    if synth_val > 0:
                        ax.text(j + width/2, synth_val + max(synth_pct.values) * 0.01, 
                               f'{synth_val:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Hide empty subplots
            for j in range(len(selected_features), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            filename = f"{generator_name}_categorical_comparison.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Generated categorical comparison plot for {len(selected_features)} features")
            return filepath
            
        except Exception as e:
            print(f"Warning: Categorical comparison plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    @staticmethod
    def _plot_target_comparison(orig_df: pd.DataFrame, synth_df: pd.DataFrame, 
                              generator_name: str, output_dir: str) -> str:
        """Compare target variable distribution between original and synthetic datasets"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'{generator_name} - Target Distribution Comparison', fontsize=16, fontweight='bold')
            
            # Get target distributions
            orig_target = orig_df['target'].value_counts().sort_index()
            synth_target = synth_df['target'].value_counts().sort_index()
            
            # Ensure both have same classes
            all_classes = sorted(set(orig_target.index) | set(synth_target.index))
            orig_target = orig_target.reindex(all_classes, fill_value=0)
            synth_target = synth_target.reindex(all_classes, fill_value=0)
            
            colors = ['#2E86AB', '#A23B72']
            
            # 1. Absolute counts comparison
            ax1 = axes[0]
            x = np.arange(len(all_classes))
            width = 0.35
            
            ax1.bar(x - width/2, orig_target.values, width, label='Original', 
                   color=colors[0], alpha=0.8, edgecolor='white', linewidth=1)
            ax1.bar(x + width/2, synth_target.values, width, label='Synthetic', 
                   color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
            
            ax1.set_title('Absolute Counts', fontweight='bold', pad=15)
            ax1.set_xlabel('Target Class')
            ax1.set_ylabel('Count')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'Class {cls}' for cls in all_classes])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add count labels
            for i, (orig_val, synth_val) in enumerate(zip(orig_target.values, synth_target.values)):
                ax1.text(i - width/2, orig_val + max(orig_target.values) * 0.01, 
                        f'{orig_val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax1.text(i + width/2, synth_val + max(synth_target.values) * 0.01, 
                        f'{synth_val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 2. Percentage comparison
            ax2 = axes[1]
            orig_pct = (orig_target / orig_target.sum()) * 100
            synth_pct = (synth_target / synth_target.sum()) * 100
            
            ax2.bar(x - width/2, orig_pct.values, width, label='Original', 
                   color=colors[0], alpha=0.8, edgecolor='white', linewidth=1)
            ax2.bar(x + width/2, synth_pct.values, width, label='Synthetic', 
                   color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
            
            ax2.set_title('Percentage Distribution', fontweight='bold', pad=15)
            ax2.set_xlabel('Target Class')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'Class {cls}' for cls in all_classes])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, (orig_val, synth_val) in enumerate(zip(orig_pct.values, synth_pct.values)):
                ax2.text(i - width/2, orig_val + max(orig_pct.values) * 0.01, 
                        f'{orig_val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax2.text(i + width/2, synth_val + max(synth_pct.values) * 0.01, 
                        f'{synth_val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            filename = f"{generator_name}_target_comparison.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✓ Generated target comparison plot")
            return filepath
            
        except Exception as e:
            print(f"Warning: Target comparison plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None