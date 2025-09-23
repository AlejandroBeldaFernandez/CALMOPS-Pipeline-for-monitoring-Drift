import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Check for visualization availability
VISUALIZATION_AVAILABLE = True
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Visualization.AutoVisualizer import AutoVisualizer
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: AutoVisualizer not available - visualizations will be skipped")

class SyntheticReporter:
    """
    Simplified Synthetic Data Reporter - generates basic reports with essential metrics
    """
    
    def generate_report(self, df: pd.DataFrame, target_col: str = "target", 
                       drift_type: str = "none", position_of_drift: int = None,
                       is_block_dataset: bool = False,
                       output_path: str = None):
        """
        Generate simplified console report
        
        Args:
            df: DataFrame with generated data
            target_col: Name of target column
            drift_type: Type of drift ('none', 'concept', 'data', 'both')
            position_of_drift: Position where drift occurs (if applicable)
            is_block_dataset: Whether this is a block-structured dataset
            extra_info: Additional information to include
            output_path: Path for saving visualizations
        """
        print("\n" + "=" * 80)
        print("CALMOPS SYNTHETIC DATA REPORT")
        print("=" * 80)
        
        if is_block_dataset:
            # Block-based analysis
            self._report_block_dataset(df, target_col, drift_type, output_path)
        else:
            # Standard dataset analysis
            self._report_standard_dataset(df, target_col, drift_type, position_of_drift, output_path)
        
        # Generate visualizations at the end
        if output_path and VISUALIZATION_AVAILABLE:
            print(f"\nGENERATING VISUALIZATIONS:")
            print("=" * 80)
            self._generate_visualizations(df, output_path)
        
        print("=" * 80)
        print("REPORT COMPLETE")
        print("=" * 80)
    
    def _report_standard_dataset(self, df: pd.DataFrame, target_col: str, drift_type: str, 
                                position_of_drift: int, output_path: str):
        """Report for standard (non-block) datasets"""
        
        # Basic dataset information
        print(f"\nDATASET INFORMATION:")
        print(f"Shape: {df.shape}")
        print(f"Size: {df.size:,} total elements")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Dataset description
        print(f"\nDATASET DESCRIPTION:")
        print(f"Column names: {list(df.columns)}")
        print(f"Target column: {target_col}")
        print(f"Number of features: {len(df.columns) - 1}")  # Exclude target
        print(f"Number of instances: {len(df):,}")
        
        # Data types information
        print(f"\nDATA TYPES:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
        print(f"Numeric features: {len(numeric_cols)} ({numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''})")
        print(f"Categorical features: {len(categorical_cols)} ({categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''})")
        print(f"Target type: {df[target_col].dtype}")
        
        # Statistical summary for numeric columns
        if len(numeric_cols) > 0:
            print(f"\nNUMERIC FEATURES STATISTICS:")
            describe_stats = df[numeric_cols].describe()
            print(describe_stats.round(3))
        
        # Target distribution
        if target_col in df.columns:
            print(f"\nTARGET DISTRIBUTION:")
            target_counts = df[target_col].value_counts().sort_index()
            print(f"Unique classes: {len(target_counts)}")
            for class_val, count in target_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  Class {class_val}: {count:,} instances ({percentage:.1f}%)")
        
        # Data quality assessment
        print(f"\nDATA QUALITY:")
        missing_count = df.isnull().sum().sum()
        print(f"Missing values: {missing_count} ({'None' if missing_count == 0 else str(missing_count)})")
        
        duplicate_count = df.duplicated().sum()
        print(f"Duplicate rows: {duplicate_count} ({'None' if duplicate_count == 0 else str(duplicate_count)})")
        
        # Missing values per column if any
        missing_per_col = df.isnull().sum()
        missing_cols = missing_per_col[missing_per_col > 0]
        if len(missing_cols) > 0:
            print(f"Missing values by column:")
            for col, missing in missing_cols.items():
                print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        # Categorical variables statistics
        if len(categorical_cols) > 0:
            print(f"\nCATEGORICAL VARIABLES STATISTICS:")
            for col in categorical_cols:
                print(f"\n  {col}:")
                unique_count = df[col].nunique()
                total_count = len(df[col])
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                mode_count = (df[col] == mode_val).sum() if mode_val != 'N/A' else 0
                mode_pct = (mode_count / total_count) * 100 if total_count > 0 else 0
                
                print(f"    Count: {total_count}")
                print(f"    Unique values: {unique_count}")
                print(f"    Mode: '{mode_val}' ({mode_count} occurrences, {mode_pct:.1f}%)")
                
                # Show top 5 most frequent values
                value_counts = df[col].value_counts().head(5)
                print(f"    Top values:")
                for value, count in value_counts.items():
                    pct = (count / total_count) * 100
                    print(f"      '{value}': {count} ({pct:.1f}%)")
                
                # Additional statistics
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / total_count) * 100
                    print(f"    Missing: {missing_count} ({missing_pct:.1f}%)")
        
        # Basic correlation info for numeric features
        if len(numeric_cols) > 1:
            print(f"\nCORRELATION SUMMARY:")
            corr_matrix = df[numeric_cols].corr()
            # Find highest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        abs(corr_matrix.iloc[i, j])
                    ))
            if corr_pairs:
                top_corr = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:3]
                print(f"Top correlations:")
                for col1, col2, corr_val in top_corr:
                    print(f"  {col1} - {col2}: {corr_val:.3f}")
        
        # Enhanced Drift information
        print(f"\nDRIFT INFORMATION:")
        if drift_type == "none":
            print("No drift introduced")
            print("Dataset maintains consistent patterns throughout")
        else:
            print(f"Drift type: {drift_type.upper()}")
            if position_of_drift:
                print(f"Drift introduced at position: {position_of_drift} ({position_of_drift/len(df)*100:.1f}% through dataset)")
            
            # Detailed drift type explanations
            if drift_type == "concept":
                print("\nCONCEPT DRIFT DETAILS:")
                print("- Type: Changes in the relationship between features and target")
                print("- Effect: Same input features may lead to different target classes")
                print("- Detection: Look for target distribution changes while feature distributions remain stable")
                print("- Impact: Model predictions become less accurate over time")
                
            elif drift_type == "data":
                print("\nDATA DRIFT DETAILS:")
                print("- Type: Changes in the input feature distributions")
                print("- Effect: Feature statistics (mean, variance, correlations) change over time")
                print("- Detection: Monitor feature statistics and distributions")
                print("- Impact: Model receives inputs outside training distribution")
                
            elif drift_type == "both":
                print("\nCOMBINED DRIFT DETAILS:")
                print("- Type: Both data drift and concept drift present")
                print("- Data Drift: Input feature distributions change")
                print("- Concept Drift: Feature-target relationships change")
                print("- Detection: Monitor both feature statistics and prediction accuracy")
                print("- Impact: Compound effect - both input and output relationships deteriorate")
            
            # Drift implications and recommendations
            print(f"\nDRIFT IMPLICATIONS:")
            print("- Model Monitoring: Implement continuous drift detection")
            print("- Retraining Strategy: Consider adaptive learning or periodic retraining")
            print("- Data Quality: Validate new data against reference distributions")
            print("- Performance: Expect degraded model performance after drift point")
    
    def _report_block_dataset(self, df: pd.DataFrame, target_col: str, drift_type: str, output_path: str):
        """Report for block-based datasets"""
        
        # Check for block column
        block_column = None
        for col_name in ['block', 'chunk', 'Block', 'Chunk']:
            if col_name in df.columns:
                block_column = col_name
                break
        
        if block_column is None:
            print("No block column found - treating as standard dataset")
            self._report_standard_dataset(df, target_col, drift_type, None, output_path)
            return
        
        unique_blocks = sorted(df[block_column].unique())
        
        # General dataset summary with detailed statistics
        print(f"\nGENERAL DATASET SUMMARY:")
        print(f"Shape: {df.shape}")
        print(f"Size: {df.size:,} total elements")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"Number of blocks: {len(unique_blocks)}")
        print(f"Total instances: {len(df):,}")
        print(f"Average instances per block: {len(df) / len(unique_blocks):.1f}")
        print(f"Total features: {len(df.columns) - 2}")  # Exclude target and block columns
        
        # Block size distribution
        block_sizes = df[block_column].value_counts().sort_index()
        print(f"\nBLOCK SIZE DISTRIBUTION:")
        for block_id, size in block_sizes.items():
            percentage = (size / len(df)) * 100
            print(f"  Block {block_id}: {size:,} instances ({percentage:.1f}%)")
        
        # Global data types information
        print(f"\nGLOBAL DATA TYPES:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Remove target and block columns from feature lists
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        if block_column in numeric_cols:
            numeric_cols.remove(block_column)
        if block_column in categorical_cols:
            categorical_cols.remove(block_column)
            
        print(f"Numeric features: {len(numeric_cols)} ({numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''})")
        print(f"Categorical features: {len(categorical_cols)} ({categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''})")
        print(f"Target type: {df[target_col].dtype}")
        
        # Analysis per block with detailed statistics
        print(f"\nDETAILED ANALYSIS PER BLOCK:")
        print("=" * 50)
        
        for block_id in unique_blocks:
            block_data = df[df[block_column] == block_id]
            
            print(f"\nBLOCK {block_id} DETAILED STATISTICS:")
            print(f"Shape: {block_data.shape}")
            print(f"Size: {len(block_data):,} instances")
            print(f"Percentage of total: {len(block_data)/len(df)*100:.1f}%")
            
            # Statistical summary for numeric columns in this block
            block_numeric_cols = [col for col in numeric_cols if col in block_data.columns]
            if len(block_numeric_cols) > 0:
                print(f"Numeric features statistics:")
                describe_stats = block_data[block_numeric_cols].describe()
                print(describe_stats.round(3))
            
            # Target distribution for this block
            if target_col in block_data.columns:
                print(f"Target distribution:")
                target_counts = block_data[target_col].value_counts().sort_index()
                print(f"  Unique classes: {len(target_counts)}")
                for class_val, count in target_counts.items():
                    percentage = (count / len(block_data)) * 100
                    print(f"  Class {class_val}: {count:,} instances ({percentage:.1f}%)")
            
            # Data quality for this block
            print(f"Data quality:")
            missing_count = block_data.isnull().sum().sum()
            print(f"  Missing values: {missing_count} ({'None' if missing_count == 0 else str(missing_count)})")
            
            duplicate_count = block_data.duplicated().sum()
            print(f"  Duplicate rows: {duplicate_count} ({'None' if duplicate_count == 0 else str(duplicate_count)})")
            
            # Missing values per column in this block if any
            missing_per_col = block_data.isnull().sum()
            missing_cols = missing_per_col[missing_per_col > 0]
            if len(missing_cols) > 0:
                print(f"  Missing values by column:")
                for col, missing in missing_cols.items():
                    print(f"    {col}: {missing} ({missing/len(block_data)*100:.1f}%)")
            
            # Categorical variables for this block
            block_categorical_cols = [col for col in categorical_cols if col in block_data.columns]
            if len(block_categorical_cols) > 0:
                print(f"Categorical variables statistics:")
                for col in block_categorical_cols[:3]:  # Show first 3
                    unique_count = block_data[col].nunique()
                    total_count = len(block_data[col])
                    mode_val = block_data[col].mode().iloc[0] if len(block_data[col].mode()) > 0 else 'N/A'
                    mode_count = (block_data[col] == mode_val).sum() if mode_val != 'N/A' else 0
                    mode_pct = (mode_count / total_count) * 100 if total_count > 0 else 0
                    
                    print(f"    {col}:")
                    print(f"      Count: {total_count}, Unique: {unique_count}")
                    print(f"      Mode: '{mode_val}' ({mode_count} occurrences, {mode_pct:.1f}%)")
                    
                    # Show top 3 values for this block
                    value_counts = block_data[col].value_counts().head(3)
                    print(f"      Top values: ", end="")
                    top_values = []
                    for value, count in value_counts.items():
                        pct = (count / total_count) * 100
                        top_values.append(f"'{value}': {count} ({pct:.1f}%)")
                    print(", ".join(top_values))
            
            # Block-specific correlation summary for numeric features
            if len(block_numeric_cols) > 1:
                print(f"Correlation summary:")
                corr_matrix = block_data[block_numeric_cols].corr()
                # Find highest correlations in this block
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not np.isnan(corr_val):
                            corr_pairs.append((
                                corr_matrix.columns[i], 
                                corr_matrix.columns[j], 
                                abs(corr_val)
                            ))
                if corr_pairs:
                    top_corr = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:2]
                    print(f"  Top correlations:")
                    for col1, col2, corr_val in top_corr:
                        print(f"    {col1} - {col2}: {corr_val:.3f}")
        
        # Enhanced Drift detection between blocks
        print(f"\nDRIFT ANALYSIS BETWEEN BLOCKS:")
        print("=" * 50)
        
        drift_detected = False
        concept_drift_blocks = []
        data_drift_blocks = []
        
        # Get numeric columns for data drift analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if block_column in numeric_cols:
            numeric_cols.remove(block_column)
        
        for i in range(len(unique_blocks) - 1):
            current_block = unique_blocks[i]
            next_block = unique_blocks[i + 1]
            
            current_data = df[df[block_column] == current_block]
            next_data = df[df[block_column] == next_block]
            
            # Concept drift detection (target distribution changes)
            concept_drift_detected = False
            if target_col in df.columns:
                current_dist = current_data[target_col].value_counts(normalize=True).sort_index()
                next_dist = next_data[target_col].value_counts(normalize=True).sort_index()
                
                # Check for significant changes in target distribution
                for class_val in set(current_dist.index) | set(next_dist.index):
                    curr_pct = current_dist.get(class_val, 0) * 100
                    next_pct = next_dist.get(class_val, 0) * 100
                    diff = abs(next_pct - curr_pct)
                    if diff > 10:  # Significant change
                        concept_drift_detected = True
                        break
            
            # Data drift detection (feature distribution changes)
            data_drift_detected = False
            if numeric_cols:
                for col in numeric_cols[:3]:  # Check first 3 numeric columns
                    if col in current_data.columns and col in next_data.columns:
                        curr_mean = current_data[col].mean()
                        next_mean = next_data[col].mean()
                        curr_std = current_data[col].std()
                        next_std = next_data[col].std()
                        
                        # Check for significant changes in mean or std
                        mean_change = abs(next_mean - curr_mean) / (abs(curr_mean) + 1e-8)
                        std_change = abs(next_std - curr_std) / (abs(curr_std) + 1e-8)
                        
                        if mean_change > 0.2 or std_change > 0.3:  # 20% mean change or 30% std change
                            data_drift_detected = True
                            break
            
            # Report findings
            if concept_drift_detected and data_drift_detected:
                print(f"Block {current_block} -> Block {next_block}: COMBINED DRIFT detected (Concept + Data)")
                concept_drift_blocks.append((current_block, next_block))
                data_drift_blocks.append((current_block, next_block))
                drift_detected = True
            elif concept_drift_detected:
                print(f"Block {current_block} -> Block {next_block}: CONCEPT DRIFT detected")
                concept_drift_blocks.append((current_block, next_block))
                drift_detected = True
            elif data_drift_detected:
                print(f"Block {current_block} -> Block {next_block}: DATA DRIFT detected")
                data_drift_blocks.append((current_block, next_block))
                drift_detected = True
            else:
                print(f"Block {current_block} -> Block {next_block}: No significant drift detected")
        
        # Summary of drift findings
        print(f"\nDRIFT SUMMARY:")
        print(f"- Total block transitions analyzed: {len(unique_blocks) - 1}")
        print(f"- Concept drift transitions: {len(concept_drift_blocks)}")
        print(f"- Data drift transitions: {len(data_drift_blocks)}")
        
        if not drift_detected:
            print("- Overall assessment: No significant drift detected between blocks")
        else:
            print("- Overall assessment: Drift patterns detected - monitor model performance")
            
            if concept_drift_blocks:
                print(f"\nCONCEPT DRIFT TRANSITIONS:")
                for curr, next_b in concept_drift_blocks:
                    print(f"  Block {curr} → Block {next_b}: Target distribution changes")
                    
            if data_drift_blocks:
                print(f"\nDATA DRIFT TRANSITIONS:")
                for curr, next_b in data_drift_blocks:
                    print(f"  Block {curr} → Block {next_b}: Feature distribution changes")
        
        # Recommendations based on findings
        if drift_detected:
            print(f"\nRECOMMENDATIONS:")
            print("- Implement drift detection algorithms (e.g., using Frouros library)")
            print("- Set up model performance monitoring")
            print("- Consider incremental learning or periodic retraining")
            print("- Validate data quality at block boundaries")
    
    def _generate_visualizations(self, df: pd.DataFrame, output_path: str):
        """Generate visualizations at the end of the report"""
        try:
            # Convert data to tuples format
            feature_cols = [col for col in df.columns if col not in ['target', 'block', 'chunk']]
            data_tuples = []
            
            for _, row in df.iterrows():
                features_dict = {col: row[col] for col in feature_cols}
                target = row.get('target', None)
                data_tuples.append((features_dict, target))
            
            # Generate visualizations
            print("Generating synthetic data visualizations...")
            AutoVisualizer.auto_analyze_and_visualize(
                data_tuples, 
                "Synthetic_Dataset", 
                output_path
            )
            print("Visualizations generated successfully")
        
        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")
    
    # Legacy method for compatibility
    def _report_dataset(self, df: pd.DataFrame, target_col: str, extra_info: dict = None):
        """Legacy method - redirects to generate_report"""
        return self.generate_report(df, target_col, extra_info=extra_info)