import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class SyntheticReporter:
    def _report_dataset(self, df: pd.DataFrame, target_col: str, extra_info: dict = None):
        """Displays statistics of the generated dataset."""
        
        print("\n📊 === DATASET REPORT ===")
        print(f"Total instances: {len(df)}")
        features = [c for c in df.columns if c != target_col]
        print(f"Features: {len(features)} ({', '.join(features)})")
        
        # Class distribution
        if target_col in df.columns:
            print("\n⚖️ Class distribution:")
            dist = df[target_col].value_counts(normalize=True).to_dict()
            for cls, prop in dist.items():
                print(f"  - {cls}: {df[target_col].value_counts()[cls]} "
                      f"({prop:.2%})")
        
        # Basic feature statistics
        print("\n📈 Feature statistics:")
        try:
            print(df[features].describe().T)
        except Exception as e:
            print(f"Could not compute feature statistics: {e}")
        
        # Block information
        if "chunk" in df.columns:
            self._report_chunks(df)
        
        # Drift segment information
        if "drift_segment" in df.columns:
            self._report_drift(df)
        
        # Extra info provided manually
        if extra_info:
            self._report_extra_info(extra_info)
        
        # Automatic drift analysis if drift is detected
        self._analyze_drift_if_present(df, target_col, extra_info)
        
        print("=== END OF REPORT ===\n")
    
    def _report_chunks(self, df: pd.DataFrame):
        """Reports information about blocks (chunks) in the dataset."""
        print("\n🧩 Block information:")
        chunk_counts = df["chunk"].value_counts().sort_index().to_dict()
        for chunk, count in chunk_counts.items():
            print(f"  - Block {chunk}: {count} samples")
    
    def _report_drift(self, df: pd.DataFrame):
        """Reports drift information based on drift segments."""
        print("\n🔀 Drift information:")
        drift_counts = df["drift_segment"].value_counts().sort_index().to_dict()
        for seg, count in drift_counts.items():
            print(f"  - Segment {seg}: {count} samples")
    
    def _report_extra_info(self, extra_info: dict):
        """Reports additional information."""
        print("\n🔎 Extra information:")
        for key, val in extra_info.items():
            print(f"  - {key}: {val}")
    
    def _analyze_drift_if_present(self, df: pd.DataFrame, target_col: str, extra_info: dict = None):
        """Automatically analyzes if drift is present based on extra info."""
        if not extra_info:
            return
        
        drift_type = extra_info.get("Drift type")
        drift_position = extra_info.get("Drift position")
        
        # Only analyze if drift type and position are defined
        if drift_type and drift_type != "none" and drift_position:
            print("\n🔍 === DRIFT ANALYSIS ===")
            self._perform_drift_analysis(df, drift_position, target_col, drift_type)
    
    def _perform_drift_analysis(self, df: pd.DataFrame, drift_position: int, target_col: str, drift_type: str):
        """Performs a detailed drift analysis."""
        
        # Split the data before and after the drift
        before_drift = df.iloc[:drift_position]
        after_drift = df.iloc[drift_position:]
        
        print(f"Drift type: {drift_type}")
        print(f"Drift position: {drift_position}")
        print(f"Samples before drift: {len(before_drift)}")
        print(f"Samples after drift: {len(after_drift)}\n")
        
        # Class distribution analysis
        self._analyze_class_distribution(before_drift, after_drift, target_col)
        
        # Drift metrics
        self._calculate_drift_metrics(before_drift, after_drift, target_col)
        
        # Feature drift analysis (for concept drift)
        if drift_type in ["concept", "both"]:
            self._analyze_feature_drift(before_drift, after_drift, target_col)
        
        # Create visualizations
        self._create_drift_visualizations(df, drift_position, target_col, drift_type)
    
    def _analyze_class_distribution(self, before_drift, after_drift, target_col):
        """Analyzes the class distribution before and after drift."""
        print("📊 CLASS DISTRIBUTION ANALYSIS:")
        print("Before drift:")
        before_counts = before_drift[target_col].value_counts().sort_index()
        before_ratios = before_drift[target_col].value_counts(normalize=True).sort_index()
        for cls in before_counts.index:
            print(f"  Class {cls}: {before_counts[cls]} ({before_ratios[cls]:.3f})")
        
        print("\nAfter drift:")
        after_counts = after_drift[target_col].value_counts().sort_index()
        after_ratios = after_drift[target_col].value_counts(normalize=True).sort_index()
        for cls in after_counts.index:
            print(f"  Class {cls}: {after_counts[cls]} ({after_ratios[cls]:.3f})")
    
    def _calculate_drift_metrics(self, before_drift, after_drift, target_col):
        """Calculates drift metrics such as maximum ratio difference and KL Divergence."""
        print("\n📈 DRIFT METRICS:")
        
        # Maximum ratio difference
        max_diff = 0
        for cls in before_drift[target_col].value_counts(normalize=True).index:
            if cls in after_drift[target_col].value_counts(normalize=True).index:
                diff = abs(before_drift[target_col].value_counts(normalize=True)[cls] - 
                           after_drift[target_col].value_counts(normalize=True)[cls])
                max_diff = max(max_diff, diff)
        
        print(f"Maximum ratio difference: {max_diff:.4f}")
        
        # KL Divergence (approximate)
        try:
            kl_div = 0
            for cls in before_drift[target_col].value_counts(normalize=True).index:
                if cls in after_drift[target_col].value_counts(normalize=True).index and \
                   after_drift[target_col].value_counts(normalize=True)[cls] > 1e-10:
                    kl_div += before_drift[target_col].value_counts(normalize=True)[cls] * \
                               np.log(before_drift[target_col].value_counts(normalize=True)[cls] / 
                                      after_drift[target_col].value_counts(normalize=True)[cls])
            print(f"KL Divergence (approximate): {kl_div:.4f}")
        except:
            print("KL Divergence: Could not compute")
    
    def _analyze_feature_drift(self, before_drift, after_drift, target_col):
        """Analyzes feature drift for concept drift."""
        
        features = [c for c in before_drift.columns if c != target_col]
        numeric_features = before_drift[features].select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) == 0:
            return
        
        print("\n📊 FEATURE DRIFT ANALYSIS:")
        print("Feature mean changes:")
        
        for feature in numeric_features[:5]:  # Show only the first 5 features
            before_mean = before_drift[feature].mean()
            after_mean = after_drift[feature].mean()
            change = ((after_mean - before_mean) / before_mean * 100) if before_mean != 0 else 0
            print(f"  {feature}: {before_mean:.2f} → {after_mean:.2f} ({change:+.1f}%)")
    
    def _create_drift_visualizations(self, df, drift_position, target_col, drift_type):
        """Generates visualizations of drift."""
        
        try:
            # Set up the style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Drift Analysis - {drift_type.title()} Drift', fontsize=16)
            
            # Split the data before and after the drift
            before_drift = df.iloc[:drift_position]
            after_drift = df.iloc[drift_position:]
            
            # Chart 1: Distribution before the drift
            before_counts = before_drift[target_col].value_counts().sort_index()
            before_counts.plot(kind='bar', ax=axes[0,0], color='skyblue', alpha=0.8)
            axes[0,0].set_title('Distribution BEFORE drift')
            axes[0,0].set_ylabel('Number of samples')
            axes[0,0].tick_params(axis='x', rotation=0)
            axes[0,0].grid(True, alpha=0.3)
            
            # Chart 2: Distribution after the drift
            after_counts = after_drift[target_col].value_counts().sort_index()
            after_counts.plot(kind='bar', ax=axes[0,1], color='lightcoral', alpha=0.8)
            axes[0,1].set_title('Distribution AFTER drift')
            axes[0,1].set_ylabel('Number of samples')
            axes[0,1].tick_params(axis='x', rotation=0)
            axes[0,1].grid(True, alpha=0.3)
            
            # Chart 3: Temporal evolution (sliding windows)
            window_size = max(100, len(df) // 20)  # Adaptive window size
            positions = []
            class_ratios = defaultdict(list)
            
            # Get unique classes
            unique_classes = sorted(df[target_col].unique())
            
            for i in range(0, len(df) - window_size, window_size//2):
                window_data = df.iloc[i:i+window_size]
                window_ratios = window_data[target_col].value_counts(normalize=True).sort_index()
                positions.append(i + window_size//2)
                
                for cls in unique_classes:
                    class_ratios[cls].append(window_ratios.get(cls, 0))
            
            # Plot each class
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, cls in enumerate(unique_classes):
                color = colors[i % len(colors)]
                axes[1,0].plot(positions, class_ratios[cls], 
                              label=f'Class {cls}', marker='o', color=color, alpha=0.7)
            
            axes[1,0].axvline(x=drift_position, color='red', linestyle='--', 
                            linewidth=2, label='Drift point', alpha=0.8)
            axes[1,0].set_title('Temporal evolution of class ratios')
            axes[1,0].set_xlabel('Position in dataset')
            axes[1,0].set_ylabel('Class ratio')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Chart 4: Side-by-side comparison
            before_ratios = before_drift[target_col].value_counts(normalize=True).sort_index()
            after_ratios = after_drift[target_col].value_counts(normalize=True).sort_index()
            
            comparison_data = pd.DataFrame({
                'Before drift': before_ratios,
                'After drift': after_ratios
            }).fillna(0)
            
            comparison_data.plot(kind='bar', ax=axes[1,1], color=['skyblue', 'lightcoral'], alpha=0.8)
            axes[1,1].set_title('Ratio comparison')
            axes[1,1].set_ylabel('Class ratio')
            axes[1,1].tick_params(axis='x', rotation=0)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print("\n📈 Visualizations created successfully!")
            
        except Exception as e:
            print(f"\n⚠️  Could not create visualizations: {e}")
            print("This might be due to matplotlib not being available or display issues.")
    
    def analyze_data_drift_standalone(self, filepath: str, drift_position: int, target_col: str = 'target'):
        """
        Standalone method to analyze drift from a CSV file.
        Useful for post-analysis.
        """
        try:
            df = pd.read_csv(filepath)
            print(f"\n🔍 === STANDALONE DRIFT ANALYSIS ===")
            print(f"Analyzing file: {filepath}")
            
            extra_info = {
                "Drift type": "unknown",
                "Drift position": drift_position
            }
            
            self._perform_drift_analysis(df, drift_position, target_col, "data")
            
        except Exception as e:
            print(f"Error analyzing drift: {e}")
