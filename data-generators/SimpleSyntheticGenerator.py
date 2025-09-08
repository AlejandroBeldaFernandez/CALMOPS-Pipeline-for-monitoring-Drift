#!/usr/bin/env python3
"""
Simplified Synthetic Data Generator that works correctly with River
"""

import os
import pandas as pd
from typing import Dict, Optional
from collections import defaultdict

class SimpleSyntheticGenerator:
    """Simple synthetic data generator using River generators"""
    
    def __init__(self):
        pass
    
    def generate_simple(self, 
                       generator_instance,
                       output_path: str,
                       filename: str,
                       n_samples: int,
                       target_col: str = "target"):
        """Generate simple dataset without drift"""
        
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        # Generate data
        data = []
        columns = None
        
        for i, (x, y) in enumerate(generator_instance.take(n_samples)):
            if columns is None:
                # Get column names from first sample
                if hasattr(x, 'keys'):
                    columns = list(x.keys()) + [target_col]
                else:
                    columns = [f"feature_{j}" for j in range(len(x))] + [target_col]
            
            # Convert sample to row
            if hasattr(x, 'values'):
                row = list(x.values()) + [y]
            else:
                row = list(x) + [y]
            
            data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(full_path, index=False)
        
        # Print report
        self._print_simple_report(df, target_col, filename)
        
        return full_path
    
    def generate_with_concept_drift(self,
                                  generator_base,
                                  generator_drift,
                                  output_path: str,
                                  filename: str,
                                  n_samples: int,
                                  drift_position: int,
                                  target_col: str = "target"):
        """Generate dataset with concept drift"""
        
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        data = []
        columns = None
        
        # Generate first part (before drift)
        for i, (x, y) in enumerate(generator_base.take(drift_position)):
            if columns is None:
                if hasattr(x, 'keys'):
                    columns = list(x.keys()) + [target_col]
                else:
                    columns = [f"feature_{j}" for j in range(len(x))] + [target_col]
            
            if hasattr(x, 'values'):
                row = list(x.values()) + [y]
            else:
                row = list(x) + [y]
            
            data.append(row)
        
        # Generate second part (after drift)
        remaining_samples = n_samples - drift_position
        for i, (x, y) in enumerate(generator_drift.take(remaining_samples)):
            if hasattr(x, 'values'):
                row = list(x.values()) + [y]
            else:
                row = list(x) + [y]
            
            data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(full_path, index=False)
        
        # Print report
        self._print_drift_report(df, target_col, filename, drift_position)
        
        return full_path
    
    def generate_with_data_drift(self,
                                generator_instance,
                                output_path: str,
                                filename: str,
                                n_samples: int,
                                drift_position: int,
                                ratio_before: Dict[int, float],
                                ratio_after: Dict[int, float],
                                target_col: str = "target"):
        """Generate dataset with data drift (class distribution change)"""
        
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        
        # Generate enough samples to select from
        all_samples = list(generator_instance.take(n_samples * 3))
        
        # Separate by class
        class_samples = defaultdict(list)
        columns = None
        
        for x, y in all_samples:
            if columns is None:
                if hasattr(x, 'keys'):
                    columns = list(x.keys()) + [target_col]
                else:
                    columns = [f"feature_{j}" for j in range(len(x))] + [target_col]
            
            if hasattr(x, 'values'):
                row = list(x.values()) + [y]
            else:
                row = list(x) + [y]
            
            class_samples[y].append(row)
        
        # Select samples according to distributions
        data = []
        
        # Before drift
        for class_label, ratio in ratio_before.items():
            n_class_samples = int(drift_position * ratio)
            if len(class_samples[class_label]) >= n_class_samples:
                data.extend(class_samples[class_label][:n_class_samples])
        
        # After drift
        remaining_samples = n_samples - drift_position
        for class_label, ratio in ratio_after.items():
            n_class_samples = int(remaining_samples * ratio)
            if len(class_samples[class_label]) >= n_class_samples:
                data.extend(class_samples[class_label][:n_class_samples])
        
        # Shuffle and create DataFrame
        import random
        random.shuffle(data)
        df = pd.DataFrame(data[:n_samples], columns=columns)
        df.to_csv(full_path, index=False)
        
        # Print report
        self._print_drift_report(df, target_col, filename, drift_position)
        
        return full_path
    
    def _print_simple_report(self, df, target_col, filename):
        """Print simple dataset report"""
        print(f"Generated: {filename}")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(df.columns) - 1}")
        
        if target_col in df.columns:
            class_dist = df[target_col].value_counts().to_dict()
            print(f"  Classes: {class_dist}")
    
    def _print_drift_report(self, df, target_col, filename, drift_position):
        """Print drift dataset report"""
        print(f"Generated: {filename}")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(df.columns) - 1}")
        print(f"  Drift position: {drift_position}")
        
        if target_col in df.columns:
            # Before drift
            before_df = df.iloc[:drift_position]
            before_dist = before_df[target_col].value_counts().to_dict()
            print(f"  Classes before drift: {before_dist}")
            
            # After drift
            after_df = df.iloc[drift_position:]
            after_dist = after_df[target_col].value_counts().to_dict()
            print(f"  Classes after drift: {after_dist}")