import os
import pandas as pd
from collections import defaultdict
from .SyntheticReporter import SyntheticReporter
import numpy as np

class SyntheticGenerator(SyntheticReporter):
    def generate(self, 
                 generator_instance,
                 output_path: str,
                 filename: str,
                 n_samples: int,
                 generator_instance_drift=None,
                 position_of_drift: int = None,
                 ratio_before: dict = None,
                 ratio_after: dict = None,
                 target_col: str = "target",
                 balance: bool = False,
                 drift_type: str = "none",  # 'none', 'concept', 'data', 'both'
                 extra_info: dict = None):
        """
        Generates synthetic data with optional drift (concept, data, or both).
        """
        # Validate drift-related parameters
        self._validate_drift_params(drift_type, generator_instance_drift, ratio_before, ratio_after)
        
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)

        data = []

        # --- Generate data based on drift type ---
        if drift_type == "none":
            data = self._generate_balanced(generator_instance, n_samples) if balance else self._generate_data(generator_instance, n_samples)
        elif drift_type == "concept":
            data = self._generate_concept_drift(generator_instance, generator_instance_drift, n_samples, position_of_drift)
        elif drift_type == "data":
            data = self._generate_data_drift(generator_instance, n_samples, position_of_drift, ratio_before, ratio_after)
        elif drift_type == "both":
            data = self._generate_both_drift(generator_instance, generator_instance_drift, n_samples, position_of_drift, ratio_before, ratio_after)
        
        # Create DataFrame
        columns = list(generator_instance.take(1)[0].keys()) + [target_col]  # Get column names from the first sample
        df = pd.DataFrame(data, columns=columns)

        # Save CSV and report generation
        df.to_csv(full_path, index=False)
        print(f"Data generated and saved at: {full_path}")
        self._report_dataset(df, target_col, extra_info={"Drift type": drift_type, "Drift position": position_of_drift, **(extra_info or {})})

        return full_path

    def _generate_balanced(self, generator_instance, n_samples):
        """
        Generates a balanced dataset by sampling equally from each class.
        """
        class_samples = defaultdict(list)
        
        # Generate enough samples to balance
        for x, y in generator_instance.take(n_samples * 5):  # Generate more to have enough options
            row = list(x.values()) + [y]
            class_samples[y].append(row)
            
            # Stop when we have enough samples from all classes
            min_class_samples = min(len(samples) for samples in class_samples.values())
            if min_class_samples >= n_samples // len(class_samples):
                break
        
        # Balance the classes
        data = []
        n_classes = len(class_samples)
        per_class = n_samples // n_classes
        
        for cls, samples in class_samples.items():
            data.extend(samples[:per_class])
        
        return data

    def _generate_data(self, generator_instance, n_samples):
        """
        Generates a simple dataset without drift (normal generation).
        """
        data = []
        for x, y in generator_instance.take(n_samples):
            row = list(x.values()) + [y]
            data.append(row)
        
        return data

    def _generate_concept_drift(self, generator_instance, generator_instance_drift, n_samples, position_of_drift):
        """
        Generates data with concept drift at a specified position.
        """
        data = []
        
        # Before the drift
        for x, y in generator_instance.take(position_of_drift):
            row = list(x.values()) + [y]
            data.append(row)
        
        # After the drift
        for x, y in generator_instance_drift.take(n_samples - position_of_drift):
            row = list(x.values()) + [y]
            data.append(row)

        return data

    def _generate_data_drift(self, generator_instance, n_samples, position_of_drift, ratio_before, ratio_after):
        """
        Generates data with data drift using specified ratios before and after the drift.
        """
        data = []
        
        # Generate data before the drift with specific ratios
        part1 = self._generate_with_ratios(generator_instance, position_of_drift, ratio_before)
        data.extend(part1)

        # Generate data after the drift with specific ratios
        part2 = self._generate_with_ratios(generator_instance, n_samples - position_of_drift, ratio_after)
        data.extend(part2)

        return data

    def _generate_both_drift(self, generator_instance, generator_instance_drift, n_samples, position_of_drift, ratio_before, ratio_after):
        """
        Generates data with both concept and data drift.
        """
        data = []
        
        # Before the drift: use first generator with ratio_before
        part1 = self._generate_with_ratios(generator_instance, position_of_drift, ratio_before)
        data.extend(part1)

        # After the drift: use second generator with ratio_after
        part2 = self._generate_with_ratios(generator_instance_drift, n_samples - position_of_drift, ratio_after)
        data.extend(part2)

        return data

    def _generate_with_ratios(self, generator_instance, n_samples, target_ratios):
        """
        Generates samples while ensuring the class ratios are respected.
        
        Args:
            generator_instance: The data generator instance
            n_samples: Total number of samples to generate
            target_ratios: Dict with target class ratios {class: ratio}
        """
        # Calculate how many samples are needed for each class
        target_counts = {cls: int(n_samples * ratio) for cls, ratio in target_ratios.items()}

        # Adjust the class with the most samples if needed
        total_assigned = sum(target_counts.values())
        if total_assigned != n_samples:
            max_class = max(target_counts, key=lambda x: target_counts[x])
            target_counts[max_class] += n_samples - total_assigned
        
        # Generate samples until enough are collected
        class_samples = defaultdict(list)
        for x, y in generator_instance.take(n_samples * 10):  # Generate enough
            if y in target_counts and len(class_samples[y]) < target_counts[y]:
                row = list(x.values()) + [y]
                class_samples[y].append(row)
            
            # Stop when we have enough samples for all classes
            if all(len(class_samples[cls]) >= target_counts[cls] for cls in target_counts.keys()):
                break
        
        # Build the final dataset with the exact ratios
        final_data = []
        actual_counts = {cls: len(class_samples[cls]) for cls in target_counts}
        
        for cls, target_count in target_counts.items():
            final_data.extend(class_samples[cls][:target_count])
        
        return final_data

    def _validate_drift_params(self, drift_type, generator_instance_drift, ratio_before, ratio_after):
        """
        Validates the drift parameters for concept drift, data drift, or both.
        """
        if drift_type in ["concept", "both"] and generator_instance_drift is None:
            raise ValueError("For concept and both drift types, generator_instance_drift must be provided.")
        
        if drift_type == "data" and (ratio_before is None or ratio_after is None):
            raise ValueError("For data drift, both ratio_before and ratio_after must be provided.")

        if drift_type in ["data", "both"]:
            if abs(sum(ratio_before.values()) - 1.0) > 1e-6:
                raise ValueError(f"ratio_before must sum to 1.0, got {sum(ratio_before.values())}")
            if abs(sum(ratio_after.values()) - 1.0) > 1e-6:
                raise ValueError(f"ratio_after must sum to 1.0, got {sum(ratio_after.values())}")
