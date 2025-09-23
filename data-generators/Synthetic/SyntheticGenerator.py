import os
import pandas as pd
from collections import defaultdict
from .SyntheticReporter import SyntheticReporter
import numpy as np

# Import auto-visualization system
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Visualization.AutoVisualizer import AutoVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class SyntheticGenerator:
    def __init__(self, enable_auto_visualization: bool = True):
        self.enable_auto_visualization = enable_auto_visualization and VISUALIZATION_AVAILABLE
        if self.enable_auto_visualization:
            print("Auto-visualization enabled for SyntheticGenerator")
    
    def generate(self,
                 output_path: str,
                 filename: str,
                 n_samples: int,
                 method: str = "sea",
                 method_params: dict = None,
                 drift_type: str = "none",
                 position_of_drift: int = None,
                 target_col: str = "target",
                 balance: bool = False,
                 random_state: int = None):
        """
        Generate synthetic data (simplified interface similar to RealGenerator)
        
        Args:
            output_path: Directory to save the generated data
            filename: Name of the output file
            n_samples: Number of samples to generate
            method: Generator type ('sea', 'agrawal', 'hyperplane', 'sine', 'stagger')
            method_params: Dictionary with method-specific parameters
            drift_type: Type of drift ('none', 'concept', 'data', 'both')
            position_of_drift: Position where drift occurs (if applicable)
            target_col: Name of target column
            balance: Whether to balance classes
            random_state: Random seed for reproducibility
        """
        from .GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
        
        # Method string to enum mapping
        method_mapping = {
            "sea": GeneratorType.SEA,
            "agrawal": GeneratorType.AGRAWAL,
            "hyperplane": GeneratorType.HYPERPLANE,
            "sine": GeneratorType.SINE,
            "stagger": GeneratorType.STAGGER,
            "random_tree": GeneratorType.RANDOM_TREE,
            "mixed": GeneratorType.MIXED,
            "friedman": GeneratorType.FRIEDMAN,
            "random_rbf": GeneratorType.RANDOM_RBF
        }
        
        if method not in method_mapping:
            available_methods = list(method_mapping.keys())
            raise ValueError(f"Invalid method '{method}'. Choose one of {available_methods}")
        
        # Create generator config
        method_params = method_params or {}
        if random_state is not None:
            method_params['random_state'] = random_state
        
        config = GeneratorConfig(**method_params)
        
        # Create generator instances
        factory = GeneratorFactory()
        generator_instance = factory.create_generator(method_mapping[method], config)
        
        # Create drift generator if needed
        generator_instance_drift = None
        if drift_type != "none" and position_of_drift is not None:
            # Create slightly different config for drift
            drift_params = method_params.copy()
            if method == "sea":
                # Change SEA function for concept drift
                drift_params['function'] = drift_params.get('function', 0) + 1
            elif method == "agrawal":
                # Change classification function for concept drift
                drift_params['classification_function'] = drift_params.get('classification_function', 0) + 1
            
            if random_state is not None:
                drift_params['random_state'] = random_state + 1
                
            drift_config = GeneratorConfig(**drift_params)
            generator_instance_drift = factory.create_generator(method_mapping[method], drift_config)
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Call internal generate method
        return self._generate_internal(
            generator_instance=generator_instance,
            generator_instance_drift=generator_instance_drift,
            output_path=output_path,
            filename=filename,
            n_samples=n_samples,
            position_of_drift=position_of_drift,
            target_col=target_col,
            balance=balance,
            drift_type=drift_type
        )
    
    def _generate_internal(self, 
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
        # Validate parameters
        self.validate_params(generator_instance,
                     output_path,
                     filename,
                     n_samples,
                     generator_instance_drift,
                     position_of_drift,
                     ratio_before,
                     ratio_after,
                     target_col,
                     balance,
                     drift_type)

        # Validate drift-related parameters
        self._validate_drift_params(drift_type, generator_instance_drift, ratio_before, ratio_after)
        
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
        # Get column names from the first sample
        first_sample = next(iter(generator_instance.take(1)))
        columns = list(first_sample[0].keys()) + [target_col]
        df = pd.DataFrame(data, columns=columns)

        # Save CSV
        df.to_csv(full_path, index=False)
        print(f"Data generated and saved at: {full_path}")
        
        # Generate comprehensive report using SyntheticReporter
        # Check for different possible block column names
        is_block_dataset = any(col in df.columns for col in ['block', 'chunk', 'Block', 'Chunk'])
        reporter = SyntheticReporter()
        reporter.generate_report(df, target_col, drift_type, position_of_drift, is_block_dataset, extra_info or {})
        
        # Generate automatic visualization if enabled
        if self.enable_auto_visualization:
            try:
                base_filename = os.path.splitext(filename)[0]
                self._generate_automatic_visualization(df, generator_instance, output_path, base_filename)
            except Exception as e:
                print(f"Warning: Automatic visualization failed: {e}")

        return full_path

    def _generate_balanced(self, generator_instance, n_samples):
        """
        Generates a balanced dataset by sampling equally from each class.
        """
        class_samples = defaultdict(list)
        
        # Generate enough samples to balance
        for x, y in generator_instance.take(n_samples * 10):  # Generate more to have enough options
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
    
    def validate_params(self,
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
                        drift_type: str = "none"):
        """
        General validation for parameters passed to the generator.
        """
        # Validate n_samples
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

        # Validate output_path
               
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError("output_path must be a non-empty string")

        if not os.path.isdir(output_path):
            raise ValueError(f"output_path does not exist or is not a directory: {output_path}")


        # Validate filename
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")

        # Validate drift_type
        valid_drift_types = ["none", "concept", "data", "both"]
        if drift_type not in valid_drift_types:
            raise ValueError(f"Invalid drift_type '{drift_type}'. Must be one of {valid_drift_types}")

        # Validate position_of_drift
        if drift_type in ["concept", "data", "both"]:
            if position_of_drift is None or not (0 < position_of_drift < n_samples):
                raise ValueError(f"position_of_drift must be between 0 and n_samples ({n_samples}) when drift is applied")

        # Validate balance
        if not isinstance(balance, bool):
            raise ValueError(f"balance must be a boolean, got {type(balance)}")

        # Delegate to drift-specific validator
        self._validate_drift_params(drift_type, generator_instance_drift, ratio_before, ratio_after)

        # Validate generator_instance
        if generator_instance is None:
            raise ValueError("generator_instance must be provided")

        return True

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
    def _generate_automatic_visualization(self, data_input, generator_instance, output_path, base_filename):
        """Generate automatic visualization using AutoVisualizer"""
        try:
            # Check if input is DataFrame or raw data
            if hasattr(data_input, 'columns'):  # DataFrame
                df = data_input
                # Convert DataFrame to tuples format for AutoVisualizer
                tuples_data = []
                feature_cols = [col for col in df.columns if col != 'target']
                for _, row in df.iterrows():
                    features_dict = {col: row[col] for col in feature_cols}
                    target = row['target']
                    tuples_data.append((features_dict, target))
            else:  # Raw data
                # Convert raw data to proper format for AutoVisualizer
                tuples_data = []
                for row in data_input:
                    features_dict = {}
                    # Get feature names from generator
                    first_sample = next(iter(generator_instance.take(1)))
                    feature_names = list(first_sample[0].keys())
                    
                    # Create features dict from row data
                    for i, feature_name in enumerate(feature_names):
                        features_dict[feature_name] = row[i]
                    
                    # Target is the last element
                    target = row[-1]
                    tuples_data.append((features_dict, target))
            
            print(f"\nGENERATING AUTOMATIC VISUALIZATIONS...")
            
            # Use AutoVisualizer to generate comprehensive plots
            viz_results = AutoVisualizer.auto_analyze_and_visualize(
                tuples_data, 
                "Synthetic_Dataset", 
                output_path
            )
            
            if viz_results and 'visualization_files' in viz_results:
                plot_count = len(viz_results['visualization_files'])
                print(f"Generated {plot_count} visualization plots:")
                for plot_name in viz_results['visualization_files'].keys():
                    print(f"  - {plot_name}")
                
                # Report quality and drift from visualization
                if 'quality_score' in viz_results:
                    quality = viz_results['quality_score']
                    print(f"\nDATA QUALITY ASSESSMENT:")
                    print(f"  Quality metrics calculated and stored")
                
                if 'drift_analysis' in viz_results:
                    drift = viz_results['drift_analysis']
                    if drift['has_drift']:
                        print(f"\nAUTOMATIC DRIFT DETECTION:")
                        print(f"  Status: DRIFT DETECTED")
                        print(f"  Drift Points: {len(drift['drift_points'])} detected")
                    else:
                        print(f"\nAUTOMATIC DRIFT DETECTION:")
                        print(f"  Status: NO DRIFT DETECTED")
            
        except Exception as e:
            print(f"Warning: Automatic visualization failed: {e}")
            print("Data generation completed successfully, but plots were not created.")
    
   