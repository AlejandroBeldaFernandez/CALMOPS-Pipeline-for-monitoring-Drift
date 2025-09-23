"""
Synthetic Data Generation Module for CalmOps
============================================

This module provides comprehensive synthetic data generation capabilities including:
- SyntheticGenerator: Core synthetic data generation with drift support
- SyntheticBlockGenerator: Block-based data generation for complex scenarios  
- GeneratorFactory: Factory pattern for creating River synthetic generators
- DriftDetector: Advanced drift detection algorithms (using Frouros)
- DriftInjector: Drift injection capabilities (shared with Real module)
- SyntheticReporter: Reporting and analysis tools

Example usage:
    from Synthetic.GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
    from Synthetic.SyntheticGenerator import SyntheticGenerator
    from DriftInjection import DriftInjector  # Import drift functionality
    
    # Generate synthetic data
    config = GeneratorConfig(random_state=42)
    generator = SyntheticGenerator()
    synthetic_path = generator.generate(
        output_path="output/", filename="data.csv", n_samples=1000, 
        method="agrawal", random_state=42
    )
    df = pd.read_csv(synthetic_path)
    
    # Inject drift using dedicated DriftInjection module
    injector = DriftInjector(random_state=42)
    drifted_df = injector.inject_feature_drift(
        df=df,
        feature_cols=['salary', 'commission'],
        drift_type='add_value',
        drift_values={'salary': 5000, 'commission': -1000},
        start_index=500
    )
"""

from .SyntheticGenerator import SyntheticGenerator
from .SyntheticBlockGenerator import SyntheticBlockGenerator
from .GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
from .DriftDetector import DriftDetector, DetectorConfig, compare_detectors
from .SyntheticReporter import SyntheticReporter

__all__ = [
    'SyntheticGenerator',
    'SyntheticBlockGenerator', 
    'GeneratorFactory',
    'GeneratorType',
    'GeneratorConfig',
    'DriftDetector',
    'DetectorConfig',
    'compare_detectors',
    'SyntheticReporter'
]

__version__ = "1.0.0"
__author__ = "CalmOps Team"
__description__ = "Synthetic data generation and drift detection framework"