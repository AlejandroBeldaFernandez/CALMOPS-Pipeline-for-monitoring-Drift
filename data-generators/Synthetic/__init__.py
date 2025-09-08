"""
Synthetic Data Generation Module for CalmOps
============================================

This module provides comprehensive synthetic data generation capabilities including:
- SyntheticGenerator: Core synthetic data generation with drift support
- BlockDriftGenerator: Block-based data generation for complex drift scenarios  
- GeneratorFactory: Factory pattern for creating River synthetic generators
- DriftDetector: Advanced drift detection algorithms
- SyntheticReporter: Reporting and analysis tools

Example usage:
    from Synthetic.GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
    from Synthetic.SyntheticGenerator import SyntheticGenerator
    
    # Create generator
    config = GeneratorConfig(random_state=42)
    gen = GeneratorFactory.create_generator(GeneratorType.AGRAWAL, config)
    
    # Generate synthetic data
    generator = SyntheticGenerator()
    generator.generate(gen, "output/", "test.csv", 1000)
"""

from .SyntheticGenerator import SyntheticGenerator
from .BlockDriftGenerator import BlockDriftGenerator
from .GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
from .DriftDetector import DriftDetector, DetectorConfig, compare_detectors
from .SyntheticReporter import SyntheticReporter

__all__ = [
    'SyntheticGenerator',
    'BlockDriftGenerator', 
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