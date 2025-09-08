"""
Real Data Generation Module for CalmOps
=======================================

This module provides capabilities for generating synthetic data from real datasets using:
- RealGenerator: Core real data synthesis with multiple algorithms (SMOTE, GMM, CTGAN, etc.)
- RealReporter: Quality assessment and comparison tools for real vs synthetic data

Supported synthesis methods:
- resample: Simple resampling with replacement
- smote: SMOTE (Synthetic Minority Oversampling Technique)
- gmm: Gaussian Mixture Models
- ctgan: Conditional Tabular GAN (requires SDV)
- copula: Gaussian Copula (requires SDV)

Example usage:
    from Real.RealGenerator import RealGenerator
    
    # Initialize with dataset
    generator = RealGenerator(df=my_dataframe, target_col="target")
    
    # Generate synthetic data
    generator.generate("output/", "synthetic.csv", 1000, method="smote")
"""

from .RealGenerator import RealGenerator
from .RealReporter import RealReporter
from .DriftInjector import DriftInjector
from .DistributionChanger import DistributionChanger
from .BlockDriftGenerator import RealBlockDriftGenerator

__all__ = [
    'RealGenerator',
    'RealReporter',
    'DriftInjector',
    'DistributionChanger',
    'RealBlockDriftGenerator'
]

__version__ = "1.0.0" 
__author__ = "CalmOps Team"
__description__ = "Real data synthesis and quality assessment framework"