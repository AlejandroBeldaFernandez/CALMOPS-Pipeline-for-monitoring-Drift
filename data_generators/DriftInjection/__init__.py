"""
Drift Injection Module for CalmOps
==================================

This module provides comprehensive drift injection capabilities that work seamlessly
with both Real and Synthetic data generation modules.

Key Features:
- Unified API for all drift types
- Multi-column support with per-column parameter control
- Advanced correlation manipulation
- Robust validation and error handling

Supported Drift Types:
- Feature Drift: gaussian_noise, shift, scale, add_value, subtract_value, multiply_value, divide_value
- Label Drift: Random label flipping with multi-target support
- Target Distribution Drift: Controlled class proportion changes
- Covariate Shift: Enhanced correlation modifications between feature pairs
- Advanced Covariate Shift: Cholesky decomposition for precise correlation control

Example usage:
    from DriftInjection import DriftInjector
    
    # Initialize injector
    injector = DriftInjector(random_state=42)
    
    # Single column drift
    drifted_df = injector.inject_feature_drift(
        df=my_dataframe,
        feature_cols=['feature1'],
        drift_type='add_value',
        drift_value=5,
        start_index=500
    )
    
    # Multi-column drift with different values
    drifted_df = injector.inject_feature_drift(
        df=my_dataframe,
        feature_cols=['feature1', 'feature2', 'feature3'],
        drift_type='add_value',
        drift_values={'feature1': 5, 'feature2': -2, 'feature3': 1.5},
        start_index=500
    )
    
    # Advanced covariate shift
    drifted_df = injector.inject_covariate_shift(
        df=my_dataframe,
        feature_cols=['f1', 'f2', 'f3', 'f4'],
        feature_pairs=[('f1', 'f2'), ('f3', 'f4')],
        correlation_changes={('f1', 'f2'): 0.5, ('f3', 'f4'): -0.3},
        start_index=500
    )

Integration:
    This module is automatically imported by both Real and Synthetic modules,
    providing a unified drift injection experience across all data sources.
"""

from .DriftInjector import DriftInjector

__all__ = ['DriftInjector']

__version__ = "1.0.0"
__author__ = "CalmOps Team"
__description__ = "Unified drift injection framework for Real and Synthetic data"