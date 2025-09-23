"""
Real Data Generators Package

This package provides enhanced real data processing capabilities with:
- RealGenerator: Base real data synthesis with multiple methods
- RealBlockGenerator: Block-wise real data processing
- RealReporter: Comprehensive reporting and visualization

Features:
- Multiple synthesis methods (GMM, CTGAN, Copula, SMOTE, Resample)
- Enhanced drift injection capabilities
- Comprehensive statistical validation
- Modern visualization with improved styling
- Block-based processing and analysis
"""

from .RealGenerator import RealGenerator
from .RealBlockGenerator import RealBlockGenerator  
from .RealReporter import RealReporter

__all__ = [
    'RealGenerator',
    'RealBlockGenerator', 
    'RealReporter'
]

__version__ = '1.0.0'