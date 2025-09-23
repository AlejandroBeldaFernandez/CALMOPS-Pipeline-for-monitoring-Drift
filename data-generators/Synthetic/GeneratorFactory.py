"""
Generator Factory for CalmOps Synthetic Data Generators
=======================================================
This module provides a factory pattern for creating various River synthetic data generators
with standardized configuration and type safety.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from river.datasets import synth


class GeneratorType(Enum):
    """Enumeration of available generator types"""
    AGRAWAL = "agrawal"
    SEA = "sea"
    HYPERPLANE = "hyperplane"
    RANDOM_TREE = "random_tree"
    STAGGER = "stagger"
    SINE = "sine"
    MIXED = "mixed"
    FRIEDMAN = "friedman"
    RANDOM_RBF = "random_rbf"


@dataclass
class GeneratorConfig:
    """Configuration class for generator parameters"""
    # Common parameters
    random_state: Optional[int] = None
    seed: Optional[int] = None  # Alias for random_state
    
    # Agrawal specific
    classification_function: int = 0
    balance_classes: bool = True
    perturbation: float = 0.0
    
    # SEA specific
    function: int = 0
    noise_percentage: float = 0.1
    
    # Hyperplane specific
    n_features: int = 10
    n_dims: Optional[int] = None  # Alias for n_features
    mag_change: float = 0.0
    noise_percentage_hyperplane: float = 0.05
    sigma: float = 0.1
    
    # Random Tree specific
    n_num_features: int = 5
    n_cat_features: int = 5
    n_categories_per_cat_feature: int = 5
    max_tree_depth: int = 5
    first_leaf_label: int = 1
    
    # Stagger specific
    classification_function_stagger: int = 0
    balance_classes_stagger: bool = True
    
    # Sine specific
    has_noise: bool = False
    noise_percentage_sine: float = 0.1
    
    # Mixed specific
    classification_function_mixed: int = 0
    balance_classes_mixed: bool = True
    
    # Friedman specific
    n_features_friedman: int = 10
    
    # Random RBF specific
    n_features_rbf: int = 10
    n_centroids: int = 50
    
    def __post_init__(self):
        """Post-initialization to handle aliases and validation"""
        # Handle seed/random_state alias
        if self.seed is not None and self.random_state is None:
            self.random_state = self.seed
        elif self.random_state is not None and self.seed is None:
            self.seed = self.random_state
        
        # Handle n_dims/n_features alias for hyperplane
        if self.n_dims is not None and self.n_features == 10:  # 10 is default
            self.n_features = self.n_dims
        elif self.n_features != 10 and self.n_dims is None:
            self.n_dims = self.n_features


class GeneratorFactory:
    """Factory class for creating River synthetic data generators"""
    
    @staticmethod
    def get_available_generators() -> List[GeneratorType]:
        """Returns list of all available generator types"""
        return list(GeneratorType)
    
    @staticmethod
    def create_generator(generator_type: GeneratorType, config: GeneratorConfig):
        """
        Creates a generator instance based on type and configuration
        
        Args:
            generator_type: Type of generator to create
            config: Configuration object with parameters
            
        Returns:
            Generator instance ready to use
        """
        generators = {
            GeneratorType.AGRAWAL: GeneratorFactory._create_agrawal,
            GeneratorType.SEA: GeneratorFactory._create_sea,
            GeneratorType.HYPERPLANE: GeneratorFactory._create_hyperplane,
            GeneratorType.RANDOM_TREE: GeneratorFactory._create_random_tree,
            GeneratorType.STAGGER: GeneratorFactory._create_stagger,
            GeneratorType.SINE: GeneratorFactory._create_sine,
            GeneratorType.MIXED: GeneratorFactory._create_mixed,
            GeneratorType.FRIEDMAN: GeneratorFactory._create_friedman,
            GeneratorType.RANDOM_RBF: GeneratorFactory._create_random_rbf,
        }
        
        if generator_type not in generators:
            raise ValueError(f"Unknown generator type: {generator_type}")
            
        return generators[generator_type](config)
    
    @staticmethod
    def _create_agrawal(config: GeneratorConfig):
        """Creates Agrawal generator"""
        params = {
            'classification_function': config.classification_function,
            'balance_classes': config.balance_classes,
            'perturbation': config.perturbation
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.Agrawal(**params)
    
    @staticmethod
    def _create_sea(config: GeneratorConfig):
        """Creates SEA generator"""
        params = {
            'variant': config.function,
            'noise': config.noise_percentage
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.SEA(**params)
    
    @staticmethod
    def _create_hyperplane(config: GeneratorConfig):
        """Creates Hyperplane generator"""
        params = {
            'n_features': config.n_features,
            'mag_change': config.mag_change,
            'sigma': config.sigma
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.Hyperplane(**params)
    
    @staticmethod
    def _create_random_tree(config: GeneratorConfig):
        """Creates Random Tree generator"""
        params = {
            'n_num_features': config.n_num_features,
            'n_cat_features': config.n_cat_features,
            'n_categories_per_feature': config.n_categories_per_cat_feature,
            'max_tree_depth': config.max_tree_depth,
            'first_leaf_level': config.first_leaf_label
        }
        if config.random_state is not None:
            params['seed_tree'] = config.random_state
            params['seed_sample'] = config.random_state
        return synth.RandomTree(**params)
    
    @staticmethod
    def _create_stagger(config: GeneratorConfig):
        """Creates Stagger generator"""
        params = {
            'classification_function': config.classification_function_stagger,
            'balance_classes': config.balance_classes_stagger
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.STAGGER(**params)
    
    @staticmethod
    def _create_sine(config: GeneratorConfig):
        """Creates Sine generator"""
        params = {
            'has_noise': config.has_noise
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.Sine(**params)
    
    @staticmethod
    def _create_mixed(config: GeneratorConfig):
        """Creates Mixed generator"""
        params = {
            'classification_function': config.classification_function_mixed,
            'balance_classes': config.balance_classes_mixed
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.Mixed(**params)
    
    @staticmethod
    def _create_friedman(config: GeneratorConfig):
        """Creates Friedman generator"""
        params = {
            'n_features': config.n_features_friedman
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.Friedman(**params)
    
    @staticmethod
    def _create_random_rbf(config: GeneratorConfig):
        """Creates Random RBF generator"""
        params = {
            'n_features': config.n_features_rbf,
            'n_centroids': config.n_centroids
        }
        if config.random_state is not None:
            params['seed'] = config.random_state
        return synth.RandomRBF(**params)
    
    @staticmethod
    def create_preset_concept_drift(generator_type: GeneratorType, 
                                  seed: int = 42, 
                                  drift_magnitude: float = 0.5):
        """
        Creates a pair of generators suitable for concept drift testing
        
        Args:
            generator_type: Base generator type
            seed: Random seed for reproducibility
            drift_magnitude: Magnitude of the drift between generators
            
        Returns:
            Tuple of (base_generator, drift_generator)
        """
        base_config = GeneratorConfig(random_state=seed)
        drift_config = GeneratorConfig(random_state=seed + 1)
        
        if generator_type == GeneratorType.AGRAWAL:
            drift_config.classification_function = 1
        elif generator_type == GeneratorType.SEA:
            drift_config.function = 1
            drift_config.noise_percentage = base_config.noise_percentage + drift_magnitude * 0.1
        elif generator_type == GeneratorType.HYPERPLANE:
            drift_config.mag_change = drift_magnitude
        elif generator_type == GeneratorType.STAGGER:
            drift_config.classification_function_stagger = 1
        
        base_gen = GeneratorFactory.create_generator(generator_type, base_config)
        drift_gen = GeneratorFactory.create_generator(generator_type, drift_config)
        
        return base_gen, drift_gen
    
    @staticmethod
    def get_generator_info(generator_type: GeneratorType) -> Dict[str, Any]:
        """Returns information about a specific generator type"""
        info = {
            GeneratorType.AGRAWAL: {
                "name": "Agrawal",
                "description": "Classification dataset with multiple classification functions",
                "features": 9,
                "classes": 2,
                "drift_capable": True
            },
            GeneratorType.SEA: {
                "name": "SEA Concepts",
                "description": "Streaming Ensemble Algorithm concepts with noise",
                "features": 3,
                "classes": 2,
                "drift_capable": True
            },
            GeneratorType.HYPERPLANE: {
                "name": "Hyperplane",
                "description": "Hyperplane-based classification with concept drift",
                "features": "configurable",
                "classes": 2,
                "drift_capable": True
            },
            GeneratorType.RANDOM_TREE: {
                "name": "Random Tree",
                "description": "Random tree-based classification",
                "features": "configurable",
                "classes": 2,
                "drift_capable": False
            },
            GeneratorType.STAGGER: {
                "name": "STAGGER Concepts",
                "description": "STAGGER concept classification problems",
                "features": 3,
                "classes": 2,
                "drift_capable": True
            },
            GeneratorType.SINE: {
                "name": "Sine",
                "description": "Sine-wave based regression with optional noise",
                "features": 2,
                "classes": "regression",
                "drift_capable": False
            },
            GeneratorType.MIXED: {
                "name": "Mixed",
                "description": "Mixed classification functions",
                "features": 4,
                "classes": 2,
                "drift_capable": False
            },
            GeneratorType.FRIEDMAN: {
                "name": "Friedman",
                "description": "Friedman synthetic regression dataset",
                "features": "configurable",
                "classes": "regression",
                "drift_capable": False
            },
            GeneratorType.RANDOM_RBF: {
                "name": "Random RBF",
                "description": "Random Radial Basis Function centers",
                "features": "configurable",
                "classes": "variable",
                "drift_capable": False
            }
        }
        return info.get(generator_type, {"name": "Unknown", "description": "Unknown generator"})