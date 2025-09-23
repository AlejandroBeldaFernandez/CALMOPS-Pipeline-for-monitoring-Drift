#!/usr/bin/env python3
"""
Universal Generator Wrapper for River Synthetic Data Generators
================================================================

Wrapper universal que permite personalizar CUALQUIER generador River:
- Cambiar nombres de características para TODOS los generadores
- Controlar número de características (con sampling/padding cuando sea necesario)
- Funciona con todos los generadores de river.datasets.synth
- AUTO-VISUALIZATION: Genera automáticamente dashboards de calidad

Autor: CalmOps Team
"""

import pandas as pd
import numpy as np
from river.datasets import synth
from typing import Dict, Any, List, Optional, Iterator, Tuple, Union
import random

# Import auto-visualization system
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Visualization.AutoVisualizer import AutoVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class UniversalGeneratorWrapper:
    """
    Wrapper universal para personalizar cualquier generador River
    """
    
    def __init__(self, 
                 base_generator, 
                 custom_feature_names: Optional[List[str]] = None,
                 n_features_target: Optional[int] = None,
                 feature_selection_strategy: str = 'first',
                 padding_strategy: str = 'noise',
                 seed: Optional[int] = None,
                 enable_auto_visualization: bool = True):
        """
        Args:
            base_generator: Cualquier generador River
            custom_feature_names: Lista de nombres personalizados
            n_features_target: Número objetivo de características (si es diferente del original)
            feature_selection_strategy: 'first', 'random', 'last' para reducir características
            padding_strategy: 'noise', 'zeros', 'repeat' para añadir características  
            seed: Semilla para reproducibilidad
        """
        self.base_generator = base_generator
        self.seed = seed
        self.enable_auto_visualization = enable_auto_visualization and VISUALIZATION_AVAILABLE
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        if self.enable_auto_visualization:
            print("Auto-visualization enabled for UniversalGeneratorWrapper")
        
        # Detectar estructura del generador
        self._analyze_generator_structure()
        
        # Configurar nombres de características
        self._setup_feature_names(custom_feature_names)
        
        # Configurar número de características objetivo
        self._setup_feature_count(n_features_target, feature_selection_strategy, padding_strategy)
    
    def _analyze_generator_structure(self):
        """Analizar la estructura del generador base"""
        # Tomar una muestra para entender la estructura
        sample = next(iter(self.base_generator.take(1)))
        
        self.original_features = list(sample[0].keys())
        self.original_n_features = len(self.original_features)
        self.sample_x_type = type(sample[0])
        self.sample_y_type = type(sample[1])
        
        # Detectar tipos de características
        self.feature_types = {}
        for key, value in sample[0].items():
            self.feature_types[key] = type(value)
        
        # Resetear generador si es posible
        self._reset_generator_if_possible()
    
    def _reset_generator_if_possible(self):
        """Intentar resetear el generador para mantener determinismo"""
        try:
            # Recrear generador con mismos parámetros si tiene seed
            if hasattr(self.base_generator, 'seed') and self.base_generator.seed is not None:
                generator_class = type(self.base_generator)
                params = self._extract_all_generator_params()
                self.base_generator = generator_class(**params)
        except Exception:
            # Si no se puede resetear, continuar con el existente
            pass
    
    def _extract_all_generator_params(self) -> Dict[str, Any]:
        """Extract all possible generator parameters"""
        params = {}
        
        # Complete list of possible River generator parameters
        possible_params = [
            'seed', 'random_state',
            # Agrawal
            'classification_function', 'balance_classes', 'perturbation',
            # SEA  
            'variant', 'noise',
            # Hyperplane
            'n_features', 'mag_change', 'sigma',
            # Random Tree
            'n_num_features', 'n_cat_features', 'n_categories_per_cat_feature',
            'leaf_fraction', 'drift_fraction',
            # Stagger
            'balance_classes',
            # Sine
            'has_noise',
            # Mixed
            'classification_function', 'balance_classes',
            # Friedman
            'n_features',
            # Random RBF
            'n_features', 'n_centroids'
        ]
        
        for param in possible_params:
            if hasattr(self.base_generator, param):
                params[param] = getattr(self.base_generator, param)
        
        return params
    
    def _setup_feature_names(self, custom_feature_names: Optional[List[str]]):
        """Configurar nombres de características personalizados"""
        if custom_feature_names is None:
            # Generar nombres por defecto basados en el tipo de generador
            generator_name = type(self.base_generator).__name__.lower()
            self.target_feature_names = [f'{generator_name}_feat_{i+1}' for i in range(self.original_n_features)]
        else:
            self.target_feature_names = custom_feature_names.copy()
    
    def _setup_feature_count(self, n_features_target: Optional[int], 
                           selection_strategy: str, padding_strategy: str):
        """Configure target number of features"""
        self.n_features_target = n_features_target or self.original_n_features
        self.selection_strategy = selection_strategy
        self.padding_strategy = padding_strategy
        
        # Ajustar nombres si el número objetivo es diferente
        if len(self.target_feature_names) != self.n_features_target:
            if len(self.target_feature_names) > self.n_features_target:
                # Recortar nombres
                self.target_feature_names = self.target_feature_names[:self.n_features_target]
            else:
                # Extender nombres
                current_count = len(self.target_feature_names)
                generator_name = type(self.base_generator).__name__.lower()
                for i in range(current_count, self.n_features_target):
                    self.target_feature_names.append(f'{generator_name}_feat_{i+1}')
    
    def _adjust_feature_count(self, x_dict: Dict) -> Dict:
        """Ajustar el número de características según la configuración"""
        original_values = list(x_dict.values())
        
        if self.n_features_target == len(original_values):
            # No hay cambio necesario, solo renombrar
            return dict(zip(self.target_feature_names, original_values))
        
        elif self.n_features_target < len(original_values):
            # Reducir características
            if self.selection_strategy == 'first':
                selected_values = original_values[:self.n_features_target]
            elif self.selection_strategy == 'last':
                selected_values = original_values[-self.n_features_target:]
            elif self.selection_strategy == 'random':
                if self.seed is not None:
                    np.random.seed(self.seed + int(sum(original_values)) % 1000)
                indices = np.random.choice(len(original_values), self.n_features_target, replace=False)
                selected_values = [original_values[i] for i in sorted(indices)]
            else:
                selected_values = original_values[:self.n_features_target]
            
            return dict(zip(self.target_feature_names, selected_values))
        
        else:
            # Añadir características
            extended_values = original_values.copy()
            n_to_add = self.n_features_target - len(original_values)
            
            if self.padding_strategy == 'zeros':
                extended_values.extend([0.0] * n_to_add)
            elif self.padding_strategy == 'repeat':
                # Repetir las últimas características
                for i in range(n_to_add):
                    extended_values.append(original_values[i % len(original_values)])
            elif self.padding_strategy == 'noise':
                # Añadir características con ruido basado en las existentes
                for i in range(n_to_add):
                    base_value = original_values[i % len(original_values)]
                    if isinstance(base_value, (int, float)):
                        # Ruido gaussiano proporcional al valor
                        noise_scale = abs(base_value) * 0.1 + 0.01
                        noise_value = base_value + np.random.normal(0, noise_scale)
                        extended_values.append(noise_value)
                    else:
                        # Para valores categóricos, usar el valor original
                        extended_values.append(base_value)
            
            return dict(zip(self.target_feature_names, extended_values))
    
    def take(self, n: int) -> Iterator[Tuple[Dict, Any]]:
        """Generar n muestras con características personalizadas"""
        for x, y in self.base_generator.take(n):
            adjusted_x = self._adjust_feature_count(x)
            yield adjusted_x, y
    
    def __iter__(self):
        return self
    
    def __next__(self):
        sample = next(iter(self.base_generator.take(1)))
        adjusted_x = self._adjust_feature_count(sample[0])
        return adjusted_x, sample[1]
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información sobre la configuración del wrapper"""
        return {
            'generator_type': type(self.base_generator).__name__,
            'original_features': self.original_features,
            'original_n_features': self.original_n_features,
            'target_feature_names': self.target_feature_names,
            'target_n_features': self.n_features_target,
            'selection_strategy': self.selection_strategy,
            'padding_strategy': self.padding_strategy,
            'feature_types': self.feature_types
        }
    
    def generate_and_visualize(self, n_samples: int, generator_name: Optional[str] = None,
                             output_dir: str = "universal_wrapper_output") -> Dict[str, Any]:
        """
        Generate data and automatically create visualizations
        
        Args:
            n_samples: Number of samples to generate
            generator_name: Name for the generator (auto-detected if None)
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with generation results and visualization info
        """
        
        if generator_name is None:
            base_name = type(self.base_generator).__name__
            generator_name = f"{base_name}_CustomWrapper"
        
        print(f"\nGenerating {n_samples} samples with UniversalGeneratorWrapper...")
        print(f"Custom Features: {self.target_feature_names[:5]}{'...' if len(self.target_feature_names) > 5 else ''}")
        print(f"Target Feature Count: {self.n_features_target}")
        
        # Generate data
        data = list(self.take(n_samples))
        
        print(f"Generated: {len(data)} samples")
        
        # Auto-visualization if enabled
        if self.enable_auto_visualization:
            try:
                viz_results = AutoVisualizer.auto_analyze_and_visualize(
                    data, generator_name, output_dir
                )
                
                print(f"Generated {len(viz_results['visualization_files'])} visualization plots")
                
                return {
                    'data': data,
                    'n_samples': len(data),
                    'feature_names': self.target_feature_names,
                    'visualization_results': viz_results
                }
                
            except Exception as e:
                print(f"Auto-visualization failed: {e}")
                return {
                    'data': data,
                    'n_samples': len(data),
                    'feature_names': self.target_feature_names,
                    'visualization_results': None
                }
        
        else:
            return {
                'data': data,
                'n_samples': len(data), 
                'feature_names': self.target_feature_names,
                'visualization_results': None
            }
