#!/usr/bin/env python3
"""
Gradual Drift Implementation for River Generators
==================================================

Sistema completo para introducir drift gradual en generadores River:
- Drift gradual por interpolación
- Múltiples tipos de transiciones (lineal, exponencial, sigmoide)
- Drift en características, clases y distribuciones
- Compatible con cualquier generador River

Autor: CalmOps Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from river.datasets import synth
from typing import Dict, Any, List, Optional, Iterator, Tuple, Union, Callable
from enum import Enum
import math

class DriftType(Enum):
    """Tipos de transición de drift"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"  
    SIGMOID = "sigmoid"
    SUDDEN = "sudden"
    INCREMENTAL = "incremental"

class GradualDriftWrapper:
    """
    Wrapper para introducir drift gradual en cualquier generador River
    """
    
    def __init__(self, 
                 base_generator,
                 drift_start: int = 500,
                 drift_duration: int = 200,
                 drift_type: DriftType = DriftType.LINEAR,
                 feature_drift_intensity: Dict[str, float] = None,
                 class_drift_probability: float = 0.0,
                 noise_drift_factor: float = 0.0,
                 seed: Optional[int] = None):
        """
        Args:
            base_generator: Generador River base
            drift_start: Posición donde comienza el drift
            drift_duration: Duración del período de transición
            drift_type: Tipo de transición gradual
            feature_drift_intensity: Intensidad de drift por característica {feature: intensity}
            class_drift_probability: Probabilidad de cambio de clase durante drift
            noise_drift_factor: Factor de incremento de ruido durante drift
            seed: Semilla para reproducibilidad
        """
        self.base_generator = base_generator
        self.drift_start = drift_start
        self.drift_duration = drift_duration
        self.drift_type = drift_type
        self.feature_drift_intensity = feature_drift_intensity or {}
        self.class_drift_probability = class_drift_probability
        self.noise_drift_factor = noise_drift_factor
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Estado interno
        self.current_sample = 0
        self.original_features = None
        self.drift_end = drift_start + drift_duration
        
        # Análisis inicial del generador
        self._analyze_generator()
        
        # Configurar drift por defecto si no se especifica
        self._setup_default_drift()
    
    def _analyze_generator(self):
        """Analizar estructura del generador base"""
        sample = next(iter(self.base_generator.take(1)))
        self.original_features = list(sample[0].keys())
        self.sample_y_type = type(sample[1])
        
        # Resetear generador
        try:
            if hasattr(self.base_generator, 'seed') and self.base_generator.seed is not None:
                generator_class = type(self.base_generator)
                params = self._extract_generator_params()
                self.base_generator = generator_class(**params)
        except:
            pass
    
    def _extract_generator_params(self) -> Dict[str, Any]:
        """Extraer parámetros del generador para reseteo"""
        params = {}
        
        common_params = ['seed', 'random_state', 'classification_function', 
                        'balance_classes', 'perturbation', 'variant', 'noise',
                        'n_features', 'mag_change', 'sigma', 'seed_tree', 'seed_sample']
        
        for param in common_params:
            if hasattr(self.base_generator, param):
                params[param] = getattr(self.base_generator, param)
        
        return params
    
    def _setup_default_drift(self):
        """Configure default drift based on generator type"""
        if not self.feature_drift_intensity:
            # Default drift for all numeric features
            for feature in self.original_features:
                # Moderate intensity by default
                self.feature_drift_intensity[feature] = 0.2
    
    def _calculate_drift_progress(self, sample_idx: int) -> float:
        """
        Calculate drift progress (0.0 = start, 1.0 = end)
        """
        if sample_idx < self.drift_start:
            return 0.0
        elif sample_idx >= self.drift_end:
            return 1.0
        else:
            # Posición relativa dentro del período de drift
            relative_pos = (sample_idx - self.drift_start) / self.drift_duration
            
            # Aplicar función de transición según el tipo
            if self.drift_type == DriftType.LINEAR:
                return relative_pos
            elif self.drift_type == DriftType.EXPONENTIAL:
                return 1.0 - math.exp(-3 * relative_pos)  # Rápido al inicio, lento al final
            elif self.drift_type == DriftType.SIGMOID:
                # Transición suave en forma de S
                x = (relative_pos - 0.5) * 6  # Centrar y escalar
                return 1.0 / (1.0 + math.exp(-x))
            elif self.drift_type == DriftType.SUDDEN:
                return 1.0 if relative_pos > 0.5 else 0.0
            elif self.drift_type == DriftType.INCREMENTAL:
                # Discrete step drift
                steps = 5
                step_size = 1.0 / steps
                return min(1.0, math.floor(relative_pos * steps) * step_size + step_size)
            else:
                return relative_pos
    
    def _apply_feature_drift(self, x_dict: Dict, drift_progress: float) -> Dict:
        """Apply gradual drift to features"""
        drifted_x = x_dict.copy()
        
        for feature, intensity in self.feature_drift_intensity.items():
            if feature in x_dict:
                original_value = x_dict[feature]
                
                if isinstance(original_value, (int, float)):
                    # Numeric drift
                    drift_amount = intensity * drift_progress * original_value
                    
                    # Different transformation types
                    transformation = hash(feature) % 4
                    
                    if transformation == 0:  # Shift
                        drifted_x[feature] = original_value + drift_amount
                    elif transformation == 1:  # Escala
                        scale_factor = 1.0 + (intensity * drift_progress)
                        drifted_x[feature] = original_value * scale_factor
                    elif transformation == 2:  # Rotación/transformación no-lineal
                        if original_value != 0:
                            sign_flip = 1.0 - 2.0 * intensity * drift_progress
                            drifted_x[feature] = original_value * abs(sign_flip)
                    else:  # Combinación
                        noise_scale = max(0.001, abs(original_value) * intensity * drift_progress * 0.1)
                        drift_noise = np.random.normal(0, noise_scale)
                        drifted_x[feature] = original_value * (1.0 + intensity * drift_progress * 0.5) + drift_noise
                
                elif isinstance(original_value, (bool, int)) and original_value in [0, 1]:
                    # Drift categórico binario
                    if np.random.random() < intensity * drift_progress * 0.1:
                        drifted_x[feature] = 1 - original_value
        
        return drifted_x
    
    def _apply_class_drift(self, y, drift_progress: float):
        """Aplicar drift gradual a las etiquetas/clases"""
        if self.class_drift_probability == 0.0:
            return y
        
        # Probabilidad de cambio basada en progreso del drift
        change_prob = self.class_drift_probability * drift_progress
        
        if np.random.random() < change_prob:
            if isinstance(y, bool):
                return not y
            elif isinstance(y, int) and y in [0, 1]:
                return 1 - y
            elif isinstance(y, int):
                # Para multi-clase, cambiar a clase aleatoria diferente
                possible_classes = list(range(max(2, y + 2)))
                possible_classes.remove(y)
                return np.random.choice(possible_classes)
        
        return y
    
    def _apply_noise_drift(self, x_dict: Dict, drift_progress: float) -> Dict:
        """Aplicar incremento gradual de ruido"""
        if self.noise_drift_factor == 0.0:
            return x_dict
        
        noisy_x = x_dict.copy()
        
        for feature, value in x_dict.items():
            if isinstance(value, (int, float)):
                noise_intensity = self.noise_drift_factor * drift_progress
                noise_scale = max(0.001, abs(value) * noise_intensity)
                noise = np.random.normal(0, noise_scale)
                noisy_x[feature] = value + noise
        
        return noisy_x
    
    def take(self, n: int) -> Iterator[Tuple[Dict, Any]]:
        """Generar n muestras con drift gradual"""
        for x, y in self.base_generator.take(n):
            # Calcular progreso del drift
            drift_progress = self._calculate_drift_progress(self.current_sample)
            
            # Aplicar transformaciones graduales
            drifted_x = self._apply_feature_drift(x, drift_progress)
            drifted_x = self._apply_noise_drift(drifted_x, drift_progress)
            drifted_y = self._apply_class_drift(y, drift_progress)
            
            self.current_sample += 1
            yield drifted_x, drifted_y
    
    def get_drift_info(self) -> Dict[str, Any]:
        """Obtener información sobre la configuración del drift"""
        return {
            'drift_start': self.drift_start,
            'drift_duration': self.drift_duration,
            'drift_end': self.drift_end,
            'drift_type': self.drift_type.value,
            'feature_drift_intensity': self.feature_drift_intensity,
            'class_drift_probability': self.class_drift_probability,
            'noise_drift_factor': self.noise_drift_factor,
            'current_sample': self.current_sample
        }

