# calmops/IPIP/ipip_model.py
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

class IpipModel:
    """
    Represents the IPIP model structure, which is an ensemble of ensembles.
    
    Attributes:
        ensembles_ (List[List[Any]]): A list of ensembles, where each ensemble is a list of base models.
    """
    def __init__(self, ensembles: List[List[Any]] | None = None):
        self.ensembles_: List[List[Any]] = ensembles if ensembles is not None else []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class probabilities by averaging the predictions of all base models
        across all ensembles.
        
        Args:
            X (pd.DataFrame): The input features.
            
        Returns:
            np.ndarray: An array of shape (n_samples, n_classes) with the predicted
                        probabilities.
        """
        if not self.ensembles_ or not any(self.ensembles_):
            raise ValueError("Model has not been trained or has no ensembles.")

        all_probas = []
        for ensemble in self.ensembles_:
            for model in ensemble:
                # Assuming the model has a standard predict_proba method
                if hasattr(model, "predict_proba"):
                    all_probas.append(model.predict_proba(X))
        
        if not all_probas:
            raise ValueError("None of the base models in the ensemble have a 'predict_proba' method.")
            
        # Average the probabilities across all models
        mean_probas = np.mean(all_probas, axis=0)
        return mean_probas

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the class by taking the argmax of the predicted probabilities.
        
        Args:
            X (pd.DataFrame): The input features.
            
        Returns:
            np.ndarray: An array of shape (n_samples,) with the predicted class labels.
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def __len__(self):
        return len(self.ensembles_)
