import joblib
import os
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    classification_report, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import is_classifier, is_regressor
from scikeras.wrappers import KerasClassifier
from torch.nn import Module
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.base import clone
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
from datetime import datetime

def train(X, y, last_processed_file, logger, output_dir):
    logger.info(">>> Entrando en el método de entrenamiento con Random Forest")
    
    # División de datos (80% entrenamiento, 20% evaluación)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    try:
        # Instanciamos el modelo de Random Forest
        model = RandomForestClassifier(random_state=42)
        
        # Entrenamos el modelo
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_eval)

        # Métricas
        metrics = {
            "balanced_accuracy": balanced_accuracy_score(y_eval, y_pred),
            "classification_report": classification_report(y_eval, y_pred, output_dict=True)
        }

        # Resultados
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tipo": "train",
            "modelo": str(model),
            "archivo": last_processed_file,
            **metrics
        }

        # Guardamos los resultados
        guardar_train_results(results, output_dir)
        
        return model, X_eval, y_eval, results

    except Exception as e:
        logger.error(f"Error en el método de entrenamiento: {e}")
        raise

# Función para guardar resultados (como ejemplo)
def guardar_train_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "train_results.json")
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Resultados guardados en {result_path}")
