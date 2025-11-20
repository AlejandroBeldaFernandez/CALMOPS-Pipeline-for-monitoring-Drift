# IPIP/modules/default_train_retrain.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

import joblib
from pathlib import Path

from ..ipip_model import IpipModel


def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.ndarray, list, tuple)): return [_jsonable(x) for x in o]
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    return o


def _save_results(path: Path, payload: Dict[str, Any], logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=4, ensure_ascii=False)
    logger.info(f"[TRAIN] Results saved at {path.resolve()}")


def _blocks_in_order(X: pd.DataFrame, block_col: str) -> List[str]:
    return [str(x) for x in pd.unique(X[block_col]).tolist()]


def _get_metric(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    return balanced_accuracy_score(y_true, y_pred)


def _predict_ensemble_proba(ensemble: List[Any], X: pd.DataFrame) -> np.ndarray:
    if not ensemble:
        # Return a dummy probability array that will result in a low metric
        return np.zeros((X.shape[0], 2))
        
    probas = [model.predict_proba(X) for model in ensemble]
    return np.mean(probas, axis=0)


def best_models(
    model: IpipModel,
    metric_max: str, # e.g. "BA"
    x: pd.DataFrame,
    y: pd.Series,
    p: int,
    logger
) -> List[int]:
    """
    Evaluates all ensembles in the IpipModel and returns the indices of the best 'p' ensembles.
    """
    logger.info(f"Finding best {p} models from {len(model.ensembles_)} ensembles.")
    model_performance = []

    for j, ensemble in enumerate(model.ensembles_):
        try:
            # Here we assume the ensemble prediction is an average of probabilities
            results_prob = _predict_ensemble_proba(ensemble, x)
            # The metric is calculated based on the predicted class
            y_pred = np.argmax(results_prob, axis=1)
            
            # Assuming y has class labels that need to be mapped to 0, 1...
            # This part is tricky without knowing the exact classes.
            # We'll assume the classes are already encoded if they are not 0,1
            y_true = y.values
            
            # For simplicity, let's assume the metric is balanced accuracy
            perf = balanced_accuracy_score(y_true, y_pred)
            model_performance.append(perf)
        except Exception as e:
            logger.error(f"Error evaluating ensemble {j}: {e}")
            model_performance.append(-1.0) # Penalize failing ensembles

    # Sort indices by performance (higher is better)
    sorted_indices = np.argsort(model_performance)[::-1]
    
    # Return the top 'p' indices
    return sorted_indices[:p].tolist()


# =========================================================
# IPIP TRAIN (based on R code)
# =========================================================
def default_train(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    last_processed_file: str,
    model_instance,
    random_state: int,
    logger,
    output_dir: str,
    block_col: str,
    ipip_config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    if model_instance is None:
        raise ValueError("[TRAIN] 'model_instance' is mandatory for initial training.")
    if ipip_config is None:
        raise ValueError("[TRAIN] 'ipip_config' is mandatory for IPIP training.")

    t0 = time.time()
    
    # --- Config from ipip_config ---
    p = ipip_config.get("p", 5)
    b = ipip_config.get("b", 10)
    majority_prop = ipip_config.get("majority_prop", 0.55)
    max_attempts = ipip_config.get("max_attempts", 5)
    val_size = ipip_config.get("val_size", 0.2)
    prop_minor_frac = ipip_config.get("prop_minor_frac", 0.75)
    target_col = y.name

    logger.info(f"Starting IPIP train with config: p={p}, b={b}, majority_prop={majority_prop}")

    # The R code trains on the first chunk and evaluates on the second.
    # Our pipeline gives us all data, so we'll simulate this.
    blocks = _blocks_in_order(X, block_col)
    if len(blocks) < 2:
        raise ValueError("IPIP training requires at least 2 blocks (1 for train, 1 for test).")

    train_block = blocks[0]
    test_block = blocks[1]
    
    current_chunk = pd.concat([X[X[block_col] == train_block], y], axis=1)
    next_chunk = pd.concat([X[X[block_col] == test_block], y], axis=1)

    # Split current chunk into train/test for building the ensembles
    train_df, test_df = train_test_split(current_chunk, test_size=val_size, random_state=random_state, stratify=current_chunk[target_col])
    
    X_test_ensemble = test_df.drop(columns=[target_col, block_col])
    y_test_ensemble = test_df[target_col]

    # --- Start of the main logic from R file ---
    
    discharge = train_df[train_df[target_col] == 0] # Assuming 'NO' class is 0
    expired = train_df[train_df[target_col] == 1]   # Assuming 'YES' class is 1

    if len(expired) == 0 or len(discharge) == 0:
        raise ValueError("Training data must contain samples from both classes.")

    np_val = round(len(expired) * prop_minor_frac)
    if np_val == 0:
        raise ValueError("Not enough samples in the minority class to create training sets.")

    # Create 'p' balanced datasets
    dfs = []
    for _ in range(p):
        id_ex = expired.sample(n=np_val, replace=True, random_state=random_state)
        n_dis = round(np_val * majority_prop / (1 - majority_prop))
        id_dis = discharge.sample(n=n_dis, replace=True, random_state=random_state)
        dfs.append(pd.concat([id_dis, id_ex]))

    E = [] # This will be the list of ensembles (our IpipModel)

    for k in range(p):
        logger.info(f"Building ensemble {k+1}/{p}...")
        Ek = []  # The k-th ensemble
        i = 0    # Attempts counter
        df_k = dfs[k]
        
        X_k = df_k.drop(columns=[target_col, block_col])
        y_k = df_k[target_col]

        while len(Ek) < b and i < max_attempts:
            # Train a new candidate model
            candidate_model = clone(model_instance)
            if hasattr(candidate_model, "random_state"):
                candidate_model.random_state = random_state
            
            # The R code creates a perfectly balanced set here, let's do that too
            g0 = df_k[df_k[target_col] == 0]
            g1 = df_k[df_k[target_col] == 1]
            n_min = min(len(g0), len(g1))
            if n_min == 0: 
                i += 1
                continue

            train_sample = pd.concat([
                g0.sample(n=n_min, replace=True, random_state=random_state + i),
                g1.sample(n=n_min, replace=True, random_state=random_state + i)
            ])
            X_train_sample = train_sample.drop(columns=[target_col, block_col])
            y_train_sample = train_sample[target_col]

            candidate_model.fit(X_train_sample, y_train_sample)

            # Evaluate current ensemble
            proba_current = _predict_ensemble_proba(Ek, X_test_ensemble)
            metric_current = _get_metric(y_test_ensemble, proba_current) if Ek else -1.0

            # Evaluate ensemble with the new candidate
            proba_new = _predict_ensemble_proba(Ek + [candidate_model], X_test_ensemble)
            metric_new = _get_metric(y_test_ensemble, proba_new)

            if metric_new > metric_current:
                Ek.append(candidate_model)
                i = 0 # Reset attempts
                logger.debug(f"  Ensemble {k+1} improved to BA={metric_new:.4f} with {len(Ek)} models.")
            else:
                i += 1 # Increment attempts
        
        E.append(Ek)
        logger.info(f"Finished building ensemble {k+1} with {len(Ek)} models.")

    final_model = IpipModel(ensembles=E)

    # Calculate internal validation metrics for each final ensemble
    ensemble_validation_metrics = []
    for k, ensemble in enumerate(final_model.ensembles_):
        try:
            proba = _predict_ensemble_proba(ensemble, X_test_ensemble)
            metric = _get_metric(y_test_ensemble, proba)
            ensemble_validation_metrics.append(metric)
        except Exception as e:
            logger.warning(f"Could not calculate validation metric for ensemble {k+1}: {e}")
            ensemble_validation_metrics.append(None)

    # The pipeline expects (model, X_test, y_test, results)
    # The "test" set for the pipeline is the *next* chunk
    X_test_pipeline = next_chunk.drop(columns=[target_col])
    y_test_pipeline = next_chunk[target_col]

    # Save results (optional, can be adapted)
    results = {
        "type": "train_ipip",
        "timestamp": pd.Timestamp.now().isoformat(),
        "file": last_processed_file,
        "model_type": "IpipModel",
        "p": p,
        "b": b,
        "num_ensembles": len(final_model.ensembles_),
        "models_per_ensemble": [len(e) for e in final_model.ensembles_],
        "ensemble_validation_metrics": ensemble_validation_metrics
    }
    _save_results(Path(output_dir) / "train_results.json", results, logger)

    logger.info(f"[TRAIN] IPIP training completed in {time.time() - t0:.1f}s")
    return final_model, X_test_pipeline, y_test_pipeline, results


# =========================================================
# IPIP RETRAIN
# =========================================================
def default_retrain(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    last_processed_file: str,
    model_path: str, # Path to the existing IpipModel
    random_state: int,
    logger,
    output_dir: str,
    block_col: str,
    ipip_config: Optional[Dict[str, Any]] = None,
    model_instance, # Base model for new ensembles
    **kwargs,
):
    t0 = time.time()
    
    try:
        prev_model: IpipModel = joblib.load(model_path)
        logger.info("Previous IPIP model loaded successfully.")
    except Exception as e:
        raise ValueError(f"Could not load previous IpipModel from {model_path}: {e}")

    # --- Train a new model on the latest data (same logic as default_train) ---
    # This is a simplification. The R code seems to use the *current* chunk for training
    # and the *next* for evaluation. Here, we'll use the last block for training the new
    # ensembles and the second to last for testing them.
    
    blocks = _blocks_in_order(X, block_col)
    if len(blocks) < 2:
        logger.warning("Not enough blocks to perform retrain, returning previous model.")
        return prev_model, pd.DataFrame(), pd.Series(), {}

    new_ensembles_train_block = blocks[-1]
    new_ensembles_test_block = blocks[-2]

    logger.info(f"Retraining: new ensembles on block '{new_ensembles_train_block}', testing on '{new_ensembles_test_block}'")

    # We call default_train on a subset of the data to generate the new ensembles
    new_model, _, _, _ = default_train(
        X=X[X[block_col].isin([new_ensembles_test_block, new_ensembles_train_block])],
        y=y,
        last_processed_file=last_processed_file,
        model_instance=model_instance,
        random_state=random_state,
        logger=logger,
        output_dir=output_dir,
        block_col=block_col,
        ipip_config=ipip_config
    )

    # --- Combine and Prune ---
    p = ipip_config.get("p", 5)
    
    # Add new ensembles to the old ones
    combined_model = IpipModel(ensembles=prev_model.ensembles_ + new_model.ensembles_)
    logger.info(f"Combined model has {len(combined_model.ensembles_)} ensembles.")

    # Use the latest block as the hold-out set for pruning
    pruning_X = X[X[block_col] == new_ensembles_train_block].drop(columns=[block_col])
    pruning_y = y[X[block_col] == new_ensembles_train_block]

    # Find the best 'p' ensembles from the combined list
    best_indices = best_models(combined_model, "BA", pruning_X, pruning_y, p, logger)
    
    final_ensembles = [combined_model.ensembles_[i] for i in best_indices]
    final_model = IpipModel(ensembles=final_ensembles)

    logger.info(f"Pruned model to {len(final_model.ensembles_)} ensembles.")

    # The pipeline's test set is the very last block
    X_test_pipeline = X[X[block_col] == new_ensembles_train_block]
    y_test_pipeline = y[X[block_col] == new_ensembles_train_block]

    results = {
        "type": "retrain_ipip",
        "timestamp": pd.Timestamp.now().isoformat(),
        "file": last_processed_file,
        "model_type": "IpipModel",
        "p": p,
        "num_ensembles": len(final_model.ensembles_),
        "models_per_ensemble": [len(e) for e in final_model.ensembles_]
    }
    _save_results(Path(output_dir) / "retrain_results.json", results, logger)

    logger.info(f"[RETRAIN] IPIP retraining completed in {time.time() - t0:.1f}s")
    return final_model, X_test_pipeline, y_test_pipeline, results