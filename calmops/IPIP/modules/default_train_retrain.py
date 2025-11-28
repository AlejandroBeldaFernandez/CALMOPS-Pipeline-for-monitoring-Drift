# IPIP/modules/default_train_retrain.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import math
import logging
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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.ndarray, list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    return o


def _save_results(path: Path, payload: Dict[str, Any], logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=4, ensure_ascii=False)
    logger.info(f"[TRAIN] Results saved at {path.resolve()}")


def _blocks_in_order(X: pd.DataFrame, block_col: str) -> List[str]:
    return [str(x) for x in pd.unique(X[block_col]).tolist()]


def _get_metric(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
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
    metric_max: str,  # e.g. "BA"
    x: pd.DataFrame,
    y: pd.Series,
    p: int,
    logger: logging.Logger,
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
            model_performance.append(-1.0)  # Penalize failing ensembles

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
    model_instance: Any,
    random_state: int,
    logger: logging.Logger,
    output_dir: str,
    block_col: str,
    ipip_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[IpipModel, pd.DataFrame, pd.Series, dict]:
    if model_instance is None:
        raise ValueError("[TRAIN] 'model_instance' is mandatory for initial training.")
    if ipip_config is None:
        raise ValueError("[TRAIN] 'ipip_config' is mandatory for IPIP training.")

    t0 = time.time()

    # --- Config from ipip_config ---
    # --- Config from ipip_config ---
    # p and b are now calculated dynamically, ignoring config values if present
    # 'prop_majoritaria' matches the user's config and R script convention
    majority_prop = ipip_config.get(
        "prop_majoritaria", ipip_config.get("majority_prop", 0.55)
    )
    max_attempts = ipip_config.get("max_attempts", 5)
    val_size = ipip_config.get("val_size", 0.2)
    prop_minor_frac = ipip_config.get("prop_minor_frac", 0.75)
    target_col = y.name

    logger.info(f"Starting IPIP train with config: majority_prop={majority_prop}")

    # The R code trains on the first chunk and evaluates on the second.
    # Our pipeline gives us all data, so we'll simulate this.
    blocks = _blocks_in_order(X, block_col)

    # MODIFIED: Allow single block training (needed for faithful retraining)
    if len(blocks) < 1:
        raise ValueError("IPIP training requires at least 1 block.")

    train_block = blocks[0]
    # If we have a second block, we can define it, but it's not strictly used for ensemble building
    test_block = blocks[1] if len(blocks) > 1 else None

    current_chunk = pd.concat([X[X[block_col] == train_block], y], axis=1)
    # next_chunk is unused in the ensemble building logic, so we can ignore it if test_block is None

    # Split current chunk into train/test for building the ensembles
    train_df, test_df = train_test_split(
        current_chunk,
        test_size=val_size,
        random_state=random_state,
        stratify=current_chunk[target_col],
    )

    X_test_ensemble = test_df.drop(columns=[target_col, block_col])
    y_test_ensemble = test_df[target_col]

    # --- Start of the main logic from R file ---

    discharge = train_df[train_df[target_col] == 0]  # Assuming 'NO' class is 0
    expired = train_df[train_df[target_col] == 1]  # Assuming 'YES' class is 1

    if len(expired) == 0 or len(discharge) == 0:
        raise ValueError("Training data must contain samples from both classes.")

    np_val = round(len(expired) * prop_minor_frac)
    if np_val == 0:
        raise ValueError(
            "Not enough samples in the minority class to create training sets."
        )

    # Dynamic calculation of p and b (faithful to R script)
    # p = ceiling(log(.01)/(log(1-1/nrow(expired))*np))
    # b = ceiling(log(.01)/(log(1-1/np)*np))

    n_expired = len(expired)

    # Calculate p
    if n_expired > 1:
        denom_p = math.log(1 - 1 / n_expired) * np_val
        if denom_p != 0:
            p = math.ceil(math.log(0.01) / denom_p)
        else:
            p = 5
            logger.warning("Denominator for p calculation is 0. Using default p=5.")
    else:
        p = 5
        logger.warning(
            "Not enough expired samples for p calculation. Using default p=5."
        )

    # Calculate b
    if np_val > 1:
        denom_b = math.log(1 - 1 / np_val) * np_val
        if denom_b != 0:
            b = math.ceil(math.log(0.01) / denom_b)
        else:
            b = 10
            logger.warning("Denominator for b calculation is 0. Using default b=10.")
    else:
        b = 10
        logger.warning("Not enough np_val for b calculation. Using default b=10.")

    logger.info(
        f"Dynamic parameters calculated: p={p}, b={b} (n_expired={n_expired}, np={np_val})"
    )

    # Create 'p' balanced datasets
    dfs = []
    for _ in range(p):
        id_ex = expired.sample(n=np_val, replace=True, random_state=random_state)
        n_dis = round(np_val * majority_prop / (1 - majority_prop))
        id_dis = discharge.sample(n=n_dis, replace=True, random_state=random_state)
        dfs.append(pd.concat([id_dis, id_ex]))

    E = []  # This will be the list of ensembles (our IpipModel)

    for k in range(p):
        logger.info(f"Building ensemble {k + 1}/{p}...")
        Ek = []  # The k-th ensemble
        i = 0  # Attempts counter
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

            train_sample = pd.concat(
                [
                    g0.sample(n=n_min, replace=True, random_state=random_state + i),
                    g1.sample(n=n_min, replace=True, random_state=random_state + i),
                ]
            )
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
                i = 0  # Reset attempts
                logger.debug(
                    f"  Ensemble {k + 1} improved to BA={metric_new:.4f} with {len(Ek)} models."
                )
            else:
                i += 1  # Increment attempts

        E.append(Ek)
        logger.info(f"Finished building ensemble {k + 1} with {len(Ek)} models.")

    final_model = IpipModel(ensembles=E)

    # Calculate internal validation metrics for each final ensemble
    ensemble_validation_metrics = []
    for k, ensemble in enumerate(final_model.ensembles_):
        try:
            proba = _predict_ensemble_proba(ensemble, X_test_ensemble)
            metric = _get_metric(y_test_ensemble, proba)
            ensemble_validation_metrics.append(metric)
        except Exception as e:
            logger.warning(
                f"Could not calculate validation metric for ensemble {k + 1}: {e}"
            )
            ensemble_validation_metrics.append(None)

    # The "test" set for the pipeline is the whole dataset
    X_test_pipeline = X
    y_test_pipeline = y

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
        "ensemble_validation_metrics": ensemble_validation_metrics,
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
    model_path: str
    | IpipModel,  # Path to the existing IpipModel OR the IpipModel object itself
    random_state: int,
    logger: logging.Logger,
    output_dir: str,
    block_col: str,
    ipip_config: Optional[Dict[str, Any]] = None,
    model_instance: Any,  # Base model for new ensembles
    **kwargs,
) -> Tuple[IpipModel, pd.DataFrame, pd.Series, dict]:
    t0 = time.time()

    prev_model: IpipModel
    if isinstance(model_path, IpipModel):
        prev_model = model_path
        logger.info("Previous IPIP model object passed directly.")
    else:
        try:
            prev_model = joblib.load(model_path)
            logger.info("Previous IPIP model loaded successfully from path.")
        except Exception as e:
            raise ValueError(
                f"Could not load previous IpipModel from {model_path}: {e}"
            )

    # --- Train a new model on the latest data (same logic as default_train) ---
    # This is a simplification. The R code seems to use the *current* chunk for training
    # and the *next* for evaluation. Here, we'll use the last block for training the new
    # ensembles and the second to last for testing them.

    blocks = _blocks_in_order(X, block_col)
    if len(blocks) < 1:
        logger.warning(
            "Not enough blocks to perform retrain, returning previous model."
        )
        return prev_model, pd.DataFrame(), pd.Series(), {}

    new_ensembles_train_block = blocks[-1]
    # new_ensembles_test_block = blocks[-2] # No longer needed for training

    logger.info(f"Retraining: new ensembles on block '{new_ensembles_train_block}'")

    # We call default_train on a subset of the data to generate the new ensembles
    # MODIFIED: Pass ONLY the current block to ensure we train on t, not t-1
    new_model, _, _, _ = default_train(
        X=X[X[block_col].isin([new_ensembles_train_block])],
        y=y,
        last_processed_file=last_processed_file,
        model_instance=model_instance,
        random_state=random_state,
        logger=logger,
        output_dir=output_dir,
        block_col=block_col,
        ipip_config=ipip_config,
    )

    # --- Combine and Prune ---
    # We need to recalculate p for pruning based on the NEW training block's minority class size
    # The R code uses 'p' calculated from the current chunk (which is the training chunk for the next step)

    # Get minority count from the training block
    train_block_data = X[X[block_col] == new_ensembles_train_block]
    train_block_y = y[X[block_col] == new_ensembles_train_block]

    # Assuming binary classification and we need to find the minority class count
    # We can infer the minority class from the full dataset or just count here.
    # Let's assume the same class distribution logic as default_train
    # We need to know which is 'expired' (minority).
    # In default_train we assumed 1 is expired. Let's stick to that or check counts.

    # Heuristic: assume class 1 is minority or check counts
    counts = train_block_y.value_counts()
    if len(counts) < 2:
        # Fallback if only one class present
        p = 5
        logger.warning(
            f"Only one class in block {new_ensembles_train_block}. Using default p=5 for pruning."
        )
    else:
        # Assuming the smaller class is the minority one we care about for the formula
        n_expired_retrain = counts.min()
        np_val_retrain = round(
            n_expired_retrain * 0.75
        )  # Using default prop_minor_frac 0.75

        if n_expired_retrain > 1 and np_val_retrain > 0:
            denom_p = math.log(1 - 1 / n_expired_retrain) * np_val_retrain
            if denom_p != 0:
                p = math.ceil(math.log(0.01) / denom_p)
            else:
                p = 5
        else:
            p = 5

    logger.info(f"Dynamic p calculated for pruning: p={p}")

    # Add new ensembles to the old ones
    combined_model = IpipModel(ensembles=prev_model.ensembles_ + new_model.ensembles_)
    logger.info(f"Combined model has {len(combined_model.ensembles_)} ensembles.")

    # Pruning Step
    # Check if X_next/y_next are provided (Transductive Pruning)
    X_next = kwargs.get("X_next")
    y_next = kwargs.get("y_next")

    if X_next is not None and y_next is not None:
        pruning_X = X_next.drop(columns=[block_col], errors="ignore")
        pruning_y = y_next
    else:
        logger.info("Using CURRENT block for pruning (Standard behavior).")
        # Use the latest block as the hold-out set for pruning
        pruning_X = X[X[block_col] == new_ensembles_train_block].drop(
            columns=[block_col]
        )
        pruning_y = y[X[block_col] == new_ensembles_train_block]

    # Find the best 'p' ensembles from the combined list
    best_indices = best_models(combined_model, "BA", pruning_X, pruning_y, p, logger)

    final_ensembles = [combined_model.ensembles_[i] for i in best_indices]
    final_model = IpipModel(ensembles=final_ensembles)

    logger.info(f"Pruned model to {len(final_model.ensembles_)} ensembles.")

    # The pipeline's test set is the full dataset
    X_test_pipeline = X
    y_test_pipeline = y

    results = {
        "type": "retrain_ipip",
        "timestamp": pd.Timestamp.now().isoformat(),
        "file": last_processed_file,
        "model_type": "IpipModel",
        "p": p,
        "num_ensembles": len(final_model.ensembles_),
        "models_per_ensemble": [len(e) for e in final_model.ensembles_],
    }
    _save_results(Path(output_dir) / "retrain_results.json", results, logger)

    logger.info(f"[RETRAIN] IPIP retraining completed in {time.time() - t0:.1f}s")
    return final_model, X_test_pipeline, y_test_pipeline, results
