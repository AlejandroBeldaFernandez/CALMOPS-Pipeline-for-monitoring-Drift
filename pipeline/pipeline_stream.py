import os
import time
import json
import logging
import importlib.util
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from logger.logger import PipelineLogger
from Detector.drift_detector import DriftDetector
from .modules.data_loader import data_loader
from .modules.check_drift import check_drift
from .modules.evaluador import evaluator
from .modules.default_train_retrain import default_train, default_retrain


# -------------------------------
# Circuit Breaker Utilities
# -------------------------------

def _health_path(metrics_dir: str) -> str:
    return os.path.join(metrics_dir, "health.json")

def _load_health(metrics_dir: str):
    """Load health state for circuit breaker."""
    path = _health_path(metrics_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None}

def _save_health(metrics_dir: str, data: dict):
    """Persist health state."""
    path = _health_path(metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _should_pause(health: dict) -> bool:
    """Return True if breaker is currently pausing retraining attempts."""
    paused_until = health.get("paused_until")
    if paused_until is None:
        return False
    try:
        return time.time() < float(paused_until)
    except Exception:
        return False

def _update_on_result(health: dict, approved: bool, backoff_minutes: int, max_failures: int):
    """
    Update breaker state after an evaluation result.

    If consecutive failures reach the threshold, engage a pause/backoff period.
    """
    if approved:
        health["consecutive_failures"] = 0
        health["last_failure_ts"] = None
        health["paused_until"] = None
    else:
        health["consecutive_failures"] = int(health.get("consecutive_failures", 0)) + 1
        health["last_failure_ts"] = time.time()
        if health["consecutive_failures"] >= max_failures:
            health["paused_until"] = time.time() + backoff_minutes * 60
    return health


# -------------------------------
# Dynamic function loader
# -------------------------------

def load_custom_function(py_file: str, func_name: str):
    """
    Load a function named `func_name` from python file `py_file`.
    """
    try:
        spec = importlib.util.spec_from_file_location("custom_module", py_file)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        if not hasattr(custom_module, func_name):
            raise AttributeError(f"The file {py_file} does not contain the function {func_name}")
        return getattr(custom_module, func_name)
    except Exception as e:
        raise ImportError(f"Error loading function '{func_name}' from file '{py_file}': {e}")


# -------------------------------
# Main pipeline
# -------------------------------

def run_pipeline(
    *,
    pipeline_name: str,
    data_dir: str,
    preprocess_file: str,
    thresholds_drift: dict,
    thresholds_perf: dict,
    model_instance,
    retrain_mode: int,
    fallback_mode: int,
    random_state: int,
    param_grid: dict = None,
    cv: int = None,
    custom_train_file: str = None,
    custom_retrain_file: str = None,
    custom_fallback_file: str = None,
    delimiter: str = ",",
    target_file: str = None,
    window_size: int = None,
    # breaker config:
    breaker_max_failures: int = 3,
    breaker_backoff_minutes: int = 120,
):
    """
    Orchestrates training/retraining and evaluation with:
      - Champion/Challenger policy (only promote on approval)
      - Circuit breaker to pause after repeated failures
      - Candidate saving for non-approved models
    """
    # Directories
    base_dir = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    output_dir = os.path.join(base_dir, "modelos")
    control_dir = os.path.join(base_dir, "control")
    logs_dir = os.path.join(base_dir, "logs")
    metrics_dir = os.path.join(base_dir, "metrics")
    candidates_dir = os.path.join(base_dir, "candidates")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(control_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{pipeline_name}.pkl")

    # Logger
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()
    logging.basicConfig()
    logger.info("Pipeline started")

    # Circuit breaker check
    health = _load_health(metrics_dir)
    if _should_pause(health):
        logger.warning("⚠️ Retraining paused by circuit breaker. Skipping this run.")
        return

    # Validate thresholds (avoid mixing classification/regression)
    if {"accuracy", "f1", "balanced_accuracy"} & set(thresholds_perf.keys()) and \
       {"rmse", "mae", "mse", "r2"} & set(thresholds_perf.keys()):
        raise ValueError("Cannot define both classification and regression thresholds at the same time.")

    # Preprocess function
    if not os.path.exists(preprocess_file):
        raise FileNotFoundError(f"Please provide a valid preprocessing file: {preprocess_file}")
    preprocess_func = load_custom_function(preprocess_file, "data_preprocessing")

    # Drift detector
    detector = DriftDetector()

    # Load and preprocess data
    df, last_processed_file, last_mtime = data_loader(
        logger, data_dir, control_dir, delimiter=delimiter, target_file=target_file
    )

    if df.empty:
        logger.warning("No new data to process.")
        return

    X, y = preprocess_func(df)
    logger.info(f"Data preprocessed: {X.shape[0]} rows, {X.shape[1]} columns")
    df = pd.concat([X, y], axis=1)

    # Drift decision
    decision = check_drift(
        X, y, detector, logger, thresholds_drift,
        f"{pipeline_name}.pkl", data_dir, metrics_dir, control_dir, model_dir=output_dir
    )

    # TRAIN / RETRAIN / NO-ACTION
    try:
        if decision == "train":
            logger.info("Starting TRAIN phase")
            if custom_train_file:
                train_func = load_custom_function(custom_train_file, "train")
                model, X_test, y_test, _ = train_func(X, y, last_processed_file, logger, metrics_dir)
            else:
                model, X_test, y_test, _ = default_train(
                    X, y, last_processed_file,
                    model_instance=model_instance,
                    random_state=random_state,
                    logger=logger,
                    param_grid=param_grid,
                    cv=cv,
                    output_dir=metrics_dir
                )

            # Evaluate (no fallback after TRAIN by request)
            approved = evaluator(
                model=model,
                X_test=X_test,
                y_test=y_test,
                last_processed_file=last_processed_file,
                last_mtime=last_mtime,
                logger=logger,
                is_first_model=True,
                thresholds_perf=thresholds_perf,
                model_dir=output_dir,
                model_filename=f"{pipeline_name}.pkl",
                control_dir=control_dir,
                df=df,
                output_dir=metrics_dir,
                candidates_dir=candidates_dir,
                save_candidates=True
            )

            # Update breaker
            health = _update_on_result(health, approved, breaker_backoff_minutes, breaker_max_failures)
            _save_health(metrics_dir, health)

        elif decision == "retrain":
            logger.info("Starting RETRAIN phase")
            if custom_retrain_file:
                retrain_func = load_custom_function(custom_retrain_file, "retrain")
                model, X_test, y_test, _ = retrain_func(X, y, last_processed_file, logger, metrics_dir)
            else:
                model, X_test, y_test, _ = default_retrain(
                    X, y, last_processed_file,
                    model_path,
                    mode=retrain_mode,
                    random_state=random_state,
                    logger=logger,
                    param_grid=param_grid,
                    cv=cv,
                    output_dir=metrics_dir,
                    window_size=window_size
                )

            approved = evaluator(
                model=model,
                X_test=X_test,
                y_test=y_test,
                last_processed_file=last_processed_file,
                last_mtime=last_mtime,
                logger=logger,
                is_first_model=False,
                thresholds_perf=thresholds_perf,
                model_dir=output_dir,
                model_filename=f"{pipeline_name}.pkl",
                control_dir=control_dir,
                df=df,
                output_dir=metrics_dir,
                candidates_dir=candidates_dir,
                save_candidates=True
            )

            # If not approved, try fallback (RETRAIN only)
            if not approved and fallback_mode is not None:
                logger.info(f"Attempting fallback with mode {fallback_mode}...")
                if custom_fallback_file:
                    fallback_func = load_custom_function(custom_fallback_file, "fallback")
                    model, X_test, y_test, _ = fallback_func(X, y, last_processed_file, logger, metrics_dir)
                else:
                    model, X_test, y_test, _ = default_retrain(
                        X, y, last_processed_file,
                        model_path,
                        mode=fallback_mode,
                        random_state=random_state,
                        logger=logger,
                        param_grid=param_grid,
                        cv=cv,
                        output_dir=metrics_dir,
                        window_size=window_size
                    )

                approved = evaluator(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    last_processed_file=last_processed_file,
                    last_mtime=last_mtime,
                    logger=logger,
                    is_first_model=False,
                    thresholds_perf=thresholds_perf,
                    model_dir=output_dir,
                    model_filename=f"{pipeline_name}.pkl",
                    control_dir=control_dir,
                    df=df,
                    output_dir=metrics_dir,
                    candidates_dir=candidates_dir,
                    save_candidates=True
                )

            # Update breaker after retrain (and possible fallback)
            health = _update_on_result(health, approved, breaker_backoff_minutes, breaker_max_failures)
            _save_health(metrics_dir, health)

        else:
            logger.info("Model already trained, no retraining needed.")
            # Optional: periodic re-eval of current champion
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=random_state
                )
                approved = evaluator(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    last_processed_file=last_processed_file,
                    last_mtime=last_mtime,
                    logger=logger,
                    is_first_model=False,
                    thresholds_perf=thresholds_perf,
                    model_dir=output_dir,
                    model_filename=f"{pipeline_name}.pkl",
                    control_dir=control_dir,
                    df=df,
                    output_dir=metrics_dir,
                    candidates_dir=None,        # do not save champion as candidate
                    save_candidates=False
                )
                # Re-eval shouldn't typically affect breaker, but we keep parity:
                health = _update_on_result(health, approved, breaker_backoff_minutes, breaker_max_failures)
                _save_health(metrics_dir, health)

    except Exception as e:
        logger.error(f"[CRITICAL] Critical error in run_pipeline: {e}")
        raise
