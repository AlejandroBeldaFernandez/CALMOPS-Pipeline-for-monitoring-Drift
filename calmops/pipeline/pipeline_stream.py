"""
Pipeline Stream Module

This module implements a comprehensive ML pipeline orchestrator with the following key features:
- Circuit breaker pattern to prevent repeated failures
- Champion/Challenger model promotion strategy
- Dynamic function loading for custom training/retraining logic
- Health tracking system for pipeline reliability
- Comprehensive error handling and logging

Author: CalmOps Team
"""

import os

from typing import Optional
import time
import json
import logging
import importlib.util
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from calmops.logger.logger import PipelineLogger
from calmops.utils import get_pipelines_root
from calmops.Detector.drift_detector import DriftDetector
from .modules.data_loader import data_loader
from .modules.check_drift import check_drift
from .modules.evaluator import evaluate_model
from .modules.default_train_retrain import default_train, default_retrain

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")
# ===============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# ===============================================================================
#
# The circuit breaker pattern prevents repeated attempts when the system is
# experiencing consecutive failures. It tracks failure counts and implements
# exponential backoff to give the system time to recover.
#
# Key Components:
# - Health state persistence across pipeline runs
# - Configurable failure thresholds and backoff periods
# - Automatic recovery after successful evaluations
# ===============================================================================


def _health_path(metrics_dir: Path) -> Path:
    """
    Generate the file path for circuit breaker health state persistence.

    Args:
        metrics_dir: Directory where metrics and health state are stored

    Returns:
        Full path to the health.json file
    """
    return metrics_dir / "health.json"


def _load_health(metrics_dir: Path) -> dict:
    """
    Load circuit breaker health state from persistent storage.

    The health state tracks:
    - consecutive_failures: Number of consecutive model evaluation failures
    - last_failure_ts: Timestamp of the most recent failure
    - paused_until: Timestamp until which retraining is paused (None if not paused)

    Args:
        metrics_dir: Directory containing the health.json file

    Returns:
        Dictionary containing health state with default values if file doesn't exist
    """
    path = _health_path(metrics_dir)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # If health file is corrupted, return default state
            return {
                "consecutive_failures": 0,
                "last_failure_ts": None,
                "paused_until": None,
            }

    # Return default health state for new pipelines
    return {"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None}


def _save_health(metrics_dir: Path, data: dict) -> None:
    """
    Persist circuit breaker health state to disk.

    Ensures the metrics directory exists and writes health state as JSON.
    This allows the circuit breaker to maintain state across pipeline runs.

    Args:
        metrics_dir: Directory where health state should be saved
        data: Health state dictionary to persist
    """
    path = _health_path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        # Log error but don't fail the pipeline
        logging.warning(f"Failed to save circuit breaker health state: {e}")


def _should_pause(health: dict) -> bool:
    """
    Determine if circuit breaker should prevent retraining attempts.

    Checks if the current time is still within the pause period established
    after consecutive failures exceeded the threshold.

    Args:
        health: Health state dictionary containing pause timestamp

    Returns:
        True if retraining should be paused, False otherwise
    """
    paused_until = health.get("paused_until")
    if paused_until is None:
        return False

    try:
        current_time = time.time()
        return current_time < float(paused_until)
    except (ValueError, TypeError):
        # If timestamp is invalid, don't pause
        return False


def _update_on_result(
    health: dict, approved: bool, backoff_minutes: int, max_failures: int
) -> dict:
    """
    Update circuit breaker state based on model evaluation result.

    Implements the core circuit breaker logic:
    - On success: Reset failure count and clear pause state
    - On failure: Increment failure count and potentially engage pause period

    The circuit breaker engages when consecutive failures reach the threshold,
    implementing exponential backoff to allow system recovery time.

    Args:
        health: Current health state dictionary
        approved: Whether the model evaluation was successful
        backoff_minutes: Minutes to pause retraining after max failures
        max_failures: Maximum consecutive failures before engaging circuit breaker

    Returns:
        Updated health state dictionary
    """
    current_time = time.time()

    if approved:
        # Success: Reset circuit breaker to healthy state
        health["consecutive_failures"] = 0
        health["last_failure_ts"] = None
        health["paused_until"] = None
    else:
        # Failure: Increment counter and potentially engage breaker
        health["consecutive_failures"] = int(health.get("consecutive_failures", 0)) + 1
        health["last_failure_ts"] = current_time

        # Engage circuit breaker if threshold exceeded
        if health["consecutive_failures"] >= max_failures:
            health["paused_until"] = current_time + (backoff_minutes * 60)

    return health


# ===============================================================================
# DYNAMIC FUNCTION LOADING SYSTEM
# ===============================================================================
#
# This system allows users to provide custom Python files containing specialized
# training, retraining, and fallback functions. The dynamic loader safely imports
# and validates these functions at runtime, enabling flexible pipeline customization.
#
# Key Features:
# - Safe module loading with proper error handling
# - Function existence validation before execution
# - Detailed error reporting for troubleshooting
# - Support for custom train/retrain/fallback strategies
# ===============================================================================


def load_custom_function(py_file: str, func_name: str):
    """
    Dynamically load and validate a custom function from a Python file.

    This function enables users to extend pipeline functionality by providing
    custom implementations for training, retraining, and fallback strategies.
    The loaded function is validated to ensure it exists before being returned.

    Args:
        py_file: Absolute path to the Python file containing the function
        func_name: Name of the function to load from the file

    Returns:
        The loaded function object, ready for execution

    Raises:
        ImportError: If file loading fails or function doesn't exist
        AttributeError: If the specified function is not found in the module

    Example:
        >>> train_func = load_custom_function("/path/to/custom.py", "train")
        >>> model = train_func(X, y, ...)
    """
    if not Path(py_file).exists():
        raise ImportError(f"Custom function file does not exist: {py_file}")

    try:
        # Create module specification from file path
        spec = importlib.util.spec_from_file_location("custom_module", py_file)
        if spec is None:
            raise ImportError(f"Could not create module specification for: {py_file}")

        # Create and execute the module
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        # Validate function exists
        if not hasattr(custom_module, func_name):
            available_functions = [
                attr
                for attr in dir(custom_module)
                if callable(getattr(custom_module, attr)) and not attr.startswith("_")
            ]
            raise AttributeError(
                f"Function '{func_name}' not found in {py_file}. "
                f"Available functions: {available_functions}"
            )

        return getattr(custom_module, func_name)

    except Exception as e:
        # Provide detailed error context for debugging
        raise ImportError(
            f"Failed to load function '{func_name}' from '{py_file}': {type(e).__name__}: {e}"
        )


# ===============================================================================
# ML PIPELINE ORCHESTRATOR
# ===============================================================================
#
# This is the main pipeline orchestration function that implements a comprehensive
# ML operations workflow with the following key patterns:
#
# 1. CHAMPION/CHALLENGER PATTERN:
#    - Current production model serves as the "Champion"
#    - New models are "Challengers" that must prove themselves
#    - Only approved Challengers are promoted to Champion status
#    - Non-approved models are saved as candidates for analysis
#
# 2. CIRCUIT BREAKER RELIABILITY:
#    - Prevents repeated failures from degrading system performance
#    - Implements configurable backoff periods after consecutive failures
#    - Automatically recovers when models start performing well again
#
# 3. FLEXIBLE TRAINING STRATEGIES:
#    - Supports custom training, retraining, and fallback functions
#    - Default implementations available for standard workflows
#    - Dynamic loading allows for pipeline customization without code changes
#
# 4. COMPREHENSIVE ERROR HANDLING:
#    - Graceful degradation on failures
#    - Detailed logging for troubleshooting and monitoring
#    - State persistence for recovery across runs
# ===============================================================================


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
    prediction_only: bool = False,
    # Circuit breaker configuration
    breaker_max_failures: int = 3,
    breaker_backoff_minutes: int = 120,
    dir_predictions: Optional[str] = None,
    encoding: str = "utf-8",
    file_type: str = "csv",
):
    """
    Execute the complete ML pipeline with champion/challenger promotion and circuit breaker protection.

    This function orchestrates the entire machine learning pipeline lifecycle including:
    - Data loading and preprocessing
    - Drift detection and decision making
    - Model training, retraining, and evaluation
    - Champion/Challenger promotion logic
    - Circuit breaker failure protection
    - Comprehensive logging and monitoring

    Champion/Challenger Strategy:
        The pipeline maintains a production "Champion" model and evaluates new "Challenger"
        models against performance thresholds. Only Challengers that meet approval criteria
        are promoted to Champion status, ensuring production stability.

    Circuit Breaker Protection:
        After consecutive evaluation failures, the circuit breaker pauses retraining
        attempts for a configurable period, preventing system degradation and allowing
        time for issue resolution.

    Args:
        pipeline_name: Unique identifier for this pipeline instance
        data_dir: Directory containing input data files
        preprocess_file: Path to Python file containing data_preprocessing function
        thresholds_drift: Drift detection thresholds for triggering retraining
        thresholds_perf: Performance thresholds for model approval
        model_instance: Base model instance for training (scikit-learn compatible)
        retrain_mode: Strategy for retraining (see default_retrain documentation)
        fallback_mode: Fallback strategy if primary retraining fails
        random_state: Random seed for reproducibility
        param_grid: Hyperparameter grid for model tuning (optional)
        cv: Cross-validation folds for hyperparameter tuning (optional)
        custom_train_file: Path to custom training function (optional)
        custom_retrain_file: Path to custom retraining function (optional)
        custom_fallback_file: Path to custom fallback function (optional)
        delimiter: CSV delimiter for data files (default: ",")
        target_file: Specific target file to process (optional)
        window_size: Rolling window size for certain retraining modes (optional)
        breaker_max_failures: Circuit breaker failure threshold (default: 3)
        breaker_backoff_minutes: Circuit breaker pause duration in minutes (default: 120)
        dir_predictions: Directory to save predictions (optional)
        encoding: File encoding (default: "utf-8")
        file_type: Explicit file type (default: "csv")

    Raises:
        FileNotFoundError: If required files (preprocessing, custom functions) don't exist
        ValueError: If conflicting thresholds are specified
        ImportError: If custom functions cannot be loaded

    Returns:
        None: Function performs side effects (model saving, logging, metrics)
    """
    # ============================================================================
    # PIPELINE INITIALIZATION AND SETUP
    # ============================================================================

    # Create directory structure for pipeline artifacts
    project_root = get_pipelines_root()
    base_dir = project_root / "pipelines" / pipeline_name
    output_dir = base_dir / "models"  # Champion model storage
    control_dir = base_dir / "control"  # Processing state tracking
    logs_dir = base_dir / "logs"  # Pipeline execution logs
    metrics_dir = base_dir / "metrics"  # Performance metrics and health
    candidates_dir = base_dir / "candidates"  # Non-approved model storage

    # Ensure all required directories exist
    for directory in [output_dir, control_dir, logs_dir, metrics_dir, candidates_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{pipeline_name}.pkl"

    # Initialize logging system
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()
    logging.basicConfig()
    logger.info(f"Pipeline '{pipeline_name}' execution started")

    # ============================================================================
    # CIRCUIT BREAKER PROTECTION CHECK
    # ============================================================================

    # Load circuit breaker health state and check if retraining should be paused
    health = _load_health(metrics_dir)
    if _should_pause(health):
        failures = health.get("consecutive_failures", 0)
        pause_until_ts = health.get("paused_until", 0)

        logger.warning(
            f"Circuit breaker engaged: Retraining paused after {failures} consecutive failures. "
            f"Will resume after {time.ctime(pause_until_ts)} (backoff: {breaker_backoff_minutes}min)"
        )
        return

    # ============================================================================
    # CONFIGURATION VALIDATION
    # ============================================================================

    # Prevent mixing classification and regression metrics in the same pipeline
    classification_metrics = {
        "accuracy",
        "F1",
        "balanced_accuracy",
        "precision",
        "recall",
    }
    regression_metrics = {"rmse", "mae", "mse", "r2"}

    has_classification = bool(classification_metrics & set(thresholds_perf.keys()))
    has_regression = bool(regression_metrics & set(thresholds_perf.keys()))

    if has_classification and has_regression:
        raise ValueError(
            "Configuration error: Cannot mix classification metrics "
            f"({classification_metrics & set(thresholds_perf.keys())}) "
            f"with regression metrics ({regression_metrics & set(thresholds_perf.keys())}) "
            "in the same pipeline"
        )

    # ============================================================================
    # CUSTOM FUNCTION LOADING
    # ============================================================================

    # Load and validate the required preprocessing function
    if not Path(preprocess_file).exists():
        raise FileNotFoundError(f"Preprocessing file not found: {preprocess_file}")

    try:
        preprocess_func = load_custom_function(preprocess_file, "data_preprocessing")
        logger.info(f"Loaded preprocessing function from: {preprocess_file}")
    except ImportError as e:
        logger.error(f"Failed to load preprocessing function: {e}")
        raise

    # ============================================================================
    # DATA LOADING AND PREPROCESSING
    # ============================================================================

    # Initialize drift detection system
    detector = DriftDetector()
    logger.info("Drift detector initialized")

    # Load new data using the data loader module
    logger.info(f"Loading data from directory: {data_dir}")
    df, last_processed_file, last_mtime = data_loader(
        logger,
        data_dir,
        control_dir,
        delimiter=delimiter,
        target_file=target_file,
        encoding=encoding,
        file_type=file_type,
    )

    # Check if new data is available for processing
    if df.empty:
        logger.info(
            "No new data available for processing - pipeline execution complete"
        )
        return

    # Apply preprocessing to prepare data for model operations
    try:
        X, y = preprocess_func(df)
        logger.info(
            f"Data preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features. "
            f"Source file: {last_processed_file}"
        )
        df = pd.concat([X, y], axis=1)
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

    # ============================================================================
    # DRIFT DETECTION AND DECISION MAKING
    # ============================================================================

    # Analyze data drift to determine if retraining is necessary
    logger.info("Performing drift analysis to determine pipeline action")
    decision = check_drift(
        X,
        y,
        detector,
        logger,
        thresholds_drift,
        f"{pipeline_name}.pkl",
        data_dir,
        metrics_dir,
        control_dir,
        model_dir=output_dir,
        current_filename=last_processed_file,
        prediction_only=prediction_only,
    )
    logger.info(f"Drift analysis decision: {decision.upper()}")

    # ============================================================================
    # PIPELINE EXECUTION: TRAINING, RETRAINING, OR MONITORING
    # ============================================================================

    try:
        if decision == "train":
            # ========================================================================
            # INITIAL TRAINING PHASE
            # ========================================================================
            # This phase occurs when no existing model is found or when starting fresh.
            # The trained model becomes the initial "Champion" if it passes evaluation.

            logger.info("Initiating TRAIN phase - creating initial model")

            if custom_train_file:
                logger.info(f"Using custom training function from: {custom_train_file}")
                train_func = load_custom_function(custom_train_file, "train")
                model, X_test, y_test, _ = train_func(
                    X, y, last_processed_file, logger, metrics_dir
                )
            else:
                logger.info("Using default training implementation")
                model, X_test, y_test, _ = default_train(
                    X,
                    y,
                    last_processed_file,
                    model_instance=model_instance,
                    random_state=random_state,
                    logger=logger,
                    param_grid=param_grid,
                    cv=cv,
                    output_dir=metrics_dir,
                )

            # Champion/Challenger Evaluation for Initial Training
            # Note: For initial training, there is no fallback - the model either
            # becomes the first Champion or is saved as a candidate for analysis
            logger.info("Evaluating trained model for Champion promotion")
            approved = evaluate_model(
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
                save_candidates=True,
                dir_predictions=dir_predictions,
            )

            if approved:
                logger.info("Model approved - promoted to Champion status")
            else:
                logger.warning(
                    "Model did not meet approval thresholds - saved as candidate"
                )

            # Update circuit breaker health state
            health = _update_on_result(
                health, approved, breaker_backoff_minutes, breaker_max_failures
            )
            _save_health(metrics_dir, health)

        elif decision == "retrain":
            # ========================================================================
            # CHALLENGER RETRAINING PHASE
            # ========================================================================
            # This phase creates a new "Challenger" model to potentially replace
            # the current "Champion". The Challenger must prove superior performance
            # to be promoted. If it fails, fallback strategies may be attempted.

            logger.info("Initiating RETRAIN phase - creating Challenger model")

            if custom_retrain_file:
                logger.info(
                    f"Using custom retraining function from: {custom_retrain_file}"
                )
                retrain_func = load_custom_function(custom_retrain_file, "retrain")
                model, X_test, y_test, _ = retrain_func(
                    X, y, last_processed_file, logger, metrics_dir
                )
            else:
                logger.info(
                    f"Using default retraining implementation (mode: {retrain_mode})"
                )
                model, X_test, y_test, _ = default_retrain(
                    X,
                    y,
                    last_processed_file,
                    model_path,
                    mode=retrain_mode,
                    random_state=random_state,
                    logger=logger,
                    param_grid=param_grid,
                    cv=cv,
                    output_dir=metrics_dir,
                    window_size=window_size,
                )

            # Champion/Challenger Evaluation for Retraining
            # The new model competes against the current Champion for promotion
            logger.info("Evaluating Challenger model against Champion thresholds")
            approved = evaluate_model(
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
                save_candidates=True,
                dir_predictions=dir_predictions,
            )

            # Fallback Strategy Implementation
            # If the primary Challenger fails evaluation and fallback is configured,
            # attempt alternative retraining approach before giving up
            if not approved and fallback_mode is not None:
                logger.info(
                    f"Challenger model failed evaluation - attempting fallback strategy (mode: {fallback_mode})"
                )

                if custom_fallback_file:
                    logger.info(
                        f"Using custom fallback function from: {custom_fallback_file}"
                    )
                    fallback_func = load_custom_function(
                        custom_fallback_file, "fallback"
                    )
                    model, X_test, y_test, _ = fallback_func(
                        X, y, last_processed_file, logger, metrics_dir
                    )
                else:
                    logger.info("Using default fallback implementation")
                    model, X_test, y_test, _ = default_retrain(
                        X,
                        y,
                        last_processed_file,
                        model_path,
                        mode=fallback_mode,
                        random_state=random_state,
                        logger=logger,
                        param_grid=param_grid,
                        cv=cv,
                        output_dir=metrics_dir,
                        window_size=window_size,
                    )

                # Evaluate fallback Challenger model
                logger.info("Evaluating fallback Challenger model")
                approved = evaluate_model(
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
                    save_candidates=True,
                    dir_predictions=dir_predictions,
                )

                if approved:
                    logger.info(
                        "Fallback Challenger approved - promoted to Champion status"
                    )
                else:
                    logger.warning(
                        "Both primary and fallback Challengers failed - Champion remains unchanged"
                    )
            elif approved:
                logger.info("Challenger approved - promoted to Champion status")
            else:
                logger.warning(
                    "Challenger failed evaluation - Champion remains unchanged"
                )

            # Update circuit breaker health state after retraining attempt
            health = _update_on_result(
                health, approved, breaker_backoff_minutes, breaker_max_failures
            )
            _save_health(metrics_dir, health)

        elif decision == "predict":
            logger.info("Prediction-only mode. Loading model and predicting.")
            if model_path.exists():
                model = joblib.load(model_path)
                predictions = model.predict(X)
                # I need to decide where to save the predictions.
                # The user just said "save the results".
                # I'll save them to a file in the metrics_dir.
                predictions_df = pd.DataFrame(predictions, columns=["prediction"])
                predictions_path = (
                    metrics_dir / f"predictions_{last_processed_file}.csv"
                )
                predictions_df.to_csv(predictions_path, index=False)
                logger.info(f"Predictions saved to {predictions_path}")

                if dir_predictions:
                    try:
                        os.makedirs(dir_predictions, exist_ok=True)
                        extra_pred_path = (
                            Path(dir_predictions)
                            / f"predictions_{Path(last_processed_file).stem}.csv"
                        )
                        predictions_df.to_csv(extra_pred_path, index=False)
                        logger.info(f"Predictions also saved to {extra_pred_path}")
                    except Exception as e:
                        logger.warning(
                            f"Could not save predictions to dir_predictions ({dir_predictions}): {e}"
                        )
            else:
                logger.error("No model found to make predictions.")

        else:
            # ========================================================================
            # CHAMPION MONITORING PHASE
            # ========================================================================
            # This phase occurs when no drift is detected but we still want to
            # monitor the current Champion's performance on new data. This helps
            # ensure the Champion continues to perform well in production.

            logger.info(
                "No retraining required - performing Champion health monitoring"
            )

            if model_path.exists():
                # Load current Champion model for health evaluation
                logger.info("Loading current Champion model for performance monitoring")
                model = joblib.load(model_path)

                # Create test split for Champion evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=random_state
                )

                # Evaluate Champion performance on new data
                # Note: This is monitoring only - Champion is not saved as a candidate
                logger.info("Evaluating Champion performance on new data")
                approved = evaluate_model(
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
                    candidates_dir=None,  # Champion not saved as candidate
                    save_candidates=False,  # Disable candidate saving for monitoring
                    dir_predictions=dir_predictions,
                )

                if approved:
                    logger.info(
                        "Champion model continues to perform within acceptable thresholds"
                    )
                else:
                    logger.warning(
                        "Champion model performance degraded - consider manual investigation"
                    )

                # Update circuit breaker state
                # Note: Champion monitoring results may indicate systemic issues
                # that warrant circuit breaker consideration
                health = _update_on_result(
                    health, approved, breaker_backoff_minutes, breaker_max_failures
                )
                _save_health(metrics_dir, health)
            else:
                logger.warning(
                    "No Champion model found - pipeline may need initialization"
                )

    except Exception as e:
        # ========================================================================
        # COMPREHENSIVE ERROR HANDLING
        # ========================================================================
        # Critical errors in the pipeline are logged with full context and
        # re-raised to ensure proper error propagation to calling systems.
        # This ensures pipeline failures are visible and actionable.

        logger.error(
            f"Critical pipeline failure in '{pipeline_name}': {type(e).__name__}: {e}. "
            f"Data file: {last_processed_file if 'last_processed_file' in locals() else 'unknown'}"
        )

        # Log additional context for debugging if available
        if "decision" in locals():
            logger.error(f"Pipeline failed during {decision.upper()} phase")

        # Re-raise to ensure calling code can handle the error appropriately
        raise

    logger.info(f"Pipeline '{pipeline_name}' execution completed successfully")
