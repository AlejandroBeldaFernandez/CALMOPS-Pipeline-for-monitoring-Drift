import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
import sys
import logging
from typing import Any
from pathlib import Path
from datetime import datetime

# Add config path to import defaults
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.defaults import DRIFT_DETECTION


def save_drift_results(
    results: dict, output_dir: str, logger: logging.Logger = None
) -> None:
    """
    Serialize and save drift detection results to JSON format.

    Recursively converts numpy arrays and scalar types to native Python types
    to ensure proper JSON serialization. This function handles the complete
    drift analysis results including statistical tests, performance metrics,
    and decision outcomes.

    Args:
        results (dict): Complete drift analysis results containing test outcomes,
                       performance metrics, thresholds, and final decision
        output_dir (str): Directory path where drift_results.json will be saved
        logger (logging.Logger, optional): Logger instance.
    """
    drift_path = os.path.join(output_dir, "drift_results.json")

    def make_serializable(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray, list)):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {kk: make_serializable(vv) for kk, vv in obj.items()}
        return obj

    try:
        serializable_results = make_serializable(results)
        with open(drift_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
        if logger:
            logger.info(f"Drift analysis results saved to {drift_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save drift analysis results: {e}")


def check_drift(
    X: pd.DataFrame,
    y: pd.Series,
    detector: Any,
    logger: logging.Logger,
    perf_thresholds: dict,
    model_filename: str,
    data_dir: str,
    output_dir: str,
    control_dir: str,
    model_dir: str,
    current_filename: str,
    prediction_only: bool = False,
) -> str:
    """
    Comprehensive drift detection system with automatic model management.

    This function implements a multi-layered drift detection approach combining:
    1. Performance-based drift detection using user-defined thresholds
    2. Statistical drift detection using Frouros library implementations
    3. Comparative analysis between current and previous model versions
    4. Automatic rollback/promotion logic for operational stability

    Performance Detection:
        - Tests current model against absolute performance thresholds
        - Tests previous model (if available) against same thresholds
        - Implements majority voting: drift flagged if >=50% metrics fail

    Statistical Detection (Frouros Integration):
        - Univariate tests: Kolmogorov-Smirnov, PSI, Mann-Whitney,
          Cramér-von Mises, Hellinger Distance, Earth Mover's Distance
        - Uses alpha=0.05 for hypothesis tests, configurable thresholds for others
        - Majority voting across all statistical tests for final decision

    Rollback Logic:
        - Promotes previous model if it passes all performance checks
        - Promotes previous model if current model shows >=30% relative degradation
        - Uses atomic file swapping for safe model replacement

    Decision Outcomes:
        - 'train': No current model exists, initial training required
        - 'previous_promoted': Previous model restored due to better performance
        - 'retrain': Drift detected, model retraining recommended
        - 'no_drift': No significant drift detected, current model maintained
        - 'end_error': Critical error prevented drift analysis completion

    Args:
        X (pd.DataFrame): Feature matrix for drift analysis
        y (pd.Series): Target vector for performance evaluation
        detector (Any): Frouros detector instance with statistical test methods
        logger (logging.Logger): Logging instance for operational messages
        perf_thresholds (dict): Dictionary of performance metric thresholds
        model_filename (str): Name of current model file
        data_dir (str): Directory containing current dataset
        output_dir (str): Directory for saving drift analysis results
        control_dir (str): Directory containing reference/baseline data
        model_dir (str): Directory containing model files
        current_filename (str): The name of the current file being processed.
        prediction_only (bool): If True, skips drift detection and returns 'predict'.

    Returns:
        str: Decision code indicating required action
    """
    if prediction_only:
        logger.info("Prediction-only mode enabled. Skipping drift detection.")
        return "predict"
    model_path = os.path.join(model_dir, model_filename)
    prev_model_path = model_path.replace(".pkl", "_previous.pkl")

    drift_results = {
        "thresholds": perf_thresholds,
        "tests": {},
        "drift": {},
        "metrics": {},
    }

    # Initial training scenario: no current model exists
    if not os.path.exists(model_path):
        logger.info("No current model found, requesting initial training")
        drift_results["info"] = "First model deployment, drift analysis skipped"
        drift_results["decision"] = "train"
        drift_results["files_compared"] = {
            "reference_file": None,
            "current_file": os.path.join(data_dir, current_filename),
        }
        save_drift_results(drift_results, output_dir, logger)
        return "train"

    # Load current model for analysis
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load current model from {model_path}: {e}")
        drift_results["error"] = "failed_to_load_current_model"
        drift_results["decision"] = "end_error"
        save_drift_results(drift_results, output_dir, logger)
        return "end"

    # Performance evaluation: current model against thresholds
    task = "classification" if is_classifier(model) else "regression"

    # Prepare thresholds for the suite method
    threshold_params = {
        "balanced_accuracy_threshold": perf_thresholds.get("balanced_accuracy", 0.9),
        "accuracy_threshold": perf_thresholds.get("accuracy", 0.9),
        "f1_threshold": perf_thresholds.get("F1", 0.9),
        "rmse_threshold": perf_thresholds.get("rmse", 1.2),
        "r2_threshold": perf_thresholds.get("r2", 0.5),
        "mae_threshold": perf_thresholds.get("mae", 0.2),
        "mse_threshold": perf_thresholds.get("mse", 0.4),
    }

    current_perf_drift, perf_results_current, perf_flags_current = (
        detector.absolute_performance_degradation_suite(
            X, y, model, task=task, **threshold_params
        )
    )

    logger.debug(
        f"perf_results_current BEFORE manual correction: {perf_results_current}"
    )

    # Manual correction for drift flags, as the suite might have bugs
    logger.info("MANUALLY CORRECTING DRIFT FLAGS NOW...")
    higher_is_better = ["accuracy", "balanced_accuracy", "F1", "r2"]
    lower_is_better = ["rmse", "mae", "mse"]

    for metric, results in perf_results_current.items():
        threshold = results.get("threshold")
        # Robustly find the value key, which is any key other than 'threshold'
        value_key = next((k for k in results if k != "threshold"), None)
        value = results.get(value_key) if value_key else None

        if value is not None and threshold is not None:
            has_drift = False
            if metric in higher_is_better:
                has_drift = value < threshold
            elif metric in lower_is_better:
                has_drift = value > threshold

            logger.info(
                f"Correcting metric '{metric}': value={value}, threshold={threshold}, drift -> {has_drift}"
            )
            results["drift"] = has_drift
            perf_flags_current[metric] = has_drift

    logger.debug(
        f"perf_results_current AFTER manual correction: {perf_results_current}"
    )

    # Recalculate the overall drift flag based on the corrected individual flags
    if perf_flags_current:
        num_metrics = len(perf_flags_current)
        num_drifted = sum(1 for flag in perf_flags_current.values() if flag)
        # Majority voting: drift if >=50% of metrics fail
        current_perf_drift = (
            (num_drifted / num_metrics) >= 0.5 if num_metrics > 0 else False
        )

    drift_results["tests"]["Performance_Current"] = perf_results_current
    drift_results["drift"].update(
        {f"current::{k}": v for k, v in perf_flags_current.items()}
    )

    # Save drift check performance to eval_history for unified tracking
    try:
        timestamp = datetime.now()

        # Reformat perf_results_current to match a simpler metrics structure
        metrics_to_save = {}
        for metric, results in perf_results_current.items():
            value_key = next(
                (k for k in results if k != "threshold" and k != "drift"), None
            )
            if value_key:
                metrics_to_save[metric] = results.get(value_key)

        history_payload = {
            "timestamp": timestamp.isoformat(),
            "model_role": "Model Evaluated",  # This was the incumbent Champion being tested
            "metrics": metrics_to_save,
            "thresholds": perf_thresholds,
            "performance_drift": current_perf_drift,
            "drift_metrics": perf_flags_current,
        }

        # Ensure eval_history directory exists
        eval_history_dir = os.path.join(output_dir, "eval_history")
        os.makedirs(eval_history_dir, exist_ok=True)

        # Save to a new timestamped file
        history_filename = f"drift_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        history_filepath = os.path.join(eval_history_dir, history_filename)

        # Custom save logic to avoid using save_drift_results
        def make_serializable(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray, list)):
                return [make_serializable(x) for x in obj]
            if isinstance(obj, dict):
                return {kk: make_serializable(vv) for kk, vv in obj.items()}
            return obj

        serializable_payload = make_serializable(history_payload)
        with open(history_filepath, "w") as f:
            json.dump(serializable_payload, f, indent=4)
        logger.info(f"Drift performance check saved to {history_filepath}")

    except Exception as e:
        logger.warning(f"Could not save drift performance to eval_history: {e}")

    # Comparative analysis: previous model evaluation and rollback logic
    if os.path.exists(prev_model_path):
        try:
            prev_model = joblib.load(prev_model_path)

            # Performance evaluation: previous model against thresholds
            # Prepare thresholds for the suite method
            threshold_params_prev = {
                "balanced_accuracy_threshold": perf_thresholds.get(
                    "balanced_accuracy", 0.9
                ),
                "accuracy_threshold": perf_thresholds.get("accuracy", 0.9),
                "f1_threshold": perf_thresholds.get("1", 0.9),
                "rmse_threshold": perf_thresholds.get("rmse", 1.2),
                "r2_threshold": perf_thresholds.get("r2", 0.5),
                "mae_threshold": perf_thresholds.get("mae", 0.2),
                "mse_threshold": perf_thresholds.get("mse", 0.4),
            }

            prev_perf_drift, perf_results_prev, perf_flags_prev = (
                detector.absolute_performance_degradation_suite(
                    X, y, prev_model, task=task, **threshold_params_prev
                )
            )
            drift_results["tests"]["Performance_Previous"] = perf_results_prev
            drift_results["drift"].update(
                {f"previous::{k}": v for k, v in perf_flags_prev.items()}
            )
            # Strict evaluation: previous model must pass all threshold checks
            previous_passes = (
                not prev_perf_drift
            )  # If suite detects drift, it means it didn't pass

            # Relative performance comparison with configurable degradation threshold
            # This suite compares current vs previous model performance directly
            task = "classification" if is_classifier(model) else "regression"
            degradation_ratio = DRIFT_DETECTION["comparative_analysis"][
                "degradation_ratio"
            ]
            comp_drift, comp_results, comp_flags = (
                detector.performance_comparison_suite(
                    X,
                    y,
                    prev_model,
                    model,
                    task=task,
                    decay_ratio=degradation_ratio,
                    average="macro",
                )
            )
            drift_results["tests"]["Performance_Comparison"] = comp_results
            drift_results["drift"].update(
                {f"comparison::{k}": bool(v) for k, v in comp_flags.items()}
            )

            # Automatic rollback decision logic using two-tier safety check:
            # Tier 1: Previous model meets all performance thresholds (strict validation)
            # Tier 2: Current model shows >=30% relative performance degradation vs previous
            # Either condition triggers immediate model rollback for system stability
            if previous_passes or comp_drift:
                reason = (
                    "previous_passed_thresholds"
                    if previous_passes
                    else "current_degraded_vs_previous_30pct"
                )
                try:
                    # Atomic model file swapping for safe replacement
                    tmp_swap = model_path + ".swap"
                    os.replace(model_path, tmp_swap)
                    os.replace(prev_model_path, model_path)
                    os.replace(tmp_swap, prev_model_path)
                    logger.info(
                        f"Previous model promoted to current - Reason: {reason}"
                    )
                    drift_results["promoted_model"] = True
                    drift_results["promotion_reason"] = reason
                    drift_results["decision"] = "previous_promoted"
                    save_drift_results(drift_results, output_dir, logger)
                    return "end"  # Operational recovery completed, no retraining needed
                except Exception as e:
                    logger.error(f"Failed to swap previous/current models: {e}")
        except Exception as e:
            logger.error(f"Error during previous model evaluation: {e}")

    # Statistical drift detection using Frouros library implementations
    stats_drift_detected = False
    ref_file_path_for_json = None

    control_file = os.path.join(control_dir, "control_file.txt")
    if os.path.exists(control_file):
        with open(control_file, "r") as f:
            lines = f.readlines()

        if lines:
            # The last line is the most recent reference file
            last_ref_filename = lines[-1].strip().split(",")[0]
            ref_path = os.path.join(data_dir, last_ref_filename)
            ref_file_path_for_json = ref_path  # Save for later

            if os.path.exists(ref_path):
                try:
                    # Load reference dataset and prepare feature matrix
                    df_ref = pd.read_csv(ref_path)
                    X_ref = (
                        df_ref.drop(columns=[y.name])
                        if (y is not None and y.name in df_ref.columns)
                        else df_ref
                    )

                    # Univariate statistical tests for distribution drift detection using data_drift_suite
                    alpha = DRIFT_DETECTION["statistical_tests"]["alpha"]
                    psi_config = DRIFT_DETECTION["statistical_tests"]

                    stats_drift_detected, stats_results, stats_flags = (
                        detector.data_drift_suite(
                            X_ref,
                            X,
                            alpha=alpha,
                            psi_threshold=psi_config["psi_threshold"],
                            num_bins=psi_config["psi_num_bins"],
                        )
                    )

                    # Store detailed test results for analysis
                    drift_results["tests"].update(stats_results)
                    drift_results["drift"].update(stats_flags)

                except Exception as e:
                    logger.error(f"Statistical drift analysis failed: {e}")
            else:
                logger.warning(
                    f"Reference file '{last_ref_filename}' from control_file not found in data_dir."
                )
    else:
        logger.info("No control_file found, skipping statistical drift detection.")

    # Final decision: combine performance and statistical drift indicators
    # Uses logical OR: drift detected if either performance OR statistical tests indicate drift
    drift_detected = current_perf_drift or stats_drift_detected

    if drift_detected:
        logger.info("Drift detected - Model retraining recommended")
        drift_results["decision"] = "retrain"
    else:
        logger.info("No significant drift detected - Current model maintained")
        drift_results["decision"] = "no_drift"

    drift_results["files_compared"] = {
        "reference_file": ref_file_path_for_json,
        "current_file": os.path.join(data_dir, current_filename),
    }
    save_drift_results(drift_results, output_dir, logger)
    return "retrain" if drift_detected else "end"
