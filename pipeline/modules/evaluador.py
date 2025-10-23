import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc
)

def _upsert_control_entry(control_file: Path, file_path: str, mtime, logger=None):
    """
    Atomically create or update a file entry in the control file with format '<basename>,<mtime>'.
    
    This function implements atomic control file management to prevent data corruption
    during concurrent operations. It maintains a registry of processed files with their
    modification timestamps to support incremental processing workflows.
    
    Args:
        control_file (Path): Path to the control file that tracks processed files
        file_path (str): Full path to the file being registered
        mtime: Modification time of the file (typically from os.path.getmtime)
        logger: Optional logger instance for operation tracking
    
    Note:
        - Uses only the filename with extension as the key (no directory paths)
        - If duplicate basenames exist from different directories, the last entry overwrites
        - Implements atomic write pattern using temporary file + os.replace() for safety
    """
    control_file.parent.mkdir(parents=True, exist_ok=True)

    # Key = only the filename with extension (no directories)
    key_name = Path(file_path).name

    # Load current entries
    existing = {}
    if control_file.exists():
        with open(control_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue  # skip malformed
                fname, raw_mtime = parts
                existing[fname] = raw_mtime

    # Upsert
    existing[key_name] = str(mtime)

    # Atomic rewrite
    tmp = control_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f"{k},{v}\n")
    os.replace(tmp, control_file)

    if logger:
        logger.info(f"Control file updated: {key_name} registered with modification time {mtime}")


def save_eval_results(results: dict, output_dir: str, logger=None):
    """
    Persist model evaluation results to JSON file for audit trail and monitoring.
    
    This function handles the serialization of evaluation metrics and ensures
    compatibility with JSON format by converting numpy types and complex objects.
    The results are stored in a standardized location for downstream analysis.
    
    Args:
        results (dict): Dictionary containing evaluation metrics, predictions, and metadata
        output_dir (str): Directory path where eval_results.json will be saved
        logger: Optional logger instance for operation tracking
    
    Note:
        - Automatically converts numpy types and arrays to JSON-compatible formats
        - Creates output directory if it doesn't exist
        - Results include timestamp, metrics, thresholds, and sample predictions
    """
    eval_path = os.path.join(output_dir, "eval_results.json")

    def make_serializable(obj):
        # Convert non-serializable objects into JSON-serializable types.
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, list): return [make_serializable(x) for x in obj]
        if isinstance(obj, dict): return {kk: make_serializable(vv) for kk, vv in obj.items()}
        return obj

    serializable_results = make_serializable(results)
    os.makedirs(output_dir, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(serializable_results, f, indent=4)

    if logger:
        logger.info(f"Evaluation results persisted to {eval_path}")


def calculate_metrics(model, X_test, y_test, is_classification: bool):
    """
    Calculate performance metrics based on model type and generate predictions.
    
    Automatically selects appropriate metrics based on problem type:
    - Classification: accuracy, balanced_accuracy, f1_score, classification_report
    - Regression: r2_score, rmse, mae, mse
    
    Args:
        model: Trained scikit-learn compatible model
        X_test: Test features for evaluation
        y_test: True target values for comparison
        is_classification (bool): Whether this is a classification or regression problem
    
    Returns:
        tuple: (metrics_dict, predictions_array, prediction_probabilities_array)
            - metrics: Dictionary containing computed performance metrics
            - y_pred: Model predictions on test set
            - y_pred_proba: Model prediction probabilities on test set
    """
    y_pred = model.predict(X_test)
    y_pred_proba = None

    if is_classification:
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro"),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        if y_pred_proba is not None and np.unique(y_test).size == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            metrics['roc_auc'] = roc_auc
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    else:
        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred)
        }

    return metrics, y_pred, y_pred_proba


def _save_candidate(model, results: dict, candidates_dir: str, base_name: str, logger=None):
    """
    Archive non-approved candidate models for future analysis and model archaeology.
    
    When a model fails to meet performance thresholds, it's preserved in the candidates
    directory rather than being discarded. This enables offline analysis, A/B testing
    preparation, and maintaining a complete audit trail of all training attempts.
    
    Directory structure created:
      candidates/<base_name>__YYYYmmdd_HHMMSS/
        - model.pkl: Serialized model object
        - eval_results.json: Complete evaluation metrics and predictions
        - meta.json: Compact metadata summary for quick filtering
    
    Args:
        model: The trained model object to be archived
        results (dict): Complete evaluation results including metrics and predictions
        candidates_dir (str): Root directory for candidate model storage
        base_name (str): Base name for the model (typically derived from model_filename)
        logger: Optional logger instance for operation tracking
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cand_root = os.path.join(candidates_dir, f"{base_name}__{ts}")
    os.makedirs(cand_root, exist_ok=True)

    # Save model
    model_path = os.path.join(cand_root, "model.pkl")
    joblib.dump(model, model_path)

    # Save eval results (as-is; you already persisted one globally)
    eval_path = os.path.join(cand_root, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save a minimal meta summary
    meta = {
        "approved": results.get("approved", False),
        "file": results.get("file"),
        "timestamp": results.get("timestamp"),
        "metrics_keys": list(results.get("metrics", {}).keys()) if isinstance(results.get("metrics"), dict) else []
    }
    with open(os.path.join(cand_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    if logger:
        logger.info(f"Candidate model archived at {cand_root} for future analysis")


def evaluator(
    *,
    model,
    X_test,
    y_test,
    last_processed_file,
    last_mtime,
    logger,
    is_first_model: bool,
    thresholds_perf: dict,
    model_dir: str,
    model_filename: str,
    control_dir: str,
    output_dir: str,
    df: pd.DataFrame,
    candidates_dir: str = None,
    save_candidates: bool = True,
    min_improvement_thresholds: dict = None,
):
    """
    Champion/Challenger model evaluation system with atomic model rotation.
    
    This function implements a robust model deployment strategy where new models
    (challengers) must meet or exceed performance thresholds to replace the current
    production model (champion). The evaluation process is atomic and includes
    comprehensive safety mechanisms.
    
    CHAMPION/CHALLENGER STRATEGY:
    - Current production model serves as the champion
    - New trained models are challengers that must prove superiority
    - Only models meeting all threshold criteria become the new champion
    - Failed challengers are optionally archived for analysis
    
    THRESHOLD VALIDATION LOGIC:
    - Automatically detects problem type (classification vs regression)
    - Validates threshold consistency to prevent configuration errors
    - Supports multiple metrics with min/max threshold semantics:
      * Classification: accuracy, balanced_accuracy, f1 (minimum thresholds)
      * Regression: r2 (minimum), rmse/mae/mse (maximum thresholds)
    
    ATOMIC MODEL ROTATION PROCESS:
    - Current champion is backed up with _previous.pkl suffix
    - New champion is saved only after successful backup
    - Control files are updated atomically to prevent corruption
    - Previous training data is preserved for debugging and rollback scenarios
    
    CANDIDATE MODEL STORAGE SYSTEM:
    - Non-approved models are archived with timestamps
    - Complete evaluation results and metadata are preserved
    - Enables offline analysis and future model archaeology
    
    CONTROL FILE MANAGEMENT:
    - Maintains registry of processed files with modification times
    - Prevents reprocessing unchanged data
    - Supports incremental training workflows
    - Uses atomic writes to prevent corruption during concurrent access
    
    Args:
        model: Trained model object to evaluate
        X_test: Test features for evaluation
        y_test: True target values
        last_processed_file: Path to the data file that generated this model
        last_mtime: Modification time of the processed file
        logger: Logger instance for operation tracking
        is_first_model (bool): Whether this is the initial model (affects validation)
        thresholds_perf (dict): Performance thresholds for model approval
        model_dir (str): Directory where champion models are stored
        model_filename: Filename for the champion model
        control_dir (str): Directory for control files and metadata
        output_dir (str): Directory for evaluation results
        df (pd.DataFrame): Training data to preserve with control files
        candidates_dir (str, optional): Directory for archiving failed candidates
        save_candidates (bool): Whether to save non-approved models
        min_improvement_thresholds (dict, optional): Minimum improvement thresholds over champion for each metric.
    
    Returns:
        bool: True if model was approved and deployed, False otherwise
    
    Raises:
        ValueError: If threshold configuration is inconsistent with model type
    """
    logger.info("Starting champion/challenger model evaluation")

    approved = True
    results = {}
    sample_predictions = []

    # Detect problem type heuristically and validate thresholds type coherence
    is_classification = hasattr(model, "predict_proba") or (len(set(y_test)) <= 20)
    classification_metrics = {"accuracy", "balanced_accuracy", "f1"}
    regression_metrics = {"rmse", "mae", "mse", "r2"}

    user_keys = set(thresholds_perf.keys())
    user_type = "classification" if (user_keys & classification_metrics) else \
                "regression" if (user_keys & regression_metrics) else None

    # Threshold validation: prevent configuration errors by ensuring metric consistency
    if user_type == "classification" and (user_keys & regression_metrics):
        raise ValueError("Configuration error: Cannot mix classification and regression metrics in thresholds_perf.")
    if user_type == "regression" and (user_keys & classification_metrics):
        raise ValueError("Configuration error: Cannot mix regression and classification metrics in thresholds_perf.")
    if is_classification and user_type == "regression":
        raise ValueError("Model type mismatch: Classification model detected but thresholds_perf contains regression metrics.")
    if (not is_classification) and user_type == "classification":
        raise ValueError("Model type mismatch: Regression model detected but thresholds_perf contains classification metrics.")

    # Load Champion model and evaluate its performance if not the first model
    champion_metrics = {}
    if not is_first_model:
        champion_model_path = os.path.join(model_dir, model_filename)
        if os.path.exists(champion_model_path):
            try:
                champion_model = joblib.load(champion_model_path)
                champion_metrics, _ = calculate_metrics(champion_model, X_test, y_test, is_classification)
                logger.info("Champion model loaded and evaluated for comparison.")
            except Exception as e:
                logger.warning(f"Could not load or evaluate Champion model for comparison: {e}")
        else:
            logger.info("No Champion model found for comparison (expected if first model, unexpected otherwise).")

    try:
        metrics, y_pred, y_pred_proba = calculate_metrics(model, X_test, y_test, is_classification)

        # Keep a small sample of predictions to aid debugging
        sample_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        })
        if y_pred_proba is not None:
            for i in range(y_pred_proba.shape[1]):
                sample_df[f"y_pred_proba_{i}"] = y_pred_proba[:, i]

        sample_predictions = sample_df.sample(n=min(10, len(y_test)), random_state=42).to_dict(orient="records")

        # Evaluate thresholds:
        for metric, threshold in thresholds_perf.items():
            value = metrics.get(metric)
            if value is None:
                logger.warning(f"Metric '{metric}' could not be calculated, skipping threshold check")
                continue

            # Regression thresholds (lower is better for error metrics)
            if metric in {"rmse", "mae", "mse"}:
                if value > threshold:
                    logger.warning(f"Threshold violation: {metric}={value:.6f} exceeds maximum allowed threshold of {threshold}")
                    approved = False
                else:
                    logger.info(f"Threshold passed: {metric}={value:.6f} <= {threshold}")
            elif metric == "r2":
                if value < threshold:
                    logger.warning(f"Threshold violation: {metric}={value:.6f} below minimum required threshold of {threshold}")
                    approved = False
                else:
                    logger.info(f"Threshold passed: {metric}={value:.6f} >= {threshold}")

            # Classification thresholds (higher is better)
            elif metric in {"accuracy", "balanced_accuracy", "f1"}:
                if value < threshold:
                    logger.warning(f"Threshold violation: {metric}={value:.6f} below minimum required threshold of {threshold}")
                    approved = False
                else:
                    logger.info(f"Threshold passed: {metric}={value:.6f} >= {threshold}")

            # Check for minimum improvement over Champion (if Champion metrics available)
            if not is_first_model and champion_metrics and metric in champion_metrics and min_improvement_thresholds:
                champion_value = champion_metrics.get(metric)
                min_improvement_ratio = min_improvement_thresholds.get(metric, 0.10) # Default to 10%

                if champion_value is not None and min_improvement_ratio is not None:
                    if metric in {"rmse", "mae", "mse"}: # Lower is better
                        required_value = champion_value * (1 - min_improvement_ratio)
                        if value > required_value:
                            logger.warning(f"Improvement check violation: {metric}={value:.6f} is not {min_improvement_ratio:.0%} better than Champion's {champion_value:.6f} (required <= {required_value:.6f})")
                            approved = False
                        else:
                            logger.info(f"Improvement check passed: {metric}={value:.6f} is {min_improvement_ratio:.0%} better than Champion's {champion_value:.6f}")
                    elif metric in {"accuracy", "balanced_accuracy", "f1", "r2"}: # Higher is better
                        required_value = champion_value * (1 + min_improvement_ratio)
                        if value < required_value:
                            logger.warning(f"Improvement check violation: {metric}={value:.6f} is not {min_improvement_ratio:.0%} better than Champion's {champion_value:.6f} (required >= {required_value:.6f})")
                            approved = False
                        else:
                            logger.info(f"Improvement check passed: {metric}={value:.6f} is {min_improvement_ratio:.0%} better than Champion's {champion_value:.6f}")

        # Assemble results payload
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": last_processed_file,
            "approved": approved,
            "metrics": metrics,
            "thresholds": thresholds_perf,
            "predictions": sample_predictions
        }

    except Exception as e:
        # Evaluation failed - reject model and log error details
        logger.error(f"Model evaluation failed due to unexpected error: {e}")
        approved = False
        # Create minimal results payload without exposing internal error details
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": last_processed_file,
            "approved": False
        }

    # Persist eval results (always)
    save_eval_results(results, output_dir, logger=logger)

    # Execute atomic model rotation if challenger is approved
    if approved:
        try:
            model_current_path = os.path.join(model_dir, model_filename)
            
            # Step 1: Backup existing champion model before replacement
            if os.path.exists(model_current_path):
                previous_model = model_current_path.replace(".pkl", "_previous.pkl")
                os.replace(model_current_path, previous_model)
                logger.info(f"Champion model backed up to {previous_model}")
            
            # Step 2: Promote challenger to champion status
            joblib.dump(model, model_current_path)
            logger.info("Model approved - challenger promoted to champion")

            # Step 3: Update control file registry atomically
            control_file = Path(control_dir) / "control_file.txt"
            _upsert_control_entry(control_file, last_processed_file, last_mtime, logger)

            # Step 4: Preserve training data for debugging and rollback scenarios
            df.to_csv(os.path.join(control_dir, "previous_data.csv"), index=False)
            logger.info("Model deployment completed - control files updated")

        except Exception as e:
            logger.error(f"Critical error during model deployment: {e}")
            # Model deployment failed but evaluation results are still persisted

    else:
        logger.info("Model rejected - performance thresholds not met, champion remains unchanged")
        # Archive challenger model for future analysis if configured
        if save_candidates and candidates_dir:
            base_name = os.path.splitext(os.path.basename(model_filename))[0]
            _save_candidate(model, results, candidates_dir, base_name, logger=logger)

    return approved
