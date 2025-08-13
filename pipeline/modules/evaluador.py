import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


def save_eval_results(results: dict, output_dir: str, logger=None):
    """
    Persist evaluation results into metrics/eval_results.json.

    Notes:
    - Uses logger instead of print (if provided).
    - Results are assumed to be JSON-serializable; non-serializable types are converted.
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
        logger.info(f"[EVAL] Results saved at {eval_path}")


def calculate_metrics(model, X_test, y_test, is_classification: bool):
    """
    Compute metrics for classification or regression models.
    Returns:
        metrics: dict with computed metrics
        y_pred: model predictions (array-like)
    """
    y_pred = model.predict(X_test)

    if is_classification:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro"),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
    else:
        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred)
        }

    return metrics, y_pred


def _save_candidate(model, results: dict, candidates_dir: str, base_name: str, logger=None):
    """
    Persist a non-approved model and its evaluation results for offline analysis.

    Creates a timestamped folder:
      candidates/<base_name>__YYYYmmdd_HHMMSS/
        - model.pkl
        - eval_results.json
        - meta.json (with short summary)
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
        logger.info(f"[CANDIDATE] Saved non-approved model at {cand_root}")


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
    # new:
    candidates_dir: str = None,    # where to store non-approved candidates
    save_candidates: bool = True,  # toggle saving candidates
):
    """
    Evaluate a trained/retrained model against user thresholds and optionally persist it.

    Behavior:
    - If approved:
        - Save current model to model_dir/model_filename
        - Rotate previous model (suffix _previous.pkl)
        - Append control_file.txt with file+mtime
        - Save previous_data.csv
    - If not approved:
        - Optionally save candidate under candidates/
        - Do NOT touch current champion or control files
    - Always writes metrics to metrics/eval_results.json (without storing error text)
    """
    logger.info(">>> Starting model evaluation")

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

    # Guardrails: avoid mixing threshold types and mismatching with model type
    if user_type == "classification" and (user_keys & regression_metrics):
        raise ValueError("❌ Do not mix classification and regression metrics in thresholds_perf.")
    if user_type == "regression" and (user_keys & classification_metrics):
        raise ValueError("❌ Do not mix regression and classification metrics in thresholds_perf.")
    if is_classification and user_type == "regression":
        raise ValueError("⚠️ Classification model but thresholds_perf contains regression metrics.")
    if (not is_classification) and user_type == "classification":
        raise ValueError("⚠️ Regression model but thresholds_perf contains classification metrics.")

    try:
        metrics, y_pred = calculate_metrics(model, X_test, y_test, is_classification)

        # Keep a small sample of predictions to aid debugging
        sample_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        }).sample(n=min(10, len(y_test)), random_state=42)
        sample_predictions = sample_df.to_dict(orient="records")

        # Evaluate thresholds:
        for metric, threshold in thresholds_perf.items():
            value = metrics.get(metric)
            if value is None:
                logger.warning(f"⚠️ Metric '{metric}' not calculated.")
                continue

            # Regression thresholds
            if metric in {"rmse", "mae", "mse"}:
                if value > threshold:
                    logger.warning(f"❌ {metric}: {value:.6f} > {threshold} (max allowed)")
                    approved = False
            elif metric == "r2":
                if value < threshold:
                    logger.warning(f"❌ r2: {value:.6f} < {threshold} (min required)")
                    approved = False

            # Classification thresholds
            elif metric in {"accuracy", "balanced_accuracy", "f1"}:
                if value < threshold:
                    logger.warning(f"❌ {metric}: {value:.6f} < {threshold} (min required)")
                    approved = False

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
        # Do NOT store error text in JSON (per your request)
        logger.error(f"Error during evaluation: {e}")
        approved = False
        # Minimal payload without error text
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": last_processed_file,
            "approved": False
        }

    # Persist eval results (always)
    save_eval_results(results, output_dir, logger=logger)

    # If approved, rotate and save model + update control
    if approved:
        try:
            model_current_path = os.path.join(model_dir, model_filename)
            if os.path.exists(model_current_path):
                previous_model = model_current_path.replace(".pkl", "_previous.pkl")
                os.replace(model_current_path, previous_model)
                logger.info(f"Previous model backed up at {previous_model}")

            # Save the newly approved model (champion promotion)
            joblib.dump(model, model_current_path)
            logger.info("✅ Model approved and saved successfully")

            # Update control file and persist the last dataset used for training
            control_file = Path(control_dir) / "control_file.txt"
            with open(control_file, "a") as f:
                f.write(f"{last_processed_file},{last_mtime}\n")

            df.to_csv(os.path.join(control_dir, "previous_data.csv"), index=False)
            logger.info(f"Control file updated at {control_file}")

        except Exception as e:
            logger.error(f"Error saving model/metrics: {e}")
            # Note: as requested, no error text is written into eval_results.json here.

    else:
        logger.warning("❌ Model did not pass thresholds. Model not updated.")
        if save_candidates and candidates_dir:
            base_name = os.path.splitext(os.path.basename(model_filename))[0]
            _save_candidate(model, results, candidates_dir, base_name, logger=logger)

    return approved
