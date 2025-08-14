# pipeline/modules/evaluador.py
# -*- coding: utf-8 -*-
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
    """Persist evaluation results into <output_dir>/eval_results.json."""
    eval_path = os.path.join(output_dir, "eval_results.json")

    def make_serializable(obj):
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
    Compute global metrics for classification or regression models.
    Returns (metrics, y_pred)
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

def _metrics_from_preds(y_true, y_pred, is_classification: bool) -> dict:
    """Compute metrics using precomputed predictions (used for per-block summaries)."""
    if is_classification:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro"),
        }
    else:
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }

def _save_candidate(model, results: dict, candidates_dir: str, base_name: str, logger=None):
    """
    Persist a non-approved model and its evaluation results for offline analysis.
    candidates/<base_name>__YYYYmmdd_HHMMSS/{ model.pkl, eval_results.json, meta.json }
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cand_root = os.path.join(candidates_dir, f"{base_name}__{ts}")
    os.makedirs(cand_root, exist_ok=True)

    joblib.dump(model, os.path.join(cand_root, "model.pkl"))
    with open(os.path.join(cand_root, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    meta = {
        "approved": results.get("approved", False),
        "file": results.get("file"),
        "timestamp": results.get("timestamp"),
        "metrics_keys": list(results.get("metrics", {}).keys()) if isinstance(results.get("metrics"), dict) else [],
        "evaluated_block_id": results.get("blocks", {}).get("evaluated_block_id"),
        "reference_blocks": results.get("blocks", {}).get("reference_blocks"),
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
    # Block-aware extras:
    block_col: str = None,
    evaluated_block_id: str = None,
    test_blocks: pd.Series = None,
    reference_df: pd.DataFrame = None,
    reference_blocks: list = None,
    # Candidates handling
    candidates_dir: str = None,
    save_candidates: bool = True,
):
    """
    Evaluate model vs thresholds; if approved, promote to champion and persist reference window.
    Computes optional per-block metrics on test if `test_blocks` provided.
    """
    logger.info(">>> Starting model evaluation")

    approved = True
    results = {}
    sample_predictions = []

    is_classification = hasattr(model, "predict_proba") or (len(set(y_test)) <= 20)
    classification_metrics = {"accuracy", "balanced_accuracy", "f1"}
    regression_metrics = {"rmse", "mae", "mse", "r2"}

    user_keys = set(thresholds_perf.keys())
    user_type = "classification" if (user_keys & classification_metrics) else \
                "regression" if (user_keys & regression_metrics) else None

    # Guardrails
    if user_type == "classification" and (user_keys & regression_metrics):
        raise ValueError("Do not mix classification and regression metrics in thresholds_perf.")
    if user_type == "regression" and (user_keys & classification_metrics):
        raise ValueError("Do not mix regression and classification metrics in thresholds_perf.")
    if is_classification and user_type == "regression":
        raise ValueError("Classification model but thresholds_perf contains regression metrics.")
    if (not is_classification) and user_type == "classification":
        raise ValueError("Regression model but thresholds_perf contains classification metrics.")

    try:
        metrics, y_pred = calculate_metrics(model, X_test, y_test, is_classification)

        sample_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).sample(
            n=min(10, len(y_test)), random_state=42
        )
        sample_predictions = sample_df.to_dict(orient="records")

        for metric, threshold in thresholds_perf.items():
            value = metrics.get(metric)
            if value is None:
                logger.warning(f"Metric '{metric}' not calculated.")
                continue

            if metric in {"rmse", "mae", "mse"}:
                if value > threshold:
                    logger.warning(f"{metric}: {value:.6f} > {threshold} (max allowed)")
                    approved = False
            elif metric == "r2":
                if value < threshold:
                    logger.warning(f"r2: {value:.6f} < {threshold} (min required)")
                    approved = False
            elif metric in {"accuracy", "balanced_accuracy", "f1"}:
                if value < threshold:
                    logger.warning(f"{metric}: {value:.6f} < {threshold} (min required)")
                    approved = False

        per_block_metrics = {}
        if test_blocks is not None:
            try:
                if len(test_blocks) != len(y_test):
                    logger.warning("test_blocks length mismatch with y_test; skipping per-block metrics.")
                else:
                    tb = pd.Series(test_blocks, index=pd.RangeIndex(len(test_blocks)))
                    y_true_s = pd.Series(y_test, index=pd.RangeIndex(len(y_test)))
                    y_pred_s = pd.Series(y_pred, index=pd.RangeIndex(len(y_pred)))

                    for bid, idxs in tb.groupby(tb).groups.items():
                        y_t = y_true_s.loc[idxs]
                        y_p = y_pred_s.loc[idxs]
                        per_block_metrics[str(bid)] = _metrics_from_preds(y_t, y_p, is_classification)
            except Exception as e:
                logger.warning(f"Per-block metrics computation failed: {e}")

        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": last_processed_file,
            "approved": approved,
            "metrics": metrics,
            "thresholds": thresholds_perf,
            "predictions": sample_predictions,
            "blocks": {
                "block_col": block_col,
                "evaluated_block_id": evaluated_block_id,
                "per_block_metrics": per_block_metrics if per_block_metrics else None,
                "reference_blocks": reference_blocks if reference_blocks else None,
            },
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        approved = False
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": last_processed_file,
            "approved": False,
            "blocks": {
                "block_col": block_col,
                "evaluated_block_id": evaluated_block_id,
                "reference_blocks": reference_blocks if reference_blocks else None,
            },
        }

    # Persist results
    save_eval_results(results, output_dir, logger=logger)

    if approved:
        try:
            model_current_path = os.path.join(model_dir, model_filename)
            if os.path.exists(model_current_path):
                previous_model = model_current_path.replace(".pkl", "_previous.pkl")
                os.replace(model_current_path, previous_model)
                logger.info(f"Previous model backed up at {previous_model}")

            # Save champion
            joblib.dump(model, model_current_path)
            logger.info("✅ Model approved and saved successfully")

            # Append control file
            control_file = Path(control_dir) / "control_file.txt"
            control_file.parent.mkdir(parents=True, exist_ok=True)
            with open(control_file, "a") as f:
                f.write(f"{last_processed_file},{last_mtime}\n")

            # Persist reference window for drift stats
            to_persist = reference_df if reference_df is not None else df
            prev_path = Path(control_dir) / "previous_data.csv"
            to_persist.to_csv(prev_path, index=False)
            logger.info(f"Reference data saved to {prev_path} ({len(to_persist)} rows)")

        except Exception as e:
            logger.error(f"Error saving model/control/reference: {e}")

    else:
        logger.warning("Model did not pass thresholds. Champion model not updated.")
        if save_candidates and candidates_dir:
            base_name = os.path.splitext(os.path.basename(model_filename))[0]
            _save_candidate(model, results, candidates_dir, base_name, logger=logger)

    return approved
