# pipeline/pipeline_ipip.py
from __future__ import annotations

import os
import json
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Any
import re
import numpy as np


import joblib
import pandas as pd

from calmops.logger.logger import PipelineLogger
from calmops.utils import get_pipelines_root

# --- Robust imports of your modules ---
from .modules.data_loader import data_loader
from .modules.default_train_retrain import default_train, default_retrain


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")


# =========================================================
# Helpers
# =========================================================
# =========================================================
# Helpers
# =========================================================
def _upsert_control_entry(
    control_file: Path, file_name: str, mtime: float, logger: logging.Logger
) -> None:
    """
    Update or insert an entry in the control file.

    Args:
        control_file (Path): Path to the control file.
        file_name (str): Name of the file to update/insert.
        mtime (float): Modification time.
        logger (logging.Logger): Logger instance.
    """
    control_file.parent.mkdir(parents=True, exist_ok=True)
    key = Path(file_name).name

    existing = {}
    if control_file.exists():
        with open(control_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) == 2:
                    existing[parts[0]] = parts[1]

    existing[key] = str(mtime)
    tmp = control_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f"{k},{v}\n")
    tmp.replace(control_file)
    logger.info(
        f"[CONTROL] Upserted {key} with mtime={mtime} into {control_file.resolve()}"
    )


def _persist_model(
    *, model: Any, pipeline_name: str, output_dir: Path, logger: logging.Logger
) -> Path:
    """
    Save the model to disk.

    Args:
        model (Any): The model object to save.
        pipeline_name (str): Name of the pipeline.
        output_dir (Path): Directory to save the model.
        logger (logging.Logger): Logger instance.

    Returns:
        Path: Path to the saved model file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{pipeline_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"💾 Model saved at {model_path.resolve()} (overwritten)")
    return model_path


def _load_python(file_path: str, func_name: str) -> Any:
    """
    Dynamically load a function from a Python file.

    Args:
        file_path (str): Path to the Python file.
        func_name (str): Name of the function to load.

    Returns:
        Any: The loaded function.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the function is not found in the module.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, func_name):
        raise AttributeError(f"{file_path} must define {func_name}(...)")
    return getattr(mod, func_name)


def _log_model_evolution(
    new_model: Any, old_model: Any, logger: logging.Logger
) -> dict:
    """
    Log the evolution of the model (ensembles).

    Args:
        new_model (Any): The new model.
        old_model (Any): The old model.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: Information about the evolution.
    """
    evolution_info = {"changed": False, "details": []}
    try:
        if not hasattr(old_model, "ensembles_") or not hasattr(new_model, "ensembles_"):
            logger.info(
                "Cannot compare model evolution: one of the models is not an IPIP model."
            )
            return evolution_info

        old_ensembles = old_model.ensembles_
        new_ensembles = new_model.ensembles_

        logger.info("--- Model Evolution ---")
        if len(new_ensembles) != len(old_ensembles):
            msg = f"Number of ensembles changed from {len(old_ensembles)} to {len(new_ensembles)}."
            logger.info(msg)
            evolution_info["changed"] = True
            evolution_info["details"].append(msg)
        else:
            logger.info(f"Number of ensembles is unchanged: {len(new_ensembles)}.")

        for i in range(min(len(old_ensembles), len(new_ensembles))):
            if len(new_ensembles[i]) != len(old_ensembles[i]):
                msg = f"  Ensemble {i}: Number of base models changed from {len(old_ensembles[i])} to {len(new_ensembles[i])}."
                logger.info(msg)
                evolution_info["changed"] = True
                evolution_info["details"].append(msg)

        if len(new_ensembles) > len(old_ensembles):
            for i in range(len(old_ensembles), len(new_ensembles)):
                msg = f"  New ensemble {i} added with {len(new_ensembles[i])} base models."
                logger.info(msg)
                evolution_info["changed"] = True
                evolution_info["details"].append(msg)

        logger.info("--- End of Model Evolution ---")

    except Exception as e:
        logger.warning(f"Could not compare model evolution: {e}")

    return evolution_info


def _sorted_blocks(block_series: pd.Series) -> list[str]:
    """Sorts block identifiers naturally (e.g., 'block_1', 'block_2', 'block_10')."""

    def natural_sort_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("([0-9]+)", str(s))
        ]

    # Get unique blocks and sort them
    unique_blocks = block_series.unique()
    # Handle potential mixed types by converting to string
    sorted_unique_blocks = sorted(map(str, unique_blocks), key=natural_sort_key)
    return sorted_unique_blocks


# =========================================================
# Run Pipeline
# =========================================================
def run_pipeline(
    *,
    pipeline_name: str,
    data_dir: str,
    preprocess_file: str,
    model_instance,
    random_state: int,
    custom_train_file: str | None = None,
    custom_retrain_file: str | None = None,
    delimiter: str = ",",
    target_file: str | None = None,
    window_size: int | None = None,
    block_col: str | None = None,
    ipip_config: dict | None = None,
    dir_predictions: Optional[str] = None,
    prediction_only: bool = False,
    encoding: str = "utf-8",
    file_type: str = "csv",
) -> None:
    # Paths
    project_root = get_pipelines_root()
    base_dir = project_root / "pipelines" / pipeline_name
    output_dir = base_dir / "models"
    control_dir = base_dir / "control"
    logs_dir = base_dir / "logs"
    metrics_dir = base_dir / "metrics"
    for d in (output_dir, control_dir, logs_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{pipeline_name}.pkl"

    # Logger
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()

    logger.info("Pipeline (IPIP) started — GLOBAL mode over all blocks.")

    if not block_col:
        raise ValueError("You must provide block_col explicitly (e.g., 'chunk').")

    # 1) Load dataset
    df_full, last_processed_file, last_mtime = data_loader(
        logger,
        data_dir,
        control_dir,
        delimiter=delimiter,
        target_file=target_file,
        block_col=block_col,
        encoding=encoding,
        file_type=file_type,
    )
    if df_full.empty:
        logger.warning("No new data to process.")
        return
    # 2) Preprocess (the prepro chooses target and returns X,y)
    spec = importlib.util.spec_from_file_location("custom_preproc", preprocess_file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "data_preprocessing"):
        raise AttributeError(
            f"{preprocess_file} must define data_preprocessing(df)->(X,y)"
        )

    # Attempt to retrieve block_col from raw data
    original_blocks_series = None
    if block_col in df_full.columns:
        original_blocks_series = df_full[block_col].astype(str)

    X, y = mod.data_preprocessing(df_full)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    logger.info(f"Preprocessing OK: {X.shape[0]} rows, {X.shape[1]} columns.")

    # Handle block_col logic
    if original_blocks_series is not None:
        # Case A: block_col was in raw data -> restore/align it
        try:
            blocks_series = original_blocks_series.loc[X.index]
        except Exception:
            common = X.index.intersection(original_blocks_series.index)
            X = X.loc[common]
            y = y.loc[common]
            blocks_series = original_blocks_series.loc[common]
        X = X.copy()
        X[block_col] = blocks_series
    else:
        # Case B: block_col was NOT in raw data -> must be in X (created by prepro)
        if block_col not in X.columns:
            raise ValueError(
                f"block_col='{block_col}' not found in raw data AND not created by preprocessing."
            )
        # Ensure it's string/categorical
        X[block_col] = X[block_col].astype(str)

    # --- Start Sequential IPIP Logic (R-like) ---
    all_blocks = _sorted_blocks(X[block_col])
    if len(all_blocks) < 2:
        logger.warning(
            "Sequential IPIP evaluation requires at least 2 blocks to start training/evaluation."
        )
        return

    current_model = None
    final_predictions_list = []  # To store {y_true, y_pred, block}
    training_history = []  # To store p, b, and ensemble info per step

    # Placeholder for `model_instance` from pipeline config
    _model_instance = model_instance  # Use the `model_instance` passed to run_pipeline

    # 0. Initial Training (Block 0 & 1)
    logger.info("Sequential Run: Initial training on blocks 0 & 1.")
    initial_train_blocks = all_blocks[:2]  # First two blocks for default_train

    X_initial_train_data = X[X[block_col].isin(initial_train_blocks)]
    y_initial_train_data = y.loc[X_initial_train_data.index]

    if X_initial_train_data.empty:
        logger.error("Initial training data is empty. Cannot proceed.")
        return

    current_model, X_test_dummy, y_test_dummy, train_results = default_train(
        X=X_initial_train_data,
        y=y_initial_train_data,
        last_processed_file=last_processed_file,
        model_instance=_model_instance,
        random_state=random_state,
        logger=logger,
        output_dir=metrics_dir,
        block_col=block_col,
        ipip_config=ipip_config,
    )

    if train_results:
        training_history.append(
            {
                "block": str(initial_train_blocks[-1]),
                "p": train_results.get("p"),
                "b": train_results.get("b"),
                "num_ensembles": train_results.get("num_ensembles"),
                "models_per_ensemble": train_results.get("models_per_ensemble"),
                "type": "initial_train",
            }
        )

    # 1. Initial Evaluation (Evaluate model trained on block 0 and 1 on block 1)
    # This aligns with R's first evaluation step (model on t=1 evaluates t=2)
    eval_block_id_initial = all_blocks[1]
    X_initial_eval = X[X[block_col] == eval_block_id_initial]
    y_initial_eval = y.loc[X_initial_eval.index]

    if current_model and not X_initial_eval.empty and not y_initial_eval.empty:
        predictions = current_model.predict(
            X_initial_eval.drop(columns=[block_col], errors="ignore")
        )
        # Capture probabilities (assuming binary classification, take positive class)
        try:
            probas = current_model.predict_proba(
                X_initial_eval.drop(columns=[block_col], errors="ignore")
            )
            # If probas has 2 columns, take the second one (index 1) as positive class prob
            if probas.shape[1] == 2:
                y_pred_proba = probas[:, 1]
            else:
                # Fallback or handle multiclass if needed (for now assume binary)
                y_pred_proba = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
        except Exception:
            y_pred_proba = [None] * len(predictions)

        for yt, yp, ypp in zip(y_initial_eval, predictions, y_pred_proba):
            final_predictions_list.append(
                {
                    "y_true": yt,
                    "y_pred": yp,
                    "y_pred_proba": ypp,
                    "block": eval_block_id_initial,
                }
            )
        logger.info(f"Evaluated initial model on block '{eval_block_id_initial}'.")

    # 2. Sequential Retraining and Evaluation Loop (from block 2 onwards)
    for i in range(
        1, len(all_blocks) - 1
    ):  # Start from the second block (index 1, which is all_blocks[1])
        retrain_up_to_block_id = all_blocks[i]  # Current block for retraining
        eval_block_id = all_blocks[i + 1]  # Next block for evaluation

        # Retrain model using all data up to the current block (0 to i)
        logger.info(
            f"Sequential Run: Retraining with data up to block '{retrain_up_to_block_id}'."
        )
        retrain_blocks_mask = X[block_col].isin(all_blocks[: i + 1])
        X_retrain_data = X[retrain_blocks_mask]
        y_retrain_data = y.loc[X_retrain_data.index]

        if X_retrain_data.empty:
            logger.warning(
                f"Retrain data for block '{retrain_up_to_block_id}' is empty. Skipping."
            )
            continue

        # Prepare next block data for transductive pruning (Oracle behavior matching R script)
        X_eval_step = X[X[block_col] == eval_block_id]
        y_eval_step = y.loc[X_eval_step.index]

        # Pass the model object directly to default_retrain
        current_model, X_test_dummy, y_test_dummy, retrain_results = default_retrain(
            X=X_retrain_data,
            y=y_retrain_data,
            last_processed_file=last_processed_file,
            model_path=current_model,  # Pass model object directly
            random_state=random_state,
            logger=logger,
            output_dir=metrics_dir,
            block_col=block_col,
            ipip_config=ipip_config,
            model_instance=_model_instance,
            X_next=X_eval_step,  # Transductive pruning data
            y_next=y_eval_step,  # Transductive pruning data
        )

        if retrain_results:
            training_history.append(
                {
                    "block": str(retrain_up_to_block_id),
                    "p": retrain_results.get("p"),
                    "b": retrain_results.get(
                        "b"
                    ),  # default_retrain might not return b, need to check
                    "num_ensembles": retrain_results.get("num_ensembles"),
                    "models_per_ensemble": retrain_results.get("models_per_ensemble"),
                    "type": "retrain",
                }
            )

        # Evaluate the newly retrained model on the next block
        # (X_eval_step and y_eval_step are already defined above)

        if current_model and not X_eval_step.empty and not y_eval_step.empty:
            predictions = current_model.predict(
                X_eval_step.drop(columns=[block_col], errors="ignore")
            )
            # Capture probabilities
            try:
                probas = current_model.predict_proba(
                    X_eval_step.drop(columns=[block_col], errors="ignore")
                )
                if probas.shape[1] == 2:
                    y_pred_proba = probas[:, 1]
                else:
                    y_pred_proba = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
            except Exception:
                y_pred_proba = [None] * len(predictions)

            for yt, yp, ypp in zip(y_eval_step, predictions, y_pred_proba):
                final_predictions_list.append(
                    {
                        "y_true": yt,
                        "y_pred": yp,
                        "y_pred_proba": ypp,
                        "block": eval_block_id,
                    }
                )
            logger.info(
                f"Evaluated model (retrained on up to block '{retrain_up_to_block_id}') on block '{eval_block_id}'."
            )
        else:
            logger.warning(
                f"Could not evaluate on block '{eval_block_id}' (empty data or no model)."
            )

    # 3. Final Evaluation (Process remaining last block if not covered by loop)
    # The loop evaluates up to `all_blocks[-1]`. This means the model trained on `all_blocks[-2]`
    # is evaluated on `all_blocks[-1]`. The last model is never evaluated explicitly.
    # The R code only evaluates up to `max_chunk - 1`, meaning the last chunk is never used for evaluation.
    # We will replicate this by ending evaluation at `all_blocks[-1]`.

    # Generate final eval_results.json from collected sequential predictions
    if not final_predictions_list:
        logger.warning("No predictions collected for final evaluation.")
        return

    from datetime import datetime
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    # Helper to make results JSON-serializable
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

    # Process final_predictions_list
    full_predictions_df = pd.DataFrame(final_predictions_list)
    y_true_final = full_predictions_df["y_true"]
    y_pred_final = full_predictions_df["y_pred"]

    # Handle numeric encoding for AUC and saving
    # Try to infer if y_true is already numeric
    try:
        y_true_numeric = y_true_final.astype(float)
    except ValueError:
        # If string labels, encode them.
        # We need to know which is positive.
        # Usually "SI"/"YES"/1 is positive.
        # Let's use pd.factorize but we need consistency.
        # Or simpler: if we have y_pred_proba, we assume it corresponds to the class
        # that is "greater" in sorting order if sklearn defaults were used.
        # Let's just use LabelEncoder logic: sorted classes.
        unique_classes = sorted(y_true_final.unique())
        # Map to 0, 1, ...
        class_map = {c: i for i, c in enumerate(unique_classes)}
        y_true_numeric = y_true_final.map(class_map)

    full_predictions_df["y_true_numeric"] = y_true_numeric

    # Calculate Global AUC
    try:
        # Only if we have valid probabilities and at least 2 classes
        if (
            "y_pred_proba" in full_predictions_df.columns
            and not full_predictions_df["y_pred_proba"].isna().all()
            and len(y_true_final.unique()) > 1
        ):
            # Fill NaNs if any (shouldn't be if predict_proba worked)
            y_prob_clean = full_predictions_df["y_pred_proba"].fillna(0)
            roc_auc_global = float(roc_auc_score(y_true_numeric, y_prob_clean))
        else:
            roc_auc_global = None
    except Exception as e:
        logger.warning(f"Could not calculate global ROC AUC: {e}")
        roc_auc_global = None

    # Global metrics
    metrics_global = {
        "accuracy": float(accuracy_score(y_true_final, y_pred_final)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_final, y_pred_final)),
        "f1": float(f1_score(y_true_final, y_pred_final, average="macro")),
        "roc_auc": roc_auc_global,
        "classification_report": classification_report(
            y_true_final, y_pred_final, output_dict=True, zero_division=0
        ),
    }

    # Per-block metrics
    per_block_metrics_full = {}
    for block_id, group in full_predictions_df.groupby("block"):
        # Calculate per-block AUC
        try:
            if (
                "y_pred_proba" in group.columns
                and not group["y_pred_proba"].isna().all()
                and len(group["y_true"].unique()) > 1
            ):
                y_prob_block = group["y_pred_proba"].fillna(0)
                # Use the same mapping as global to ensure consistency?
                # Or re-map? Re-mapping might flip classes if one is missing.
                # Use global mapping.
                y_true_block_num = (
                    group["y_true"].map(class_map)
                    if "class_map" in locals()
                    else group["y_true"]
                )
                roc_auc_block = float(roc_auc_score(y_true_block_num, y_prob_block))
            else:
                roc_auc_block = None
        except Exception:
            roc_auc_block = None

        metrics_block = {
            "accuracy": float(accuracy_score(group["y_true"], group["y_pred"])),
            "balanced_accuracy": float(
                balanced_accuracy_score(group["y_true"], group["y_pred"])
            ),
            "f1": float(f1_score(group["y_true"], group["y_pred"], average="macro")),
            "roc_auc": roc_auc_block,
            "support": len(group),
        }
        per_block_metrics_full[block_id] = metrics_block

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": "classification",  # Assuming classification for now
        "approved": True,  # Placeholder, actual approval logic would need thresholds
        "metrics": _jsonable(metrics_global),
        "thresholds": {},  # No thresholds defined yet
        "blocks": {
            "block_col": block_col,
            "evaluated_blocks": _sorted_blocks(full_predictions_df["block"]),
            "per_block_metrics_full": _jsonable(per_block_metrics_full),
            "per_block_approved": {},  # No per-block approval logic yet
            "approved_blocks_count": 0,
            "rejected_blocks_count": 0,
            "top_worst_blocks": [],
        },
        "predictions": full_predictions_df.head(100).to_dict(
            orient="records"
        ),  # Sample predictions
        "model_info": {
            "type": type(current_model).__name__,
            "is_ipip": True,
        },
    }

    # Save current eval_results.json
    eval_path = metrics_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Sequential evaluation results saved to {eval_path}.")

    # Save to eval_history
    try:
        history_dir = metrics_dir / "eval_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_fname = f"eval_results_{ts}.json"
        history_path = history_dir / history_fname
        with open(history_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Historical evaluation results saved to {history_path}.")
    except Exception as e:
        logger.warning(f"Could not save historical evaluation results: {e}")

    # --- NEW: Save full predictions to CSV ---
    try:
        preds_path = metrics_dir / "predictions.csv"
        full_predictions_df.to_csv(preds_path, index=False)
        logger.info(f"Full predictions saved to {preds_path}.")
    except Exception as e:
        logger.warning(f"Could not save predictions.csv: {e}")

    # --- NEW: Save training history (p, b, ensembles) ---
    try:
        # We need to have collected this during the loop.
        # Since I am editing the end of the file, I need to make sure 'training_history' exists.
        # I will inject the collection logic in a separate edit or assume I'll do it now.
        # Actually, I should do it all in one go if possible, but the file is large.
        # Let's just save what we have if it exists, but I haven't created the list yet.
        # So I will split this into multiple edits. This edit adds the saving logic.

        if "training_history" in locals():
            history_json_path = metrics_dir / "training_history.json"
            with open(history_json_path, "w") as f:
                json.dump(training_history, f, indent=4)
            logger.info(f"Training history saved to {history_json_path}.")
    except Exception as e:
        logger.warning(f"Could not save training_history.json: {e}")

    # Update control file to mark this file as processed
    if last_processed_file and last_mtime:
        _upsert_control_entry(
            control_dir / "control_file.txt",
            last_processed_file,
            last_mtime,
            logger,
        )

    # Persist the final model
    _persist_model(
        model=current_model,
        pipeline_name=pipeline_name,
        output_dir=output_dir,
        logger=logger,
    )

    # Generate model_structure.json for dashboard
    if current_model and hasattr(current_model, "ensembles_"):
        model_struct_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_ensembles": len(current_model.ensembles_),
            "models_per_ensemble": [len(e) for e in current_model.ensembles_],
        }
        model_struct_path = metrics_dir / "model_structure.json"
        try:
            with open(model_struct_path, "w") as f:
                json.dump(model_struct_info, f, indent=4)
            logger.info(f"Model structure saved to {model_struct_path}")
        except Exception as e:
            logger.warning(f"Could not save model structure JSON: {e}")
