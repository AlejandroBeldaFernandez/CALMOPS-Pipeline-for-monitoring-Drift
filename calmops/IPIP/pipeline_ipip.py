# pipeline/pipeline_ipip.py
from __future__ import annotations

import os
import json
import time
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Any, List
import re
import numpy as np


import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from calmops.logger.logger import PipelineLogger
from calmops.utils import get_pipelines_root
from calmops.utils.HistoryManager import HistoryManager

# --- Robust imports of your modules ---
from .modules.data_loader import data_loader
from .modules.default_train_retrain import default_train, default_retrain


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf


# tf.compat.v1.logging.set_verbosity(# tf.compat.v1.logging.ERROR)

# tf.get_logger().setLevel("ERROR")


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


def _save_eval_results(
    final_predictions_list,
    metrics_dir,
    logger,
    block_col,
    current_model=None,
    filename="eval_results.json",
):
    if not final_predictions_list:
        logger.warning(f"No predictions to save for {filename}.")
        return

    # Process final_predictions_list
    full_predictions_df = pd.DataFrame(final_predictions_list)
    y_true_final = full_predictions_df["y_true"]
    y_pred_final = full_predictions_df["y_pred"]

    # Handle numeric encoding for AUC and saving
    try:
        y_true_numeric = y_true_final.astype(float)
    except ValueError:
        unique_classes = sorted(y_true_final.unique())
        class_map = {c: i for i, c in enumerate(unique_classes)}
        y_true_numeric = y_true_final.map(class_map)

    full_predictions_df["y_true_numeric"] = y_true_numeric

    # Calculate Global AUC
    try:
        if (
            "y_pred_proba" in full_predictions_df.columns
            and not full_predictions_df["y_pred_proba"].isna().all()
            and len(y_true_final.unique()) > 1
        ):
            y_prob_clean = full_predictions_df["y_pred_proba"].fillna(0)
            roc_auc_global = float(roc_auc_score(y_true_numeric, y_prob_clean))
        else:
            roc_auc_global = None
    except Exception as e:
        logger.warning(f"Could not calculate global ROC AUC: {e}")
        roc_auc_global = None

    metrics_global = {
        "accuracy": float(accuracy_score(y_true_final, y_pred_final)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_final, y_pred_final)),
        "f1": float(f1_score(y_true_final, y_pred_final, average="macro")),
        "roc_auc": roc_auc_global,
        "classification_report": classification_report(
            y_true_final, y_pred_final, output_dict=True, zero_division=0
        ),
    }

    per_block_metrics_full = {}
    for block_id, group in full_predictions_df.groupby("block"):
        try:
            if (
                "y_pred_proba" in group.columns
                and not group["y_pred_proba"].isna().all()
                and len(group["y_true"].unique()) > 1
            ):
                y_prob_block = group["y_pred_proba"].fillna(0)
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
        "task": "classification",
        "approved": True,
        "metrics": _jsonable(metrics_global),
        "thresholds": {},
        "blocks": {
            "block_col": block_col,
            "evaluated_blocks": _sorted_blocks(full_predictions_df["block"]),
            "per_block_metrics_full": _jsonable(per_block_metrics_full),
            "per_block_approved": {},
            "approved_blocks_count": 0,
            "rejected_blocks_count": 0,
            "top_worst_blocks": [],
        },
        "predictions": full_predictions_df.head(100).to_dict(orient="records"),
        "model_info": {
            "type": type(current_model).__name__ if current_model else "Unknown",
            "is_ipip": True,
        },
    }

    eval_path = metrics_dir / filename
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {eval_path}.")


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
    target_files: List[str] | None = None,
    rest_preprocess_file: str | None = None,
    skip_initial_preprocessing: bool = False,
    skip_rest_preprocessing: bool = False,
    target_col: str | None = None,
    window_size: int | None = None,
    block_col: str | None = None,
    ipip_config: dict | None = None,
    dir_predictions: Optional[str] = None,
    prediction_only: bool = False,
    encoding: str = "utf-8",
    file_type: str = "csv",
    max_history_size: int = 5,
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

    control_file = control_dir / "control_file.txt"

    model_path = output_dir / f"{pipeline_name}.pkl"

    # Logger
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()

    logger.info("Pipeline (IPIP) started — GLOBAL mode over all blocks.")

    if not block_col:
        raise ValueError("You must provide block_col explicitly (e.g., 'chunk').")

    # 1) Load dataset
    # 1) Determine files to process
    files_to_process = []  # List of (filename, preloaded_df_or_None, mtime_or_None)

    # Note: data_loader returns (df, filename, mtime)

    if target_files:
        for f in target_files:
            files_to_process.append((f, None, None))
    elif target_file:
        files_to_process.append((target_file, None, None))
    else:
        # Auto-discovery / Directory scan mode
        # In this mode, we find ONE new file as per original logic
        df_found, fname_found, mtime_found = data_loader(
            logger,
            data_dir,
            control_dir,
            delimiter=delimiter,
            target_file=None,
            block_col=block_col,
            encoding=encoding,
            file_type=file_type,
        )
        if df_found.empty:
            logger.warning("No new data to process.")
            return
        files_to_process.append((fname_found, df_found, mtime_found))

    # Lists to accumulate processed parts
    X_parts = []
    y_parts = []
    processed_files_meta = []  # To update control file later: (filename, mtime)

    for i, (fname, preloaded_df, preloaded_mtime) in enumerate(files_to_process):
        # Load if needed
        if preloaded_df is not None:
            df_curr = preloaded_df
            current_mtime = preloaded_mtime
        else:
            df_curr, _, current_mtime = data_loader(
                logger,
                data_dir,
                control_dir,
                delimiter=delimiter,
                target_file=fname,
                block_col=block_col,
                encoding=encoding,
                file_type=file_type,
            )

        if df_curr.empty:
            logger.warning(f"File {fname} is empty or could not be loaded. Skipping.")
            continue

        # Track for control file update
        if fname and current_mtime:
            processed_files_meta.append((fname, current_mtime))

        # Check prior history for incremental runs
        control_file_size = 0
        if control_file.exists():
            control_file_size = control_file.stat().st_size

        is_first = i == 0
        # If incremental mode (no explicit target_files) and history exists, this is NOT the initial file
        if not target_files and control_file_size > 0:
            is_first = False

        should_skip = (
            skip_initial_preprocessing if is_first else skip_rest_preprocessing
        )

        script_path = (
            preprocess_file if is_first else (rest_preprocess_file or preprocess_file)
        )

        logger.info(
            f"Processing file {fname} (Index {i}). Skip Preproc: {should_skip}. Script: {script_path}"
        )

        if should_skip:
            # RAW Mode: splitting only
            if not target_col:
                raise ValueError(
                    "target_col must be specified when skipping preprocessing."
                )

            # Check if target_col exists
            if target_col not in df_curr.columns:
                raise ValueError(
                    f"Column '{target_col}' not found in {fname} (Columns: {list(df_curr.columns)})."
                )

            X_curr = df_curr.drop(columns=[target_col], errors="ignore")
            y_curr = df_curr[target_col]
        else:
            # Script Mode
            spec = importlib.util.spec_from_file_location(
                f"custom_preproc_{i}", script_path
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            if not hasattr(mod, "data_preprocessing"):
                raise AttributeError(
                    f"{script_path} must define data_preprocessing(df)->(X,y)"
                )
            X_curr, y_curr = mod.data_preprocessing(df_curr)

            if isinstance(y_curr, pd.DataFrame):
                y_curr = y_curr.iloc[:, 0]

        # Align columns? (Optional safety step, but generally expected to match)
        X_parts.append(X_curr)
        y_parts.append(y_curr)
        logger.info(f"File {fname} processed. X shape: {X_curr.shape}.")

    if not X_parts:
        logger.warning("No data resulted from processing files.")
        return

    # Concatenate all parts
    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)

    logger.info(f"Total Combined Data: {X.shape[0]} rows, {X.shape[1]} columns.")

    # Handle block_col logic
    # Handle block_col logic
    # In multi-file/flexible mode, we require block_col to be in the final X.
    if block_col not in X.columns:
        raise ValueError(
            f"block_col='{block_col}' not found in processed data. "
            "Ensure it is present in input files or created by preprocessing."
        )
    X[block_col] = X[block_col].astype(str)

    # Determine last_processed_file for logging/training functions
    last_processed_file = processed_files_meta[-1][0] if processed_files_meta else None

    # --- Start Sequential IPIP Logic with Year-Based Splitting ---
    # Sort blocks naturally
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
    _model_instance = model_instance

    # Identify Unique Years
    year_col = ipip_config.get("year_col") if ipip_config else None
    unique_years = []

    if year_col and year_col in X.columns:
        try:
            # Extract unique years from the column
            unique_years = sorted(
                pd.to_datetime(X[year_col], errors="coerce", dayfirst=True)
                .dt.year.dropna()
                .unique()
                .astype(int)
                .astype(str)
                .tolist()
            )
            logger.info(f"Identified Years from '{year_col}': {unique_years}")
        except Exception as e:
            logger.warning(f"Failed to extract years from {year_col}: {e}")
            unique_years = []

    if not unique_years:
        # Fallback to block prefix logic or just unique blocks if no year concept
        # If no explicit year_col, we might try to infer from block names or just treat as one big unrecognized year?
        # Current logic tried to map every block.
        # Let's keep a simplified fallback:
        # If no year_col, we just process everything as one "Unknown" year or iterate blocks naturally?
        # If split_years is True but no year_col found/configured, we rely on prefixes.

        # Original logic fallback:
        year_map = {}
        for b in all_blocks:
            year_map[b] = str(b).split("-")[0]
        unique_years = sorted(list(set(year_map.values())))
        logger.info(f"Identified Years from block prefixes: {unique_years}")

    # Track processed blocks for retraining history
    # Correction: 'processed_blocks_history' stores blocks we have trained on.
    # For recurring blocks ("1", "2"), simply storing "1" is ambiguous if we mean "2022-1".
    # BUT, the model is trained on DATA.
    # The requirement is: "Subsequent Years: Using the model from the previous year".
    # This means the MODEL object accumulates knowledge.
    # 'processed_blocks_history' is used to selecting data for retraining: X[X[block_col].isin(processed_blocks_history)]
    # IF block "1" is in history, it selects ALL block "1" data (from 2022, 2023...).
    # FAST FIX: When selecting retraining data, we should probably select ALL data processed SO FAR (up to current year/block).
    # OR, if the requirement is to use "sequential retraining",
    # usually it means training on everything seen so far.
    # So if we are in 2023, block 1, we want to retrain on (2022-all) + (2023-1).
    # If we just filter by block_id "1", we get 2022-1 and 2023-1.
    # We essentially need to accumulate the DATAFRAME indices or mask, not just block names, if block names are not unique.

    # STRATEGY:
    # Maintain a 'historical_indices' list or mask.
    historical_mask = pd.Series(False, index=X.index)

    features_to_drop_always = []
    if year_col:
        features_to_drop_always.append(year_col)

    def _clean_for_train(df):
        return df.drop(columns=features_to_drop_always, errors="ignore")

    def _clean_for_predict(df):
        cols = features_to_drop_always + [block_col]
        return df.drop(columns=cols, errors="ignore")

    for year_idx, year in enumerate(unique_years):
        logger.info(f"=== Processing Year: {year} ===")

        # 1. Filter X for this specific year
        # We need to act on the specific subset of rows for this year.
        if year_col:
            # Re-extract year to be safe or use pre-calculated mask
            # This is a bit expensive inside loop but safe
            current_year_mask = pd.to_datetime(
                X[year_col], errors="coerce", dayfirst=True
            ).dt.year == int(year)
            X_year = X[current_year_mask]
        else:
            # Fallback: Filter by blocks that map to this year (using prefix logic)
            # Re-create map for this fallback case
            current_year_blocks = [
                b for b in all_blocks if str(b).startswith(str(year))
            ]
            X_year = X[X[block_col].isin(current_year_blocks)]

        if X_year.empty:
            logger.warning(f"No data found for year {year}. Skipping.")
            continue

        y_year = y.loc[X_year.index]

        # Identify blocks present IN THIS YEAR'S data
        # Sort naturally
        year_blocks = _sorted_blocks(X_year[block_col])
        logger.info(f"Blocks found in year {year}: {year_blocks}")

        if not year_blocks:
            continue

        # Local results accumulator for this year
        year_predictions_list = []

        start_idx = 0

        # --- PHASE 1: INITIAL TRAINING ---
        if year_idx == 0:
            if len(year_blocks) < 2:
                logger.warning(
                    f"Year {year} has fewer than 2 blocks. Cannot perform initial training."
                )
                continue

            logger.info(
                f"Sequential Run: Initial training on blocks 0 & 1 of year {year}."
            )
            initial_train_blocks = year_blocks[:2]

            X_initial_train_data = X_year[X_year[block_col].isin(initial_train_blocks)]
            y_initial_train_data = y.loc[X_initial_train_data.index]

            if X_initial_train_data.empty:
                logger.error("Initial training data is empty. Cannot proceed.")
                return

            # Update history mask
            historical_mask.loc[X_initial_train_data.index] = True

            # DROP ONLY EXTRA COLS (Keep block_col for default_train logic)
            X_train_clean = _clean_for_train(X_initial_train_data)

            current_model, X_test_dummy, y_test_dummy, train_results = default_train(
                X=X_train_clean,
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

            # processed_blocks_history.extend(initial_train_blocks) # No longer used

            # --- EVALUATE INITIAL MODEL ---
            eval_block_id_initial = year_blocks[1]
            X_initial_eval = X_year[X_year[block_col] == eval_block_id_initial]
            y_initial_eval = y.loc[X_initial_eval.index]

            # DROP ALL EXTRA COLUMNS AND BLOCK COL FOR PREDICT
            X_eval_clean = _clean_for_predict(X_initial_eval)

            if current_model and not X_eval_clean.empty:
                predictions = current_model.predict(X_eval_clean)
                try:
                    probas = current_model.predict_proba(X_eval_clean)
                    if probas.shape[1] == 2:
                        y_pred_proba = probas[:, 1]
                    else:
                        y_pred_proba = (
                            probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
                        )
                except Exception:
                    y_pred_proba = [None] * len(predictions)

                for yt, yp, ypp in zip(y_initial_eval, predictions, y_pred_proba):
                    pred_entry = {
                        "y_true": yt,
                        "y_pred": yp,
                        "y_pred_proba": ypp,
                        "block": eval_block_id_initial,
                    }
                    final_predictions_list.append(pred_entry)
                    year_predictions_list.append(pred_entry)

            start_idx = 1

        else:
            # --- PHASE 2: CONTINUITY ---
            first_block_of_year = year_blocks[0]
            logger.info(
                f"Evaluating first block of year {year} ({first_block_of_year}) with previous year's model."
            )

            X_eval_step = X_year[X_year[block_col] == first_block_of_year]
            y_eval_step = y.loc[X_eval_step.index]

            X_eval_clean = _clean_for_predict(X_eval_step)

            if current_model and not X_eval_clean.empty:
                predictions = current_model.predict(X_eval_clean)
                try:
                    probas = current_model.predict_proba(X_eval_clean)
                    if probas.shape[1] == 2:
                        y_pred_proba = probas[:, 1]
                    else:
                        y_pred_proba = (
                            probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
                        )
                except Exception:
                    y_pred_proba = [None] * len(predictions)

                for yt, yp, ypp in zip(y_eval_step, predictions, y_pred_proba):
                    pred_entry = {
                        "y_true": yt,
                        "y_pred": yp,
                        "y_pred_proba": ypp,
                        "block": first_block_of_year,
                    }
                    final_predictions_list.append(pred_entry)
                    year_predictions_list.append(pred_entry)

            start_idx = 0

        # --- PHASE 3: SEQUENTIAL LOOP FOR THE YEAR ---
        for i in range(start_idx, len(year_blocks) - 1):
            retrain_up_to_this_block = year_blocks[i]
            eval_next_block = year_blocks[i + 1]

            # Update history with the block we just finished (retrain_up_to_this_block)
            # Actually, in the loop, we retrain on [history] + [current_block].
            # Logic:
            # i=0: retrain on (initial blocks). BUT we just entered loop.
            # We want to Retrain on (History + retrain_up_to_this_block).
            # Then Predict on (eval_next_block).

            # Add retrain_up_to_this_block to history
            current_block_data = X_year[X_year[block_col] == retrain_up_to_this_block]
            historical_mask.loc[current_block_data.index] = True

            logger.info(
                f"Sequential Run: Retraining with all historical data up to year {year}, block '{retrain_up_to_this_block}'."
            )

            # Select ALL historical data
            X_retrain_data = X[historical_mask]
            y_retrain_data = y.loc[X_retrain_data.index]

            if X_retrain_data.empty:
                continue

            # Oracle Data
            X_eval_step = X_year[X_year[block_col] == eval_next_block]
            y_eval_step = y.loc[X_eval_step.index]

            # Clean X for retrain (KEEP block_col)
            X_retrain_clean = _clean_for_train(X_retrain_data)
            # Clean X next for pruning? default_retrain might use block_col there too?
            # default_retrain: if X_next provided, it calls predict.
            # predict usually needs NO block col.
            # But wait, default_retrain L442: pruning_X = X_next.drop(columns=[block_col], errors="ignore")
            # So default_retrain EXPECTS X_next to potentially have block_col and drops it.
            # safe to pass with block_col or without?
            # safest: pass with block_col (cleaned only of year), let default_retrain drop block_col if it wants.
            X_next_clean = _clean_for_train(X_eval_step)

            current_model, X_test_dummy, y_test_dummy, retrain_results = (
                default_retrain(
                    X=X_retrain_clean,
                    y=y_retrain_data,
                    last_processed_file=last_processed_file,
                    model_path=current_model,
                    random_state=random_state,
                    logger=logger,
                    output_dir=metrics_dir,
                    block_col=block_col,
                    ipip_config=ipip_config,
                    model_instance=_model_instance,
                    X_next=X_next_clean,
                    y_next=y_eval_step,
                )
            )

            if retrain_results:
                training_history.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "year": str(year),
                        "block": str(retrain_up_to_this_block),
                        "p": retrain_results.get("p"),
                        "b": retrain_results.get("b"),
                        "num_ensembles": retrain_results.get("num_ensembles"),
                        "replacement_percentage": retrain_results.get(
                            "replacement_percentage"
                        ),
                        "type": "retrain",
                    }
                )

            # Evaluate on Next Block
            X_next_predict = _clean_for_predict(X_eval_step)

            if current_model and not X_next_predict.empty:
                predictions = current_model.predict(X_next_predict)
                try:
                    probas = current_model.predict_proba(X_next_predict)
                    if probas.shape[1] == 2:
                        y_pred_proba = probas[:, 1]
                    else:
                        y_pred_proba = (
                            probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
                        )
                except Exception:
                    y_pred_proba = [None] * len(predictions)

                for yt, yp, ypp in zip(y_eval_step, predictions, y_pred_proba):
                    pred_entry = {
                        "y_true": yt,
                        "y_pred": yp,
                        "y_pred_proba": ypp,
                        "block": eval_next_block,
                    }
                    final_predictions_list.append(pred_entry)
                    year_predictions_list.append(pred_entry)

        # --- PHASE 4: YEAR END FINALIZATION ---
        last_block_of_year = year_blocks[-1]

        # Add last block to history
        last_block_data = X_year[X_year[block_col] == last_block_of_year]
        historical_mask.loc[last_block_data.index] = True

        logger.info(
            f"End of Year {year}. Performing final model update on full year data."
        )

        X_full_year_history = X[historical_mask]
        y_full_year_history = y.loc[X_full_year_history.index]

        X_full_clean = _clean_for_train(X_full_year_history)

        current_model, _, _, _ = default_retrain(
            X=X_full_clean,
            y=y_full_year_history,
            last_processed_file=last_processed_file,
            model_path=current_model,
            random_state=random_state,
            logger=logger,
            output_dir=metrics_dir,
            block_col=block_col,
            ipip_config=ipip_config,
            model_instance=_model_instance,
            X_next=None,
            y_next=None,
        )

        # --- SAVE YEAR RESULTS ---
        if ipip_config and ipip_config.get("split_years"):
            logger.info(f"Saving results for Year {year}.")
            _save_eval_results(
                year_predictions_list,  # Only this year's results
                metrics_dir,
                logger,
                block_col,
                current_model,
                filename=f"eval_results_year_{year}.json",
            )

    # --- END OF YEAR LOOP ---

    # Generate final global eval_results.json (Aggregated)
    _save_eval_results(
        final_predictions_list,
        metrics_dir,
        logger,
        block_col,
        current_model,
        filename="eval_results.json",
    )

    # Save to eval_history (using helper)
    try:
        history_dir = metrics_dir / "eval_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_fname = f"eval_results_{ts}.json"

        _save_eval_results(
            final_predictions_list,
            history_dir,
            logger,
            block_col,
            current_model,
            filename=history_fname,
        )
    except Exception as e:
        logger.warning(f"Could not save historical evaluation results: {e}")

    # --- NEW: Save training history (p, b, ensembles) ---
    try:
        history_json_path = metrics_dir / "training_history.json"
        # Load existing history if file exists to append?
        # For simplicity and robustness during sequential runs with year loop, we are accumulating in `training_history`.
        # If we restart the pipeline for a new file, we might overwrite.
        # But `run_pipeline` is called per file.
        # Actually, if we process multiple files sequentially (incremental), we usually want to append.
        # Let's read, append, write.

        current_history = []
        if history_json_path.exists():
            try:
                with open(history_json_path, "r") as f:
                    current_history = json.load(f)
            except json.JSONDecodeError:
                current_history = []

        # Avoid duplication if possible (simple check by timestamp might not be enough if fast)
        # Just append new entries
        current_history.extend(training_history)

        with open(history_json_path, "w") as f:
            json.dump(current_history, f, indent=4)
        logger.info(f"Training history saved to {history_json_path}.")

    except Exception as e:
        logger.warning(f"Could not save training_history.json: {e}")

    # Update control file to mark this file as processed
    # We iterate over all processed files
    for fname, mtime in processed_files_meta:
        if fname and mtime:
            _upsert_control_entry(
                control_dir / "control_file.txt",
                fname,
                mtime,
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

    # ========================================================================
    # SAVE HISTORY
    # ========================================================================
    try:
        # Construct history record
        history_record = {
            "timestamp": time.time(),
            "readable_timestamp": time.ctime(),
            "batch_id": last_processed_file
            if "last_processed_file" in locals()
            else "unknown",
            "pipeline_type": "ipip",
        }

        # Load latest eval results
        eval_res_file = metrics_dir / "eval_results.json"
        if eval_res_file.exists():
            try:
                with open(eval_res_file, "r") as f:
                    eval_data = json.load(f)
                    if isinstance(eval_data, list) and eval_data:
                        history_record["eval_metrics"] = eval_data[-1]
                    elif isinstance(eval_data, dict):
                        history_record["eval_metrics"] = eval_data
            except:
                pass

        # Load latest training history
        train_hist_file = metrics_dir / "training_history.json"
        if train_hist_file.exists():
            try:
                with open(train_hist_file, "r") as f:
                    t_data = json.load(f)
                    # Maybe too big to save all? Just save last entry
                    if isinstance(t_data, list) and t_data:
                        history_record["last_training_step"] = t_data[-1]
            except:
                pass

        HistoryManager.append_history_record(
            str(metrics_dir / "history.json"),
            history_record,
            max_history=max_history_size,
        )
        logger.info(f"History updated (max_size={max_history_size})")

    except Exception as h_e:
        logger.error(f"Failed to save history: {h_e}")
