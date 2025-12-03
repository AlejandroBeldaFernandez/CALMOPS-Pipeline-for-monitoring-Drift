# pipeline_block/pipeline_block.py

import os
import time
import json

import importlib.util
from typing import Optional, List
import functools
import inspect

import pandas as pd
import joblib
from pathlib import Path

from calmops.logger.logger import PipelineLogger
from calmops.utils import get_pipelines_root


# NOTE: these imports follow your folder structure indicated in the file itself
from .modules.data_loader import data_loader
from .modules.check_drift import check_drift
from .modules.default_train_retrain import default_train, default_retrain
from .modules.evaluator import evaluate_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")

# -------------------------------  Circuit Breaker  -------------------------------


def _health_path(metrics_dir: Path) -> Path:
    return metrics_dir / "health.json"


def _load_health(metrics_dir: Path) -> dict:
    p = _health_path(metrics_dir)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None}


def _save_health(metrics_dir: Path, data: dict) -> None:
    p = _health_path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _should_pause(health: dict) -> bool:
    paused_until = health.get("paused_until")
    if paused_until is None:
        return False
    try:
        return time.time() < float(paused_until)
    except Exception:
        return False


def _update_on_result(
    health: dict, approved: bool, backoff_minutes: int, max_failures: int
) -> dict:
    if approved:
        health.update(
            {"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None}
        )
    else:
        health["consecutive_failures"] = int(health.get("consecutive_failures", 0)) + 1
        health["last_failure_ts"] = time.time()
        if health["consecutive_failures"] >= max_failures:
            health["paused_until"] = time.time() + backoff_minutes * 60
    return health


# -------------------------------  Preprocess Loader  -------------------------------


def _load_preprocess_func(preprocess_file: str):
    """Loads data_preprocessing(df) from an external .py file."""
    if not Path(preprocess_file).exists():
        raise FileNotFoundError(f"Invalid preprocessing file: {preprocess_file}")
    spec = importlib.util.spec_from_file_location(
        "custom_preprocess_module", preprocess_file
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "data_preprocessing"):
        raise AttributeError(f"{preprocess_file} must define data_preprocessing(df)")
    return getattr(mod, "data_preprocessing")


# -------------------------------  Block Utils  -------------------------------


def _sorted_blocks(block_series: pd.Series) -> list[str]:
    """Sorts block identifiers naturally (e.g., 'block_1', 'block_2', 'block_10')."""
    import re

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


# -------------------------------  Main Pipeline (block_wise only)  -------------------------------


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
    param_grid: dict | None = None,
    cv: int | None = None,
    custom_train_file: str | None = None,
    custom_retrain_file: str | None = None,
    custom_fallback_file: str | None = None,
    delimiter: str = ",",
    target_file: str | None = None,
    window_size: int | None = None,
    breaker_max_failures: int = 3,
    breaker_backoff_minutes: int = 120,
    block_col: str
    | None = None,  # block column within the dataset (must exist for block_wise)
    eval_blocks: list[str] | None = None,  # blocks to evaluate; if None => last block
    split_within_blocks: bool = False,  # New: if True, splits each block for training and evaluation
    train_percentage: float = 0.8,  # New: percentage for training if split_within_blocks is True
    fallback_strategy: str = "global",
    prediction_only: bool = False,
    dir_predictions: Optional[str] = None,
    encoding: str = "utf-8",
    file_type: str = "csv",
) -> None:
    """
    Orchestration for block_wise ONLY:
      1) Load and preprocess.
      2) Determine train_blocks (all except eval_blocks) and eval_blocks (default: last).
      3) If no model -> TRAIN (global + per block) and eval on eval_blocks.
      4) If model exists -> check_drift (between blocks) -> if drift -> RETRAIN only drifted blocks; otherwise, NO-OP.
      5) Evaluate on eval_blocks and update circuit breaker.
    """
    # Paths
    project_root = get_pipelines_root()
    base_dir = project_root / "pipelines" / pipeline_name
    output_dir = base_dir / "models"
    control_dir = base_dir / "control"
    logs_dir = base_dir / "logs"
    metrics_dir = base_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{pipeline_name}.pkl"

    # Logger
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()
    logger.info("Pipeline started (block_wise).")

    # Circuit breaker
    health = _load_health(metrics_dir)
    if _should_pause(health):
        logger.warning("Retraining paused by circuit breaker. Skipping this run.")
        return

    # Validate thresholds: do not mix class/reg
    if {"accuracy", "f1", "balanced_accuracy"} & set(thresholds_perf.keys()) and {
        "rmse",
        "mae",
        "mse",
        "r2",
    } & set(thresholds_perf.keys()):
        raise ValueError(
            "Cannot define classification and regression thresholds simultaneously."
        )

    # Preprocess
    preprocess_func = _load_preprocess_func(preprocess_file)

    # 1) Full load (data_loader already manages control snapshots)
    df_full, last_processed_file, mtime = data_loader(
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

    # 2) Preprocessing -> X, y (block_col must remain in X to identify eval blocks)
    sig = inspect.signature(preprocess_func)
    if "block_cols" in sig.parameters:
        preprocess_func = functools.partial(preprocess_func, block_cols=block_col)

    # Attempt to retrieve block_col from raw data
    original_blocks_series = None
    if block_col in df_full.columns:
        original_blocks_series = df_full[block_col].astype(str)

    X, y = preprocess_func(df_full)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Preprocess must return X as a pandas DataFrame.")
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError(
            "Preprocess must return y as a pandas Series or single-column DataFrame."
        )
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a single target column.")
        y = y.iloc[:, 0]

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

    if (block_col is None) or (block_col not in X.columns):
        raise ValueError("block_col must exist in X to operate in block_wise mode.")

    # Block series (as str)
    blocks_series = X[block_col].astype(str)

    # 2.b) Determine eval_blocks (default: last block) and train_blocks (rest)
    all_blocks_sorted = _sorted_blocks(blocks_series)
    if not all_blocks_sorted:
        logger.error("No block ids detected in the data.")
        return

    if split_within_blocks:
        train_blocks = all_blocks_sorted
        eval_blocks = all_blocks_sorted
    else:
        if eval_blocks:
            eval_blocks = [str(b) for b in eval_blocks if str(b) in all_blocks_sorted]
            if not eval_blocks:
                eval_blocks = [all_blocks_sorted[-1]]
        else:
            eval_blocks = [all_blocks_sorted[-1]]
        train_blocks = [b for b in all_blocks_sorted if b not in set(eval_blocks)]

    logger.info(
        f"[BLOCKS] train_blocks={train_blocks} | eval_blocks={eval_blocks} | block_col={block_col}"
    )

    # 3) check_drift (between blocks). Passes PERFORMANCE thresholds (not thresholds_drift).
    decision = check_drift(
        X=X,
        y=y,
        logger=logger,
        perf_thresholds=thresholds_perf,  # <-- performance thresholds
        model_filename=f"{pipeline_name}.pkl",
        output_dir=metrics_dir,
        model_dir=output_dir,
        data_dir=data_dir,
        control_dir=control_dir,
        current_filename=last_processed_file,
        block_col=block_col,
        prediction_only=prediction_only,
    )

    try:
        # === TRAIN (first time) ===
        if decision == "train" or not model_path.exists():
            logger.info("TRAIN phase (block_wise).")
            if custom_train_file:
                spec_t = importlib.util.spec_from_file_location(
                    "train_module", custom_train_file
                )
                mod_t = importlib.util.module_from_spec(spec_t)
                assert spec_t.loader is not None
                spec_t.loader.exec_module(mod_t)
                if not hasattr(mod_t, "train"):
                    raise AttributeError(f"{custom_train_file} must define train(...)")
                model, X_test, y_test, _ = mod_t.train(
                    X, y, last_processed_file, logger, metrics_dir
                )
            else:
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
                    # block_wise ONLY
                    blocks=blocks_series,
                    block_col=block_col,
                    test_blocks=eval_blocks,  # eval = last (or as passed)
                    split_within_blocks=split_within_blocks,
                    train_percentage=train_percentage,
                    fallback_strategy=fallback_strategy,
                )

            # Re-inject block column into X_test for safety
            if block_col not in X_test.columns:
                X_test = X_test.copy()
                X_test[block_col] = blocks_series.loc[X_test.index].astype(str)

            # Persist model
            try:
                joblib.dump(model, model_path)
                logger.info(f"Model saved: {model_path}")
            except Exception as e:
                logger.warning(f"Could not persist model to {model_path}: {e}")

            # Evaluate on eval_blocks
            approved = evaluate_model(
                model_or_path=model,
                X_eval=X_test,
                y_eval=y_test,
                X_full=X,
                logger=logger,
                metrics_dir=metrics_dir,
                control_dir=control_dir,
                data_file=last_processed_file,
                thresholds=thresholds_perf,
                block_col=block_col,
                evaluated_blocks=eval_blocks,
                include_predictions=True,
                max_pred_examples=10,
                mtime=mtime,
                dir_predictions=dir_predictions,
            )

            health = _update_on_result(
                health, bool(approved), breaker_backoff_minutes, breaker_max_failures
            )
            _save_health(metrics_dir, health)

        # === RETRAIN (only drifted blocks) ===
        elif decision == "retrain":
            logger.info("RETRAIN phase (block_wise).")

            # Load which blocks have drift from the drift JSON
            drift_json = metrics_dir / "drift_results.json"
            drifted_blocks: List[str] = []
            try:
                with open(drift_json, "r", encoding="utf-8") as f:
                    dr = json.load(f)
                bw = dr.get("blockwise", {}) or {}
                # statistical flags per block (if the block appears in any pair with drift)
                stat_flags = bw.get("by_block_stat_drift", {}) or {}
                # performance flags per block (dict of metrics -> bool)
                perf_flags = (
                    (bw.get("performance", {}) or {}).get("current", {}) or {}
                ).get("flags", {}) or {}

                stat_set = {str(b) for b, flag in stat_flags.items() if bool(flag)}
                perf_set = {
                    str(b)
                    for b, mdict in perf_flags.items()
                    if isinstance(mdict, dict) and any(bool(v) for v in mdict.values())
                }
                drifted_blocks = sorted(list(perf_set | stat_set))
            except Exception:
                logger.warning(
                    "Could not read drift_results.json to decide on drifted blocks."
                )

            if drifted_blocks:
                logger.info(f"[RETRAIN] Drifted blocks to retrain: {drifted_blocks}")
            else:
                logger.info(
                    "[RETRAIN] No drifted blocks in train; current champion will be evaluated."
                )

            if custom_retrain_file and drifted_blocks:
                spec_r = importlib.util.spec_from_file_location(
                    "retrain_module", custom_retrain_file
                )
                mod_r = importlib.util.module_from_spec(spec_r)
                assert spec_r.loader is not None
                spec_r.loader.exec_module(mod_r)
                if not hasattr(mod_r, "retrain"):
                    raise AttributeError(
                        f"{custom_retrain_file} must define retrain(...)"
                    )
                model, X_test, y_test, _ = mod_r.retrain(
                    X, y, last_processed_file, logger, metrics_dir
                )
            elif drifted_blocks:
                model, X_test, y_test, _ = default_retrain(
                    X,
                    y,
                    last_processed_file,
                    model_path=model_path,
                    mode=retrain_mode,
                    random_state=random_state,
                    logger=logger,
                    output_dir=metrics_dir,
                    window_size=window_size,
                    blocks=blocks_series,
                    block_col=block_col,
                    test_blocks=eval_blocks,
                    drifted_blocks=drifted_blocks,  # <--- key: only these blocks
                    split_within_blocks=split_within_blocks,
                    train_percentage=train_percentage,
                    fallback_strategy=fallback_strategy,
                )
            else:
                # No drift in train blocks -> NO-OP; prepare eval on eval_blocks
                model = joblib.load(model_path)
                mask_eval = blocks_series.isin(eval_blocks)
                X_test = X.loc[mask_eval].copy()
                y_test = y.loc[mask_eval]

            # Save (if retraining occurred)
            if isinstance(model, object):
                try:
                    joblib.dump(model, model_path)
                    logger.info(f"Model saved: {model_path}")
                except Exception as e:
                    logger.warning(f"Could not persist model to {model_path}: {e}")

            # Ensure block_col in X_test
            if block_col not in X_test.columns:
                X_test = X_test.copy()
                X_test[block_col] = blocks_series.loc[X_test.index].astype(str)

            # Evaluate
            approved = evaluate_model(
                model_or_path=model,
                X_eval=X_test,
                y_eval=y_test,
                X_full=X,
                logger=logger,
                metrics_dir=metrics_dir,
                control_dir=control_dir,
                data_file=last_processed_file,
                thresholds=thresholds_perf,
                block_col=block_col,
                evaluated_blocks=eval_blocks,
                include_predictions=True,
                max_pred_examples=10,
                mtime=mtime,
                dir_predictions=dir_predictions,
            )

            # Fallback if not approved
            if (not approved) and (fallback_mode is not None):
                logger.info(f"Fallback retrain with mode={fallback_mode}.")
                if custom_fallback_file:
                    spec_f = importlib.util.spec_from_file_location(
                        "fallback_module", custom_fallback_file
                    )
                    mod_f = importlib.util.module_from_spec(spec_f)
                    assert spec_f.loader is not None
                    spec_f.loader.exec_module(mod_f)
                    if not hasattr(mod_f, "fallback"):
                        raise AttributeError(
                            f"{custom_fallback_file} must define fallback(...)"
                        )
                    model, X_test, y_test, _ = mod_f.fallback(
                        X, y, last_processed_file, logger, metrics_dir
                    )
                else:
                    model, X_test, y_test, _ = default_retrain(
                        X,
                        y,
                        last_processed_file,
                        model_path=model_path,
                        mode=fallback_mode,
                        random_state=random_state,
                        logger=logger,
                        output_dir=metrics_dir,
                        window_size=window_size,
                        blocks=blocks_series,
                        block_col=block_col,
                        test_blocks=eval_blocks,
                        drifted_blocks=drifted_blocks,  # keep focus on affected blocks
                        fallback_strategy=fallback_strategy,
                    )

                if block_col not in X_test.columns:
                    X_test = X_test.copy()
                    X_test[block_col] = blocks_series.loc[X_test.index].astype(str)

                try:
                    joblib.dump(model, model_path)
                    logger.info(f"Fallback model saved: {model_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not persist fallback model to {model_path}: {e}"
                    )

                approved = evaluate_model(
                    model_or_path=model,
                    X_eval=X_test,
                    y_eval=y_test,
                    X_full=X,
                    logger=logger,
                    metrics_dir=metrics_dir,
                    control_dir=control_dir,
                    data_file=last_processed_file,
                    thresholds=thresholds_perf,
                    block_col=block_col,
                    evaluated_blocks=eval_blocks,
                    include_predictions=True,
                    max_pred_examples=10,
                    mtime=mtime,
                    dir_predictions=dir_predictions,
                )

            health = _update_on_result(
                health, bool(approved), breaker_backoff_minutes, breaker_max_failures
            )
            _save_health(metrics_dir, health)

        elif decision == "predict":
            logger.info("Prediction-only mode. Loading model and predicting.")
            if model_path.exists():
                model = joblib.load(model_path)
                predictions = model.predict(X.drop(columns=[block_col]))
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
                        extra_pred_path = os.path.join(
                            dir_predictions,
                            f"predictions_{Path(last_processed_file).stem}.csv",
                        )
                        predictions_df.to_csv(extra_pred_path, index=False)
                        logger.info(f"Predictions also saved to {extra_pred_path}")
                    except Exception as e:
                        logger.warning(
                            f"Could not save predictions to dir_predictions ({dir_predictions}): {e}"
                        )
            else:
                logger.error("No model found to make predictions.")

        # === NO-OP (no retraining) -> evaluate champion on eval_blocks ===
        else:
            logger.info(
                "No retraining required. Re-evaluating current champion on eval_blocks."
            )
            if model_path.exists():
                model = joblib.load(model_path)
                # Evaluation subset = eval_blocks
                mask_eval = blocks_series.isin(eval_blocks)
                X_test = X.loc[mask_eval].copy()
                y_test = y.loc[mask_eval]

                if block_col not in X_test.columns:
                    X_test[block_col] = blocks_series.loc[mask_eval].astype(str)

                approved = evaluate_model(
                    model_or_path=model,
                    X_eval=X_test,
                    y_eval=y_test,
                    X_full=X,
                    logger=logger,
                    metrics_dir=metrics_dir,
                    control_dir=control_dir,
                    data_file=last_processed_file,
                    thresholds=thresholds_perf,
                    block_col=block_col,
                    evaluated_blocks=eval_blocks,
                    include_predictions=True,
                    max_pred_examples=10,
                    mtime=mtime,
                    dir_predictions=dir_predictions,
                )

                health = _update_on_result(
                    health,
                    bool(approved),
                    breaker_backoff_minutes,
                    breaker_max_failures,
                )
                _save_health(metrics_dir, health)
            else:
                logger.warning("Model file not found, nothing to evaluate.")

    except Exception as e:
        logger.error(f"[CRITICAL] Error in run_pipeline: {e}")
        raise
