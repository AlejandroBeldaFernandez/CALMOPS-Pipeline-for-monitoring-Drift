# pipeline/pipeline_ipip.py
from __future__ import annotations

import os
import json
import logging
import importlib.util
from pathlib import Path
from typing import Optional


import joblib
import pandas as pd

from calmops.logger.logger import PipelineLogger
from calmops.utils import get_project_root

# --- Robust imports of your modules ---
from .modules.data_loader import data_loader
from .modules.default_train_retrain import default_train, default_retrain
from .modules.evaluator import evaluate_model


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")


# =========================================================
# Helpers
# =========================================================
def _upsert_control_entry(
    control_file: Path, file_name: str, mtime: float, logger: logging.Logger
):
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
    *, model, pipeline_name: str, output_dir: Path, logger: logging.Logger
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{pipeline_name}.pkl"
    if model_path.exists():
        prev = model_path.with_name(
            f"{model_path.stem}_previous{model_path.suffix or '.pkl'}"
        )
        model_path.rename(prev)
        logger.info(f"[MODEL] Previous model backed up at {prev.resolve()}")
    joblib.dump(model, model_path)
    logger.info(f"💾 Model saved at {model_path.resolve()}")
    return model_path


def _load_python(file_path: str, func_name: str):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, func_name):
        raise AttributeError(f"{file_path} must define {func_name}(...)")
    return getattr(mod, func_name)


def _log_model_evolution(new_model, old_model, logger):
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


# =========================================================
# Run Pipeline
# =========================================================
def run_pipeline(
    *,
    pipeline_name: str,
    data_dir: str,
    preprocess_file: str,
    model_instance,
    retrain_mode: int,
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
) -> None:
    # Paths
    project_root = get_project_root()
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

    old_model = None
    if model_path.exists():
        try:
            old_model = joblib.load(model_path)
        except Exception as e:
            logger.warning(f"Could not load old model for evolution comparison: {e}")

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
    )
    if df_full.empty:
        logger.warning("No new data to process.")
        return
    if block_col not in df_full.columns:
        raise ValueError(f"block_col='{block_col}' not found in the loaded dataset.")

    # 2) Preprocess (the prepro chooses target and returns X,y)
    spec = importlib.util.spec_from_file_location("custom_preproc", preprocess_file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "data_preprocessing"):
        raise AttributeError(
            f"{preprocess_file} must define data_preprocessing(df)->(X,y)"
        )
    # Store block_col before preprocessing
    original_blocks_series = df_full[block_col].astype(str)

    X, y = mod.data_preprocessing(df_full)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    logger.info(f"Preprocessing OK: {X.shape[0]} rows, {X.shape[1]} columns.")

    # Re-align block index (in case the prepro does not preserve it in X)
    # Use the stored original_blocks_series
    try:
        blocks_series = original_blocks_series.loc[X.index]
    except Exception:
        common = X.index.intersection(original_blocks_series.index)
        X = X.loc[common]
        y = y.loc[common]
        blocks_series = original_blocks_series.loc[common]
    X = X.copy()
    X[block_col] = blocks_series

    if prediction_only:
        logger.info("Prediction-only mode. Loading model and predicting.")
        if model_path.exists():
            model = joblib.load(model_path)
            # The IPIP model might need the block_col
            predictions = model.predict(X)
            predictions_df = pd.DataFrame(predictions, columns=["prediction"])
            predictions_path = metrics_dir / f"predictions_{last_processed_file}.csv"
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Predictions saved to {predictions_path}")

            if dir_predictions:
                try:
                    os.makedirs(dir_predictions, exist_ok=True)
                    extra_pred_path = (
                        Path(dir_predictions) / f"predictions_{Path(last_processed_file).stem}.csv"
                    )
                    predictions_df.to_csv(extra_pred_path, index=False)
                    logger.info(f"Predictions also saved to {extra_pred_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not save predictions to dir_predictions ({dir_predictions}): {e}"
                    )
        else:
            logger.error("No model found to make predictions.")
        return  # Exit the pipeline

    is_first_run = not model_path.exists()

    # IPIP Strategy: Always adapt/retrain on new data.
    if is_first_run:
        logger.info("No existing model found — forcing TRAIN on first run.")
        decision = "train"
    else:
        logger.info("IPIP strategy: forcing RETRAIN/UPDATE on new data block.")
        decision = "retrain"

    # 4) TRAIN / RETRAIN
    model, X_test, y_test = None, None, None

    if decision == "train":
        logger.info("TRAIN (IPIP-FILM style).")
        model, X_test, y_test, _ = default_train(
            X=X,
            y=y,
            last_processed_file=last_processed_file,
            model_instance=model_instance,
            random_state=random_state,
            logger=logger,
            output_dir=metrics_dir,
            block_col=block_col,
            ipip_config=ipip_config,
        )

    elif decision == "retrain":
        logger.info("RETRAIN (IPIP-FILM style).")
        model, X_test, y_test, _ = default_retrain(
            X=X,
            y=y,
            last_processed_file=last_processed_file,
            model_path=model_path,
            random_state=random_state,
            logger=logger,
            output_dir=metrics_dir,
            block_col=block_col,
            ipip_config=ipip_config,
            model_instance=model_instance,
        )

    if old_model and model:
        evolution_info = _log_model_evolution(model, old_model, logger)
        evolution_path = metrics_dir / "evolution_results.json"
        with open(evolution_path, "w") as f:
            json.dump(evolution_info, f, indent=4)

    # 5) Evaluate model and decide persistence
    # For IPIP, we always persist because the model internally selects the best ensembles.
    # We run evaluate_model purely for logging metrics to the dashboard.
    if (
        model is not None
        and X_test is not None
        and not X_test.empty
        and y_test is not None
        and not y_test.empty
    ):
        logger.info("EVALUATION phase (logging metrics).")
        evaluate_model(
            model_or_path=model,
            X_eval=X_test,
            y_eval=y_test,
            X_full=X,
            logger=logger,
            metrics_dir=metrics_dir,
            control_dir=control_dir,
            data_file=last_processed_file,
            block_col=block_col,
            evaluated_blocks=pd.unique(X_test[block_col]).tolist(),
            mtime=last_mtime,
            dir_predictions=dir_predictions,
        )

    # Always persist for IPIP
    _persist_model(
        model=model, pipeline_name=pipeline_name, output_dir=output_dir, logger=logger
    )
