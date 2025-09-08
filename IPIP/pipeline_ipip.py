# pipeline/pipeline_ipip.py
from __future__ import annotations

import os
import json
import logging
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional

import joblib
import pandas as pd

from logger.logger import PipelineLogger

# --- Robust imports de tus módulos ---
from .modules.data_loader import data_loader
from .modules.check_drift import check_drift
from .modules.default_train_retrain import default_train, default_retrain


# =========================================================
# Helpers
# =========================================================
def _previous_model_path(model_path: str) -> str:
    root, ext = os.path.splitext(model_path)
    return f"{root}_previous{ext or '.pkl'}"

def _upsert_control_entry(control_file: Path, file_name: str, mtime: float, logger: logging.Logger):
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
    os.replace(tmp, control_file)
    logger.info(f"[CONTROL] Upserted {key} with mtime={mtime} into {control_file.resolve()}")

def _persist_model(
    *,
    model,
    pipeline_name: str,
    output_dir: str,
    logger: logging.Logger
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{pipeline_name}.pkl")
    if os.path.exists(model_path):
        prev = _previous_model_path(model_path)
        os.replace(model_path, prev)
        logger.info(f"[MODEL] Previous model backed up at {Path(prev).resolve()}")
    joblib.dump(model, model_path)
    logger.info(f"💾 Model saved at {Path(model_path).resolve()}")
    return model_path

def _load_python(file_path: str, func_name: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, func_name):
        raise AttributeError(f"{file_path} must define {func_name}(...)")
    return getattr(mod, func_name)

def _load_drifted_blocks(metrics_dir: str) -> List[str]:
    p = os.path.join(metrics_dir, "drift_results.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            jr = json.load(f)
        if "drifted_blocks" in jr and isinstance(jr["drifted_blocks"], list):
            return [str(x) for x in jr["drifted_blocks"]]
        bw = (jr.get("blockwise", {}) or {})
        if "drifted_blocks_stats" in bw:
            return [str(x) for x in bw["drifted_blocks_stats"]]
        return []
    except Exception:
        return []


# =========================================================
# Run Pipeline
# =========================================================
def run_pipeline(
    *,
    pipeline_name: str,
    data_dir: str,
    preprocess_file: str,
    thresholds_drift: dict,       # <-- se usa en check_drift
    thresholds_perf: dict,        # <-- se usa para aprobar en train/retrain
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
) -> None:

    # Paths
    base_dir   = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    output_dir = os.path.join(base_dir, "modelos")
    control_dir = os.path.join(base_dir, "control")
    logs_dir   = os.path.join(base_dir, "logs")
    metrics_dir = os.path.join(base_dir, "metrics")
    for d in (output_dir, control_dir, logs_dir, metrics_dir):
        os.makedirs(d, exist_ok=True)

    model_path = os.path.join(output_dir, f"{pipeline_name}.pkl")

    # Logger
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()
    logging.basicConfig()
    logger.info("Pipeline (IPIP) started — GLOBAL mode over all blocks.")

    if not block_col:
        raise ValueError("You must provide block_col explicitly (e.g., 'chunk').")

    # 1) Carga dataset
    df_full, last_processed_file, last_mtime = data_loader(
        logger, data_dir, control_dir, delimiter=delimiter, target_file=target_file, block_col=block_col
    )
    if df_full.empty:
        logger.warning("No new data to process.")
        return
    if block_col not in df_full.columns:
        raise ValueError(f"block_col='{block_col}' not found in the loaded dataset.")

    # 2) Preprocess (el prepro elige target y devuelve X,y)
    spec = importlib.util.spec_from_file_location("custom_preproc", preprocess_file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "data_preprocessing"):
        raise AttributeError(f"{preprocess_file} must define data_preprocessing(df)->(X,y)")
    X, y = mod.data_preprocessing(df_full)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    logger.info(f"Preprocesado OK: {X.shape[0]} filas, {X.shape[1]} columnas.")

    # Re-alinear índice de bloques (por si el prepro no lo preserva en X)
    try:
        blocks_series = df_full.loc[X.index, block_col].astype(str)
    except Exception:
        common = X.index.intersection(df_full.index)
        X = X.loc[common]
        y = y.loc[common]
        blocks_series = df_full.loc[common, block_col].astype(str)
    X = X.copy()
    X[block_col] = blocks_series

    # 3) Drift check (para decidir train/retrain) — usando thresholds_drift
    # Pasamos X con block_col porque check_drift lo necesita para cortar por bloque.
    decision = check_drift(
        X=X,
        y=y,
        logger=logger,
        drift_thresholds=thresholds_drift,   # umbrales de data-drift
        model_filename=f"{pipeline_name}.pkl",
        output_dir=metrics_dir,
        model_dir=output_dir,
        block_col=block_col,
    )

    is_first_run = not os.path.exists(model_path)
    if is_first_run and decision not in ("train", "retrain"):
        logger.info("No existing model found — forcing TRAIN on first run.")
        decision = "train"

    # 4) TRAIN / RETRAIN (la aprobación usa thresholds_perf)
    approved = False
    results: Optional[Dict] = None

    if decision == "train":
        logger.info("TRAIN (IPIP) sobre TODOS los bloques.")

        if custom_train_file:
            train_func = _load_python(custom_train_file, "train")
            # Intento con nombre estándar
            try:
                model, _, _, results = train_func(
                    X=X, y=y, last_processed_file=last_processed_file,
                    logger=logger, output_dir=metrics_dir,
                    ipip_config=ipip_config, thresholds_perf=thresholds_perf,
                    block_col=block_col, model_instance=model_instance, random_state=random_state
                )
            except TypeError:
                # Compat: algunos custom podrían usar 'perf_thresholds'
                model, _, _, results = train_func(
                    X=X, y=y, last_processed_file=last_processed_file,
                    logger=logger, output_dir=metrics_dir,
                    ipip_config=ipip_config, perf_thresholds=thresholds_perf,
                    block_col=block_col, model_instance=model_instance, random_state=random_state
                )
        else:
            model, _, _, results = default_train(
                X=X, y=y, last_processed_file=last_processed_file,
                model_instance=model_instance, random_state=random_state, logger=logger,
                output_dir=metrics_dir, block_col=block_col,
                ipip_config=ipip_config, thresholds_perf=thresholds_perf
            )
        approved = bool((results or {}).get("approval", {}).get("approved", False))

    elif decision == "retrain":
        logger.info("RETRAIN (IPIP).")
        affected_blocks = _load_drifted_blocks(metrics_dir)

        if custom_retrain_file:
            retrain_func = _load_python(custom_retrain_file, "retrain")
            try:
                model, _, _, results = retrain_func(
                    X=X, y=y, last_processed_file=last_processed_file,
                    logger=logger, output_dir=metrics_dir,
                    ipip_config=ipip_config, thresholds_perf=thresholds_perf,
                    block_col=block_col, drifted_blocks=affected_blocks,
                    model_instance=model_instance, random_state=random_state, window_size=window_size
                )
            except TypeError:
                model, _, _, results = retrain_func(
                    X=X, y=y, last_processed_file=last_processed_file,
                    logger=logger, output_dir=metrics_dir,
                    ipip_config=ipip_config, perf_thresholds=thresholds_perf,
                    block_col=block_col, drifted_blocks=affected_blocks,
                    model_instance=model_instance, random_state=random_state, window_size=window_size
                )
        else:
            model, _, _, results = default_retrain(
                X=X, y=y, last_processed_file=last_processed_file,
                model_path=model_path, mode=retrain_mode, random_state=random_state, logger=logger,
                output_dir=metrics_dir, window_size=window_size, block_col=block_col,
                drifted_blocks=affected_blocks, ipip_config=ipip_config,
                model_instance=model_instance, thresholds_perf=thresholds_perf
            )
        approved = bool((results or {}).get("approval", {}).get("approved", False))

    else:
        logger.info("No retraining needed. Skipping training step.")
        return

    # 5) Persistencia condicionada a aprobación
    if approved:
        _persist_model(model=model, pipeline_name=pipeline_name, output_dir=output_dir, logger=logger)
        _upsert_control_entry(Path(control_dir) / "control_file.txt", last_processed_file, last_mtime, logger)
    else:
        logger.warning("Modelo NO aprobado. Se mantiene el modelo previo y no se actualiza control_file.")
