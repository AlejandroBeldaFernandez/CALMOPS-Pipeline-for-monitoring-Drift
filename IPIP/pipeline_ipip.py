# pipeline/modules/pipeline_block_ipip.py
from __future__ import annotations

import os
import time
import json
import logging
import importlib.util
import inspect
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from logger.logger import PipelineLogger
from Detector.drift_detector import DriftDetector

# === Módulos IPIP (versión adaptada) ===
from .data_loader_ipip import data_loader
from .check_drift_ipip import check_drift
from .evaluador_ipip import evaluator
from .default_train_retrain_ipip import default_train, default_retrain


# -------------------------------  Circuit Breaker  -------------------------------

def _health_path(metrics_dir: str) -> str:
    return os.path.join(metrics_dir, "health.json")

def _load_health(metrics_dir: str):
    p = _health_path(metrics_dir)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None}

def _save_health(metrics_dir: str, data: dict):
    p = _health_path(metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)

def _should_pause(health: dict) -> bool:
    pu = health.get("paused_until")
    if pu is None:
        return False
    try:
        return time.time() < float(pu)
    except Exception:
        return False

def _update_on_result(health: dict, approved: bool, backoff_minutes: int, max_failures: int):
    if approved:
        health.update({"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None})
    else:
        health["consecutive_failures"] = int(health.get("consecutive_failures", 0)) + 1
        health["last_failure_ts"] = time.time()
        if health["consecutive_failures"] >= max_failures:
            health["paused_until"] = time.time() + backoff_minutes * 60
    return health


# -------------------------------  Utils  -------------------------------

def _detect_block_col(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    """Best-effort para detectar columna de bloque."""
    if explicit and explicit in df.columns:
        return explicit
    if "block_id" in df.columns:
        return "block_id"
    for c in df.columns:
        if "block" in c.lower() or c.lower() == "mes":
            return c
    return None

def _load_preprocess_func(preprocess_file: str):
    """Carga data_preprocessing(df) desde un .py externo."""
    if not os.path.exists(preprocess_file):
        raise FileNotFoundError(f"Please provide a valid preprocessing file: {preprocess_file}")
    spec = importlib.util.spec_from_file_location("custom_module", preprocess_file)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    if not hasattr(custom_module, "data_preprocessing"):
        raise AttributeError(f"{preprocess_file} must define data_preprocessing(df)")
    return getattr(custom_module, "data_preprocessing")

def _call_with_supported_args(fn, **kwargs):
    """Llama a una función pasando solo los kwargs que soporte (según su firma)."""
    sig = inspect.signature(fn)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**allowed)


# -------------------------------  Run Pipeline (GLOBAL POR DATASET)  -------------------------------

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
    breaker_max_failures: int = 3,
    breaker_backoff_minutes: int = 120,
    # reporting por bloque:
    block_col: str | None = None,
    # configuración específica de IPIP (opcional):
    ipip_config: dict | None = None,
):
    """
    Modo GLOBAL (usa TODOS los bloques disponibles):

      1) Carga todo el dataset (data_loader_ipip deja snapshot de bloques).
      2) Preprocesa con tu preprocessing_ipip (X numérico, y int).
      3) check_drift_ipip(X, y, ...) — admite blocks/block_col si están disponibles.
      4) TRAIN/RETRAIN con default_train/default_retrain IPIP (o ficheros custom).
      5) evaluator_ipip publica métricas globales y por bloque (test y/o full).
      6) Si aprueba, guarda previous_data.csv con TODO (features + target).

    Nota: `ipip_config` se reenvía a train/retrain si sus firmas lo aceptan.
    """
    # Rutas base (mantén nombres iguales a tu dashboard)
    base_dir = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    output_dir = os.path.join(base_dir, "modelos")
    control_dir = os.path.join(base_dir, "control")
    logs_dir = os.path.join(base_dir, "logs")
    metrics_dir = os.path.join(base_dir, "metrics")
    candidates_dir = os.path.join(base_dir, "candidates")
    for d in (output_dir, control_dir, logs_dir, metrics_dir, candidates_dir):
        os.makedirs(d, exist_ok=True)

    model_path = os.path.join(output_dir, f"{pipeline_name}.pkl")

    # Logger
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()
    logging.basicConfig()
    logger.info("Pipeline (IPIP) started — GLOBAL mode over all blocks")

    # Circuit breaker
    health = _load_health(metrics_dir)
    if _should_pause(health):
        logger.warning("⚠️ Retraining paused by circuit breaker. Skipping this run.")
        return

    # Validación de umbrales incompatibles
    if {"accuracy", "f1", "balanced_accuracy"} & set(thresholds_perf.keys()) and \
       {"rmse", "mae", "mse", "r2"} & set(thresholds_perf.keys()):
        raise ValueError("Cannot define both classification and regression thresholds at the same time.")

    # Preprocess
    preprocess_func = _load_preprocess_func(preprocess_file)

    # Detector
    detector = DriftDetector()

    # 1) Cargar TODO el dataset (y snapshot de bloques)
    df_full, last_processed_file, last_mtime = data_loader(
        logger, data_dir, control_dir, delimiter=delimiter, target_file=target_file, block_col=block_col
    )
    if df_full.empty:
        logger.warning("No new data to process.")
        return

    # 2) Preprocesar -> X_total, y_total
    X, y = preprocess_func(df_full)
    logger.info(f"Data preprocessed: {X.shape[0]} rows, {X.shape[1]} columns")
    df_xy = pd.concat([X, y], axis=1)  # features + target (para previous_data.csv si aprueba)

    # 3) Columna de bloque (reporting; no afecta decisión)
    blk_col = _detect_block_col(df_full, block_col)
    blocks_series = df_full[blk_col] if (blk_col and blk_col in df_full.columns) else None
    if blocks_series is not None:
        # Alinear índices (por si el preprocesado filtró/ordenó)
        try:
            blocks_series = blocks_series.loc[X.index]
        except Exception:
            common = X.index.intersection(blocks_series.index)
            blocks_series = blocks_series.loc[common]
            X = X.loc[common]; y = y.loc[common]
            df_xy = pd.concat([X, y], axis=1)

    # 4) check_drift con TODO el dataset
    decision = _call_with_supported_args(
        check_drift,
        X=X, y=y,
        detector=detector,
        logger=logger,
        perf_thresholds=thresholds_drift,
        model_filename=f"{pipeline_name}.pkl",
        data_dir=data_dir,
        output_dir=metrics_dir,
        control_dir=control_dir,
        model_dir=output_dir,
        blocks=blocks_series,
        block_col=blk_col,
    )

    try:
        # === TRAIN desde cero ===
        if decision == "train":
            logger.info("Starting TRAIN phase (IPIP) on ALL blocks")

            if custom_train_file:
                spec_t = importlib.util.spec_from_file_location("train_module", custom_train_file)
                mod_t = importlib.util.module_from_spec(spec_t); spec_t.loader.exec_module(mod_t)
                if not hasattr(mod_t, "train"):
                    raise AttributeError(f"{custom_train_file} must define train(...)")
                # Pasamos ipip_config si lo soporta el custom
                model, X_test, y_test, _ = _call_with_supported_args(
                    mod_t.train,
                    X=X, y=y,
                    last_processed_file=last_processed_file,
                    logger=logger, output_dir=metrics_dir,
                    ipip_config=ipip_config
                )
            else:
                model, X_test, y_test, _ = _call_with_supported_args(
                    default_train,
                    X=X, y=y,
                    last_processed_file=last_processed_file,
                    model_instance=model_instance,
                    random_state=random_state,
                    logger=logger,
                    param_grid=param_grid,
                    cv=cv,
                    output_dir=metrics_dir,
                    ipip_config=ipip_config
                )

            # test_blocks (si X_test es DataFrame y tenemos blk_col)
            test_blocks = None
            if isinstance(X_test, pd.DataFrame) and blocks_series is not None:
                try:
                    test_blocks = blocks_series.loc[X_test.index]
                except Exception:
                    test_blocks = None

            approved = _call_with_supported_args(
                evaluator,
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
                df=df_xy,
                output_dir=metrics_dir,
                # bloques (si el evaluador los soporta):
                block_col=blk_col,
                evaluated_block_id="ALL",
                test_blocks=test_blocks,
                reference_df=df_xy,
                reference_blocks=(sorted(map(str, blocks_series.unique())) if blocks_series is not None else None),
                X_all=X, y_all=y, all_blocks=blocks_series,
                candidates_dir=candidates_dir,
                save_candidates=True
            )

            health = _update_on_result(health, bool(approved), breaker_backoff_minutes, breaker_max_failures)
            _save_health(metrics_dir, health)

        # === RETRAIN sobre TODO el dataset ===
        elif decision == "retrain":
            logger.info("Starting RETRAIN phase (IPIP) on ALL blocks")

            if custom_retrain_file:
                spec_r = importlib.util.spec_from_file_location("retrain_module", custom_retrain_file)
                mod_r = importlib.util.module_from_spec(spec_r); spec_r.loader.exec_module(mod_r)
                if not hasattr(mod_r, "retrain"):
                    raise AttributeError(f"{custom_retrain_file} must define retrain(...)")
                model, X_test, y_test, _ = _call_with_supported_args(
                    mod_r.retrain,
                    X=X, y=y,
                    last_processed_file=last_processed_file,
                    logger=logger, output_dir=metrics_dir,
                    ipip_config=ipip_config
                )
            else:
                model, X_test, y_test, _ = _call_with_supported_args(
                    default_retrain,
                    X=X, y=y,
                    last_processed_file=last_processed_file,
                    model_path=model_path,
                    mode=retrain_mode,
                    random_state=random_state,
                    logger=logger,
                    param_grid=param_grid,
                    cv=cv,
                    output_dir=metrics_dir,
                    window_size=window_size,
                    ipip_config=ipip_config
                )

            test_blocks = None
            if isinstance(X_test, pd.DataFrame) and blocks_series is not None:
                try:
                    test_blocks = blocks_series.loc[X_test.index]
                except Exception:
                    test_blocks = None

            approved = _call_with_supported_args(
                evaluator,
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
                df=df_xy,
                output_dir=metrics_dir,
                block_col=blk_col,
                evaluated_block_id="ALL",
                test_blocks=test_blocks,
                reference_df=df_xy,
                reference_blocks=(sorted(map(str, blocks_series.unique())) if blocks_series is not None else None),
                X_all=X, y_all=y, all_blocks=blocks_series,
                candidates_dir=candidates_dir,
                save_candidates=True
            )

            # Fallback si no aprueba
            if not approved and fallback_mode is not None:
                logger.info(f"Attempting fallback (mode={fallback_mode}) on ALL blocks (IPIP)...")

                if custom_fallback_file:
                    spec_f = importlib.util.spec_from_file_location("fallback_module", custom_fallback_file)
                    mod_f = importlib.util.module_from_spec(spec_f); spec_f.loader.exec_module(mod_f)
                    if not hasattr(mod_f, "fallback"):
                        raise AttributeError(f"{custom_fallback_file} must define fallback(...)")
                    model, X_test, y_test, _ = _call_with_supported_args(
                        mod_f.fallback,
                        X=X, y=y,
                        last_processed_file=last_processed_file,
                        logger=logger, output_dir=metrics_dir,
                        ipip_config=ipip_config
                    )
                else:
                    model, X_test, y_test, _ = _call_with_supported_args(
                        default_retrain,
                        X=X, y=y,
                        last_processed_file=last_processed_file,
                        model_path=model_path,
                        mode=fallback_mode,
                        random_state=random_state,
                        logger=logger,
                        param_grid=param_grid,
                        cv=cv,
                        output_dir=metrics_dir,
                        window_size=window_size,
                        ipip_config=ipip_config
                    )

                test_blocks = None
                if isinstance(X_test, pd.DataFrame) and blocks_series is not None:
                    try:
                        test_blocks = blocks_series.loc[X_test.index]
                    except Exception:
                        test_blocks = None

                approved = _call_with_supported_args(
                    evaluator,
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
                    df=df_xy,
                    output_dir=metrics_dir,
                    block_col=blk_col,
                    evaluated_block_id="ALL",
                    test_blocks=test_blocks,
                    reference_df=df_xy,
                    reference_blocks=(sorted(map(str, blocks_series.unique())) if blocks_series is not None else None),
                    X_all=X, y_all=y, all_blocks=blocks_series,
                    candidates_dir=candidates_dir,
                    save_candidates=True
                )

            health = _update_on_result(health, bool(approved), breaker_backoff_minutes, breaker_max_failures)
            _save_health(metrics_dir, health)

        # === NO ACCIÓN (re-eval opcional del campeón) ===
        else:
            logger.info("Model already trained, no retraining needed (global, IPIP).")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=random_state
                )
                test_blocks = None
                if isinstance(X_test, pd.DataFrame) and blocks_series is not None:
                    try:
                        test_blocks = blocks_series.loc[X_test.index]
                    except Exception:
                        test_blocks = None

                approved = _call_with_supported_args(
                    evaluator,
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
                    df=df_xy,
                    output_dir=metrics_dir,
                    block_col=blk_col,
                    evaluated_block_id="ALL",
                    test_blocks=test_blocks,
                    reference_df=df_xy,
                    reference_blocks=(sorted(map(str, blocks_series.unique())) if blocks_series is not None else None),
                    X_all=X, y_all=y, all_blocks=blocks_series,
                    candidates_dir=None,
                    save_candidates=False
                )
                health = _update_on_result(health, bool(approved), breaker_backoff_minutes, breaker_max_failures)
                _save_health(metrics_dir, health)

    except Exception as e:
        logger.error(f"[CRITICAL] Critical error in run_pipeline (IPIP): {e}")
        raise
