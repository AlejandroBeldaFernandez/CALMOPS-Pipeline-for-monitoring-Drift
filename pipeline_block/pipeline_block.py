# pipeline/modules/pipeline_block.py

import os
import time
import json
import logging
import importlib.util
from typing import Optional, List

import pandas as pd
import joblib

from logger.logger import PipelineLogger
from Detector.drift_detector import DriftDetector

# OJO: estos imports siguen tu estructura de carpetas indicada en el propio fichero
from .modules.data_loader import data_loader
from .modules.check_drift import check_drift
from .modules.default_train_retrain import default_train, default_retrain
from .modules.evaluador import evaluate_model


# -------------------------------  Circuit Breaker  -------------------------------

def _health_path(metrics_dir: str) -> str:
    return os.path.join(metrics_dir, "health.json")

def _load_health(metrics_dir: str) -> dict:
    p = _health_path(metrics_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None}

def _save_health(metrics_dir: str, data: dict) -> None:
    p = _health_path(metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
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

def _update_on_result(health: dict, approved: bool, backoff_minutes: int, max_failures: int) -> dict:
    if approved:
        health.update({"consecutive_failures": 0, "last_failure_ts": None, "paused_until": None})
    else:
        health["consecutive_failures"] = int(health.get("consecutive_failures", 0)) + 1
        health["last_failure_ts"] = time.time()
        if health["consecutive_failures"] >= max_failures:
            health["paused_until"] = time.time() + backoff_minutes * 60
    return health


# -------------------------------  Preprocess Loader  -------------------------------

def _load_preprocess_func(preprocess_file: str):
    """Carga data_preprocessing(df) desde un .py externo."""
    if not os.path.exists(preprocess_file):
        raise FileNotFoundError(f"Invalid preprocessing file: {preprocess_file}")
    spec = importlib.util.spec_from_file_location("custom_preprocess_module", preprocess_file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "data_preprocessing"):
        raise AttributeError(f"{preprocess_file} must define data_preprocessing(df)")
    return getattr(mod, "data_preprocessing")


# -------------------------------  Utils de bloques  -------------------------------

def _sorted_blocks(series: pd.Series) -> List[str]:
    vals = series.dropna().astype(str).unique().tolist()
    # numérico
    try:
        nums = [float(v) for v in vals]
        return [x for _, x in sorted(zip(nums, vals))]
    except Exception:
        pass
    # datetime
    try:
        dt = pd.to_datetime(vals, errors="raise")
        return [x for _, x in sorted(zip(dt, vals))]
    except Exception:
        pass
    # lexicográfico
    return sorted(vals, key=lambda x: str(x))


# -------------------------------  Main Pipeline (solo block_wise)  -------------------------------

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
    block_col: str | None = None,          # columna de bloque dentro del dataset (debe existir para block_wise)
    eval_blocks: list[str] | None = None,  # bloques a evaluar; si None => último bloque
) -> None:
    """
    Orquestación SOLO block_wise:
      1) Carga y preprocesa.
      2) Determina train_blocks (todos menos eval_blocks) y eval_blocks (por defecto, último).
      3) Si no hay modelo → TRAIN (global + por bloque) y eval en eval_blocks.
      4) Si hay modelo → check_drift (entre bloques) → si drift → RETRAIN solo bloques con drift; si no, NO-OP.
      5) Evalúa sobre eval_blocks y actualiza circuit breaker.
    """
    # Rutas
    base_dir    = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    output_dir  = os.path.join(base_dir, "modelos")
    control_dir = os.path.join(base_dir, "control")
    logs_dir    = os.path.join(base_dir, "logs")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(control_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{pipeline_name}.pkl")

    # Logger
    logger = PipelineLogger(pipeline_name, log_dir=logs_dir).get_logger()
    logging.basicConfig()
    logger.info("Pipeline started (block_wise).")

    # Circuit breaker
    health = _load_health(metrics_dir)
    if _should_pause(health):
        logger.warning("Retraining paused by circuit breaker. Skipping this run.")
        return

    # Validación thresholds: no mezclar clas/reg
    if {"accuracy", "f1", "balanced_accuracy"} & set(thresholds_perf.keys()) and \
       {"rmse", "mae", "mse", "r2"} & set(thresholds_perf.keys()):
        raise ValueError("Cannot define classification and regression thresholds simultaneously.")

    # Preprocess
    preprocess_func = _load_preprocess_func(preprocess_file)

    # 1) Carga completa (data_loader ya gestiona snapshots de control)
    df_full, last_processed_file, mtime = data_loader(
        logger, data_dir, control_dir, delimiter=delimiter, target_file=target_file, block_col=block_col
    )
    if df_full.empty:
        logger.warning("No new data to process.")
        return

    # 2) Preprocesado -> X, y  (block_col debe quedar en X para identificar bloques de eval)
    X, y = preprocess_func(df_full)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Preprocess must return X as a pandas DataFrame.")
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("Preprocess must return y as a pandas Series or single-column DataFrame.")
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a single target column.")
        y = y.iloc[:, 0]

    if (block_col is None) or (block_col not in X.columns):
        raise ValueError("block_col debe existir en X para operar en modo block_wise.")

    # Serie de bloques (como str)
    blocks_series = X[block_col].astype(str)

    # 2.b) Determinar eval_blocks (por defecto, último bloque) y train_blocks (resto)
    all_blocks_sorted = _sorted_blocks(blocks_series)
    if not all_blocks_sorted:
        logger.error("No se detectaron ids de bloque en los datos.")
        return

    if eval_blocks:
        eval_blocks = [str(b) for b in eval_blocks if str(b) in all_blocks_sorted]
        if not eval_blocks:
            eval_blocks = [all_blocks_sorted[-1]]
    else:
        eval_blocks = [all_blocks_sorted[-1]]

    train_blocks = [b for b in all_blocks_sorted if b not in set(eval_blocks)]
    logger.info(f"[BLOCKS] train_blocks={train_blocks} | eval_blocks={eval_blocks} | block_col={block_col}")

    # 3) check_drift (entre bloques). Pasa thresholds de PERFORMANCE (no thresholds_drift).
    decision = check_drift(
        X=X, y=y,
        logger=logger,
        perf_thresholds=thresholds_perf,     # <-- thresholds de performance
        model_filename=f"{pipeline_name}.pkl",
        output_dir=metrics_dir,
        model_dir=output_dir,
        block_col=block_col,
    )

    try:
        # === TRAIN (first time) ===
        if decision == "train" or not os.path.exists(model_path):
            logger.info("TRAIN phase (block_wise).")
            if custom_train_file:
                spec_t = importlib.util.spec_from_file_location("train_module", custom_train_file)
                mod_t = importlib.util.module_from_spec(spec_t); assert spec_t.loader is not None
                spec_t.loader.exec_module(mod_t)
                if not hasattr(mod_t, "train"):
                    raise AttributeError(f"{custom_train_file} must define train(...)")
                model, X_test, y_test, _ = mod_t.train(X, y, last_processed_file, logger, metrics_dir)
            else:
                model, X_test, y_test, _ = default_train(
                    X, y, last_processed_file,
                    model_instance=model_instance,
                    random_state=random_state,
                    logger=logger,
                    param_grid=param_grid,
                    cv=cv,
                    output_dir=metrics_dir,
                    # SOLO block_wise
                    blocks=blocks_series,
                    block_col=block_col,
                    test_blocks=eval_blocks,   # eval = último (o los que nos pasen)
                )

            # Reinyectar columna de bloque en X_test por seguridad
            if block_col not in X_test.columns:
                X_test = X_test.copy()
                X_test[block_col] = blocks_series.loc[X_test.index].astype(str)

            # Persist model
            try:
                joblib.dump(model, model_path)
                logger.info(f"Model saved: {model_path}")
            except Exception as e:
                logger.warning(f"Could not persist model to {model_path}: {e}")

            # Evaluate en eval_blocks
            approved = evaluate_model(
                model_or_path=model,
                X_eval=X_test,
                y_eval=y_test,
                logger=logger,
                metrics_dir=metrics_dir,
                control_dir=control_dir,
                data_file=last_processed_file,
                thresholds=thresholds_perf,
                block_col=block_col,
                evaluated_blocks=eval_blocks,
                include_predictions=True,
                max_pred_examples=100,
                mtime=mtime,
            )

            health = _update_on_result(health, bool(approved), breaker_backoff_minutes, breaker_max_failures)
            _save_health(metrics_dir, health)

        # === RETRAIN (solo bloques con drift) ===
        elif decision == "retrain":
            logger.info("RETRAIN phase (block_wise).")

            # Cargar qué bloques tienen drift del JSON de drift
            drift_json = os.path.join(metrics_dir, "drift_results.json")
            drifted_blocks: List[str] = []
            try:
                with open(drift_json, "r", encoding="utf-8") as f:
                    dr = json.load(f)
                bw = (dr.get("blockwise", {}) or {})
                # flags estadísticos por bloque (si el bloque aparece en alguna pareja con drift)
                stat_flags = (bw.get("by_block_stat_drift", {}) or {})
                # flags de performance por bloque (dict de métricas -> bool)
                perf_flags = ((bw.get("performance", {}) or {}).get("flags", {}) or {})

                stat_set = {str(b) for b, flag in stat_flags.items() if bool(flag)}
                perf_set = {
                    str(b) for b, mdict in perf_flags.items()
                    if isinstance(mdict, dict) and any(bool(v) for v in mdict.values())
                }
                drifted_blocks = sorted((stat_set | perf_set) & set(train_blocks))
            except Exception:
                logger.warning("No se pudo leer drift_results.json para decidir bloques con drift.")

            if drifted_blocks:
                logger.info(f"[RETRAIN] Bloques con drift a reentrenar: {drifted_blocks}")
            else:
                logger.info("[RETRAIN] No hay bloques con drift en train; se evaluará el campeón actual.")

            if custom_retrain_file and drifted_blocks:
                spec_r = importlib.util.spec_from_file_location("retrain_module", custom_retrain_file)
                mod_r = importlib.util.module_from_spec(spec_r); assert spec_r.loader is not None
                spec_r.loader.exec_module(mod_r)
                if not hasattr(mod_r, "retrain"):
                    raise AttributeError(f"{custom_retrain_file} must define retrain(...)")
                model, X_test, y_test, _ = mod_r.retrain(X, y, last_processed_file, logger, metrics_dir)
            elif drifted_blocks:
                model, X_test, y_test, _ = default_retrain(
                    X, y, last_processed_file,
                    model_path=model_path,
                    mode=retrain_mode,
                    random_state=random_state,
                    logger=logger,
                    output_dir=metrics_dir,
                    window_size=window_size,
                    blocks=blocks_series,
                    block_col=block_col,
                    test_blocks=eval_blocks,
                    drifted_blocks=drifted_blocks,  # <--- clave: solo estos bloques
                )
            else:
                # No hay drift en bloques de train → NO-OP; preparamos eval en eval_blocks
                model = joblib.load(model_path)
                mask_eval = blocks_series.isin(eval_blocks)
                X_test = X.loc[mask_eval].copy()
                y_test = y.loc[mask_eval]

            # Guardar (si hubo reentrenamiento)
            if isinstance(model, object):
                try:
                    joblib.dump(model, model_path)
                    logger.info(f"Model saved: {model_path}")
                except Exception as e:
                    logger.warning(f"Could not persist model to {model_path}: {e}")

            # Ensure block_col en X_test
            if block_col not in X_test.columns:
                X_test = X_test.copy()
                X_test[block_col] = blocks_series.loc[X_test.index].astype(str)

            # Evaluate
            approved = evaluate_model(
                model_or_path=model,
                X_eval=X_test,
                y_eval=y_test,
                logger=logger,
                metrics_dir=metrics_dir,
                control_dir=control_dir,
                data_file=last_processed_file,
                thresholds=thresholds_perf,
                block_col=block_col,
                evaluated_blocks=eval_blocks,
                include_predictions=True,
                max_pred_examples=100,
                mtime=mtime,
            )

            # Fallback si no aprueba
            if (not approved) and (fallback_mode is not None):
                logger.info(f"Fallback retrain with mode={fallback_mode}.")
                if custom_fallback_file:
                    spec_f = importlib.util.spec_from_file_location("fallback_module", custom_fallback_file)
                    mod_f = importlib.util.module_from_spec(spec_f); assert spec_f.loader is not None
                    spec_f.loader.exec_module(mod_f)
                    if not hasattr(mod_f, "fallback"):
                        raise AttributeError(f"{custom_fallback_file} must define fallback(...)")
                    model, X_test, y_test, _ = mod_f.fallback(X, y, last_processed_file, logger, metrics_dir)
                else:
                    model, X_test, y_test, _ = default_retrain(
                        X, y, last_processed_file,
                        model_path=model_path,
                        mode=fallback_mode,
                        random_state=random_state,
                        logger=logger,
                        output_dir=metrics_dir,
                        window_size=window_size,
                        blocks=blocks_series,
                        block_col=block_col,
                        test_blocks=eval_blocks,
                        drifted_blocks=drifted_blocks,  # mantenemos foco en bloques afectados
                    )

                if block_col not in X_test.columns:
                    X_test = X_test.copy()
                    X_test[block_col] = blocks_series.loc[X_test.index].astype(str)

                try:
                    joblib.dump(model, model_path)
                    logger.info(f"Fallback model saved: {model_path}")
                except Exception as e:
                    logger.warning(f"Could not persist fallback model to {model_path}: {e}")

                approved = evaluate_model(
                    model_or_path=model,
                    X_eval=X_test,
                    y_eval=y_test,
                    logger=logger,
                    metrics_dir=metrics_dir,
                    control_dir=control_dir,
                    data_file=last_processed_file,
                    thresholds=thresholds_perf,
                    block_col=block_col,
                    evaluated_blocks=eval_blocks,
                    include_predictions=True,
                    max_pred_examples=100,
                    mtime=mtime,
                )

            health = _update_on_result(health, bool(approved), breaker_backoff_minutes, breaker_max_failures)
            _save_health(metrics_dir, health)

        # === NO-OP (no retraining) → evaluar campeón en eval_blocks ===
        else:
            logger.info("No retraining required. Re-evaluating current champion on eval_blocks.")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Subconjunto de evaluación = eval_blocks
                mask_eval = blocks_series.isin(eval_blocks)
                X_test = X.loc[mask_eval].copy()
                y_test = y.loc[mask_eval]

                if block_col not in X_test.columns:
                    X_test[block_col] = blocks_series.loc[mask_eval].astype(str)

                approved = evaluate_model(
                    model_or_path=model,
                    X_eval=X_test,
                    y_eval=y_test,
                    logger=logger,
                    metrics_dir=metrics_dir,
                    control_dir=control_dir,
                    data_file=last_processed_file,
                    thresholds=thresholds_perf,
                    block_col=block_col,
                    evaluated_blocks=eval_blocks,
                    include_predictions=True,
                    max_pred_examples=100,
                    mtime=mtime,
                )

                health = _update_on_result(health, bool(approved), breaker_backoff_minutes, breaker_max_failures)
                _save_health(metrics_dir, health)
            else:
                logger.warning("Model file not found, nothing to evaluate.")

    except Exception as e:
        logger.error(f"[CRITICAL] Error in run_pipeline: {e}")
        raise
