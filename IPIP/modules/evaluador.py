# pipeline/modules/evaluador.py
# -*- coding: utf-8 -*-
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# =========================================
# Utilidades
# =========================================
def _jsonable(obj):
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, list):            return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):            return {k: _jsonable(v) for k, v in obj.items()}
    return obj

def _previous_model_path(model_path: str) -> str:
    root, ext = os.path.splitext(model_path)
    return f"{root}_previous{ext or '.pkl'}"

def _upsert_control_entry(control_file: Path, file_path: str, mtime, logger=None):
    """
    Crea/actualiza una línea '<basename>,<mtime>' en control_file.txt (atomic write).
    """
    control_file.parent.mkdir(parents=True, exist_ok=True)
    key_name = Path(file_path).name

    existing = {}
    if control_file.exists():
        with open(control_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                fname, raw_mtime = parts
                existing[fname] = raw_mtime

    existing[key_name] = str(mtime)

    tmp = control_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f"{k},{v}\n")
    os.replace(tmp, control_file)

    if logger:
        logger.info(f"[CONTROL] Upserted {key_name} with mtime={mtime} into {control_file.resolve()}")

def _align_label_domains(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alinea dominios: si una parte está en {0,1} y la otra no, mapea por frecuencia.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    def is_01(arr): return set(np.unique(arr)) <= {0, 1}
    def to_01(arr):
        vals, counts = np.unique(arr, return_counts=True)
        order = np.argsort(-counts)
        mapping = {vals[order[0]]: 0}
        if len(vals) > 1: mapping[vals[order[1]]] = 1
        return np.array([mapping.get(v, 1) for v in arr], dtype=int)

    if is_01(y_pred_arr) and not is_01(y_true_arr):
        return to_01(y_true_arr), y_pred_arr
    if is_01(y_true_arr) and not is_01(y_pred_arr):
        return y_true_arr, to_01(y_pred_arr)
    return y_true_arr, y_pred_arr

def _is_classification(model, y_test) -> bool:
    if hasattr(model, "predict_proba"):
        return True
    try:
        return pd.Series(y_test).nunique(dropna=True) <= 20
    except Exception:
        return True

def _metrics_from_preds(y_true, y_pred, is_classification: bool) -> dict:
    if is_classification:
        y_t, y_p = _align_label_domains(y_true, y_pred)
        return {
            "accuracy": accuracy_score(y_t, y_p),
            "balanced_accuracy": balanced_accuracy_score(y_t, y_p),
            "f1": f1_score(y_t, y_p, average="macro"),
        }
    else:
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }

def save_eval_results(results: dict, output_dir: str, logger=None):
    """Guarda <output_dir>/eval_results.json."""
    eval_path = os.path.join(output_dir, "eval_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=4, ensure_ascii=False)
    if logger:
        logger.info(f"[EVAL] Results saved at {Path(eval_path).resolve()}")

def _drop_technical_columns(X: pd.DataFrame, *, block_col: Optional[str], target_name: Optional[str]) -> pd.DataFrame:
    """Quita block_col y (por si acaso) la columna del target si estuviera en X."""
    Xc = X.copy()
    if block_col and block_col in Xc.columns:
        Xc = Xc.drop(columns=[block_col])
    if target_name and target_name in Xc.columns:
        Xc = Xc.drop(columns=[target_name])
    return Xc

def _persist_approved_model(
    model,
    *,
    model_dir: str,
    model_filename: str,
    control_dir: str,
    last_processed_file: str,
    last_mtime,
    logger
):
    """
    Guarda el modelo aprobado y actualiza control_file.
    - Backup del anterior a *_previous.pkl si existía.
    - Persistencia del campeón en model_dir/model_filename (joblib.dump).
    - Upsert en control_file.txt.
    """
    model_current_path = os.path.join(model_dir, model_filename)
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_current_path):
        previous_model = _previous_model_path(model_current_path)
        os.replace(model_current_path, previous_model)
        logger.info(f"[MODEL] Previous model backed up at {Path(previous_model).resolve()}")

    joblib.dump(model, model_current_path)
    logger.info(f"Model approved and saved at {Path(model_current_path).resolve()}")

    control_file = Path(control_dir) / "control_file.txt"
    _upsert_control_entry(control_file, last_processed_file, last_mtime, logger)

# =========================================
# API principal
# =========================================
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
    reference_df: pd.DataFrame = None,   # no usado en IPIP
    reference_blocks: list = None,
    # Candidates handling
    candidates_dir: str = None,
    save_candidates: bool = True,
):
    """
    Evaluación para IPIP:
      - Limpia columnas técnicas antes de predecir (block_col y target accidental en X_test).
      - Alinea dominios de etiquetas para métricas (ej: {"NO","SI"} vs {0,1}).
      - Calcula métricas globales y por bloque (si test_blocks o X_test[block_col]).
      - Guarda probabilidades en muestras si el modelo expone `predict_proba`.
      - Si pasa umbrales → guarda campeón y actualiza control_file.
      - No guarda previous_data.csv (IPIP no lo usa).
    """
    logger.info(">>> Starting model evaluation (IPIP)")

    # --- Preparación de datos de test ---
    X_is_df = isinstance(X_test, pd.DataFrame)
    target_name = getattr(y_test, "name", None) if hasattr(y_test, "name") else None

    # Si no nos pasan test_blocks pero X_test trae el bloque, lo usamos
    auto_blocks = None
    if test_blocks is None and X_is_df and block_col and block_col in X_test.columns:
        try:
            auto_blocks = X_test[block_col].copy()
            logger.info("[EVAL] Using block_col from X_test to compute per-block metrics.")
        except Exception:
            auto_blocks = None

    # Prepara X para predicción
    X_for_pred = _drop_technical_columns(X_test, block_col=block_col, target_name=target_name) if X_is_df else X_test

    # Tipo de tarea
    is_classification = _is_classification(model, y_test)

    approved = True
    results: Dict[str, Any] = {}
    sample_predictions: list = []

    try:
        # --- Predicción ---
        y_pred = model.predict(X_for_pred)

        # Probabilidades (si existen)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_for_pred)
                if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    y_proba = proba[:, 1]
                else:
                    y_proba = np.ravel(proba)
            except Exception:
                y_proba = None

        # --- Métricas globales ---
        if is_classification:
            y_true_al, y_pred_al = _align_label_domains(y_test, y_pred)
            metrics = {
                "accuracy": accuracy_score(y_true_al, y_pred_al),
                "balanced_accuracy": balanced_accuracy_score(y_true_al, y_pred_al),
                "f1": f1_score(y_true_al, y_pred_al, average="macro"),
                "classification_report": classification_report(y_true_al, y_pred_al, output_dict=True),
            }
        else:
            metrics = {
                "r2": r2_score(y_test, y_pred),
                "rmse": mean_squared_error(y_test, y_pred, squared=False),
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
            }

        # --- Check de umbrales ---
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

        # --- Muestras (con proba si hay) ---
        n_show = min(10, len(y_pred))
        if n_show > 0:
            df_sample = pd.DataFrame({
                "y_true": np.asarray(y_test)[:n_show],
                "y_pred": np.asarray(y_pred)[:n_show],
            })
            if y_proba is not None:
                df_sample["p1"] = np.asarray(y_proba)[:n_show]
            sample_predictions = df_sample.to_dict(orient="records")

        # --- Métricas por bloque ---
        per_block_metrics = {}
        blocks_used = test_blocks if test_blocks is not None else auto_blocks
        if blocks_used is not None:
            try:
                if len(blocks_used) != len(y_test):
                    logger.warning("test_blocks length mismatch with y_test; skipping per-block metrics.")
                else:
                    tb = pd.Series(blocks_used).reset_index(drop=True)
                    y_true_s = pd.Series(y_test).reset_index(drop=True)
                    y_pred_s = pd.Series(y_pred).reset_index(drop=True)
                    for bid, idxs in tb.groupby(tb).groups.items():
                        y_t = y_true_s.loc[idxs].values
                        y_p = y_pred_s.loc[idxs].values
                        per_block_metrics[str(bid)] = _metrics_from_preds(y_t, y_p, is_classification)
            except Exception as e:
                logger.warning(f"Per-block metrics computation failed: {e}")

        # --- Payload de resultados ---
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
            "model_meta": getattr(model, "meta_", None),  # útil en IPIP
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
            "model_meta": getattr(model, "meta_", None),
        }

    # --- Persistencia del informe ---
    save_eval_results(results, output_dir, logger=logger)

    # --- Publicación del campeón + control file (SI y SOLO SI aprobado) ---
    if approved:
        try:
            _persist_approved_model(
                model,
                model_dir=model_dir,
                model_filename=model_filename,
                control_dir=control_dir,
                last_processed_file=last_processed_file,
                last_mtime=last_mtime,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error saving model/control: {e}")
    else:
        logger.warning("Model did not pass thresholds. Champion model not updated.")
        if save_candidates and candidates_dir:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cand_root = os.path.join(candidates_dir, f"{Path(model_filename).stem}__{ts}")
                os.makedirs(cand_root, exist_ok=True)
                joblib.dump(model, os.path.join(cand_root, "model.pkl"))
                with open(os.path.join(cand_root, "eval_results.json"), "w", encoding="utf-8") as f:
                    json.dump(_jsonable(results), f, indent=4, ensure_ascii=False)
                logger.info(f"[CANDIDATE] Saved non-approved model at {Path(cand_root).resolve()}")
            except Exception as e:
                logger.error(f"[CANDIDATE] Error saving candidate: {e}")

    return approved
