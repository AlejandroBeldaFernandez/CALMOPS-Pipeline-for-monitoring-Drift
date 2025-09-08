# pipeline/modules/default_train_retrain.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import joblib
from pathlib import Path


def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.ndarray, list, tuple)): return [_jsonable(x) for x in o]
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    return o


# -------------------------
# Métricas helper
# -------------------------
def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "BA": float(balanced_accuracy_score(y_true, y_pred)),
        "F1m": float(f1_score(y_true, y_pred, average="macro"))
    }


# -------------------------
# Política desde thresholds_perf
# -------------------------
def _resolve_policy(thresholds_perf: Optional[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Extrae (metric_name, min_threshold) desde thresholds_perf.
    Preferencia: BA/BAL_ACC/balanced_accuracy -> F1/F1m -> ACC/accuracy.
    Acepta formatos:
      {"BA":{"min":0.9}}, {"balanced_accuracy":0.9}, {"ACC":0.95}, etc.
    """
    if not thresholds_perf:
        return "BA", 0.90

    def _get(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in thresholds_perf:
                v = thresholds_perf[k]
                if isinstance(v, dict) and "min" in v:
                    return float(v["min"])
                if isinstance(v, (int, float)):
                    return float(v)
        return None

    # BA / balanced_accuracy
    v = _get(["BA", "BAL_ACC", "balanced_accuracy"])
    if v is not None:
        return "BA", v
    # F1
    v = _get(["F1m", "F1", "f1", "f1_macro"])
    if v is not None:
        return "F1m", v
    # ACC
    v = _get(["ACC", "accuracy"])
    if v is not None:
        return "ACC", v

    return "BA", 0.90


# -------------------------
# Utilidades de bloques/modelo
# -------------------------
def _blocks_in_order(X: pd.DataFrame, block_col: str) -> List[str]:
    return [str(x) for x in pd.unique(X[block_col]).tolist()]


def _save_results(path: Path, payload: Dict[str, Any], logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=4, ensure_ascii=False)
    logger.info(f"[TRAIN] Results saved at {path.resolve()}")


def _instantiate_like(prev_model, random_state: int):
    """Crea un nuevo estimador del mismo tipo y con mismos parámetros que prev_model."""
    params = prev_model.get_params(deep=True) if hasattr(prev_model, "get_params") else {}
    m = prev_model.__class__(**params)
    if hasattr(m, "random_state"):
        setattr(m, "random_state", random_state)
    return m


# =========================================================
# TRAIN
# =========================================================
def default_train(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    last_processed_file: str,
    model_instance,
    random_state: int,
    logger,
    output_dir: str,
    block_col: str,
    ipip_config: Optional[Dict[str, Any]] = None,
    thresholds_perf: Optional[Dict[str, Any]] = None,
):
    """
    Entrenamiento estilo IPIP simplificado:
      - Para cada transición de bloque t->t+1 entrena un clon del modelo en el bloque t
        y evalúa en t+1 (logging por transición).
      - Modelo final: se re-entrena en TODOS los datos (sin columna de bloque).
      - Aprobación: media de la métrica elegida frente a thresholds_perf.
    """
    if model_instance is None:
        raise ValueError("[TRAIN] 'model_instance' es obligatorio en entrenamiento inicial.")

    t0 = time.time()
    X = X.copy()
    assert block_col in X.columns, f"'{block_col}' must be in X"
    blocks = _blocks_in_order(X, block_col)

    # Para entrenamiento, quitamos la col de bloque
    X_feat = X.drop(columns=[block_col], errors="ignore")

    per_transitions: List[Dict[str, Any]] = []
    # Entrenamiento por transición (t -> t+1)
    for i in range(len(blocks) - 1):
        bi, bj = blocks[i], blocks[i + 1]
        tic = time.time()

        tr_mask = (X[block_col].astype(str) == bi)
        te_mask = (X[block_col].astype(str) == bj)

        Xtr, ytr = X_feat.loc[tr_mask], y.loc[tr_mask]
        Xte, yte = X_feat.loc[te_mask], y.loc[te_mask]

        # Entrenar un clon del modelo base
        mdl = clone(model_instance)
        if hasattr(mdl, "random_state"):
            mdl.random_state = getattr(mdl, "random_state", random_state)
        mdl.fit(Xtr, ytr)

        yhat = mdl.predict(Xte)
        m = _compute_metrics(yte, yhat)

        elapsed = time.time() - tic
        logger.info(f"[TRAIN][{bi}→{bj}] BA={m['BA']:.4f} ACC={m['ACC']:.4f} F1m={m['F1m']:.4f} | {elapsed:.1f}s")
        per_transitions.append({"from": bi, "to": bj, "metrics": m, "elapsed_s": elapsed})

    # Modelo final: todos los datos
    final_model = clone(model_instance)
    if hasattr(final_model, "random_state"):
        final_model.random_state = getattr(final_model, "random_state", random_state)
    final_model.fit(X_feat, y)

    # Agregados globales (media simple de métricas por transición)
    if per_transitions:
        BA_mean = float(np.mean([r["metrics"]["BA"] for r in per_transitions]))
        ACC_mean = float(np.mean([r["metrics"]["ACC"] for r in per_transitions]))
        F1m_mean = float(np.mean([r["metrics"]["F1m"] for r in per_transitions]))
    else:
        BA_mean = ACC_mean = F1m_mean = 0.0

    metric_name, thr = _resolve_policy(thresholds_perf)
    metric_value = {"BA": BA_mean, "ACC": ACC_mean, "F1m": F1m_mean}[metric_name]
    approved = bool(metric_value >= thr)

    results = {
        "mode": "train",
        "file": last_processed_file,
        "blocks": blocks,
        "transitions": per_transitions,
        "global": {
            "means": {"BA": BA_mean, "ACC": ACC_mean, "F1m": F1m_mean},
            "metric_used": metric_name,
            "threshold": thr,
            "metric_value": metric_value
        },
        "approval": {"approved": approved}
    }

    _save_results(Path(output_dir) / "train_results.json", results, logger)
    logger.info(f"[TRAIN] Completed in {time.time() - t0:.1f}s — Approved={approved}")
    return final_model, None, None, results


# =========================================================
# RETRAIN
# =========================================================
def default_retrain(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    last_processed_file: str,
    model_path: str,
    mode: int,
    random_state: int,
    logger,
    output_dir: str,
    window_size: Optional[int],
    block_col: str,
    drifted_blocks: Optional[List[str]] = None,
    ipip_config: Optional[Dict[str, Any]] = None,
    model_instance=None,
    thresholds_perf: Optional[Dict[str, Any]] = None,
):
    """
    Retrain:
      - Si hay `drifted_blocks`, usa solo esos bloques.
      - Si no, y hay `window_size`, usa los últimos `window_size` bloques.
      - Si no, usa todos.
      - Aprobación por media de métricas vs thresholds_perf.
      - Si model_instance es None, se reconstruye el estimador a partir del modelo previo en disco.
    """
    t0 = time.time()
    X = X.copy()
    assert block_col in X.columns, f"'{block_col}' must be in X"
    all_blocks = _blocks_in_order(X, block_col)

    # Selección de bloques para retrain
    if drifted_blocks:
        sel_blocks = [b for b in all_blocks if str(b) in set(map(str, drifted_blocks))]
        if not sel_blocks:
            sel_blocks = all_blocks
    elif window_size and window_size > 0:
        sel_blocks = all_blocks[-int(window_size):]
    else:
        sel_blocks = all_blocks

    sel_mask = X[block_col].astype(str).isin(sel_blocks)
    X_sel = X.loc[sel_mask].copy()
    y_sel = y.loc[X_sel.index]
    blocks = _blocks_in_order(X_sel, block_col)

    X_feat = X_sel.drop(columns=[block_col], errors="ignore")

    # Resolver modelo base para entrenar (transiciones y final)
    base_model = None
    if model_instance is not None:
        base_model = clone(model_instance)
        if hasattr(base_model, "random_state"):
            base_model.random_state = getattr(base_model, "random_state", random_state)
    else:
        # reconstruir desde el modelo previo en disco
        try:
            prev = joblib.load(model_path)
            base_model = _instantiate_like(prev, random_state)
            logger.info("[RETRAIN] model_instance=None → reconstruido a partir del modelo previo en disco.")
        except Exception as e:
            raise ValueError(f"[RETRAIN] No se pudo reconstruir el estimador desde '{model_path}': {e}")

    per_transitions: List[Dict[str, Any]] = []
    for i in range(len(blocks) - 1):
        bi, bj = blocks[i], blocks[i + 1]
        tic = time.time()

        tr_mask = (X_sel[block_col].astype(str) == bi)
        te_mask = (X_sel[block_col].astype(str) == bj)

        Xtr, ytr = X_feat.loc[tr_mask], y_sel.loc[tr_mask]
        Xte, yte = X_feat.loc[te_mask], y_sel.loc[te_mask]

        mdl = clone(base_model)
        mdl.fit(Xtr, ytr)

        yhat = mdl.predict(Xte)
        m = _compute_metrics(yte, yhat)

        elapsed = time.time() - tic
        logger.info(f"[RETRAIN][{bi}→{bj}] BA={m['BA']:.4f} ACC={m['ACC']:.4f} F1m={m['F1m']:.4f} | {elapsed:.1f}s")
        per_transitions.append({"from": bi, "to": bj, "metrics": m, "elapsed_s": elapsed})

    # Modelo final re-entrenado con todo el subset
    final_model = clone(base_model)
    final_model.fit(X_feat, y_sel)

    if per_transitions:
        BA_mean = float(np.mean([r["metrics"]["BA"] for r in per_transitions]))
        ACC_mean = float(np.mean([r["metrics"]["ACC"] for r in per_transitions]))
        F1m_mean = float(np.mean([r["metrics"]["F1m"] for r in per_transitions]))
    else:
        BA_mean = ACC_mean = F1m_mean = 0.0

    metric_name, thr = _resolve_policy(thresholds_perf)
    metric_value = {"BA": BA_mean, "ACC": ACC_mean, "F1m": F1m_mean}[metric_name]
    approved = bool(metric_value >= thr)

    results = {
        "mode": "retrain",
        "file": last_processed_file,
        "used_blocks": blocks,
        "transitions": per_transitions,
        "global": {
            "means": {"BA": BA_mean, "ACC": ACC_mean, "F1m": F1m_mean},
            "metric_used": metric_name,
            "threshold": thr,
            "metric_value": metric_value
        },
        "approval": {"approved": approved}
    }

    _save_results(Path(output_dir) / "retrain_results.json", results, logger)
    logger.info(f"[RETRAIN] Completed in {time.time() - t0:.1f}s — Approved={approved}")
    return final_model, None, None, results
