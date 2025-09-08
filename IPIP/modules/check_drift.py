# pipeline/modules/check_drift.py
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any, Optional, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from Detector.drift_detector import DriftDetector


# =========================
# Utils (serialization & helpers)
# =========================

def _jsonable(obj):
    """Make Python/numpy objects JSON-serializable."""
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _save_results(payload: Dict[str, Any], path: str, logger) -> None:
    """Persist results to JSON with safe serialization."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=4, ensure_ascii=False)
        logger.info(f"[DRIFT] Results saved to {path}")
    except Exception as e:
        logger.error(f"[DRIFT] Failed to save results: {e}")


def _task_from_model(model) -> str:
    """Infer task type from model (classification/regression)."""
    gm = getattr(model, "global_model", model)
    if is_classifier(gm):
        return "classification"
    if is_regressor(gm):
        return "regression"
    # Fallback heuristic
    return "classification"


def _metrics_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard classification metrics (informativas, sin umbrales)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metrics (informativas, sin umbrales)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _relative_degradation(prev: float, curr: float, metric: str, task: str) -> float:
    """
    Degradación relativa (positivo = peor) de previous → current.
    Clasificación (↑ mejor):        (prev - curr) / max(|prev|, eps)
    Errores regresión (↓ mejor):    (curr - prev) / max(|prev|, eps)
    r2 (↑ mejor):                   (prev - curr) / max(|prev|, eps)
    """
    eps = 1e-9
    if task == "classification":
        return (prev - curr) / max(abs(prev), eps)
    else:
        if metric.lower() in ("r2", "r²", "r_2"):
            return (prev - curr) / max(abs(prev), eps)
        return (curr - prev) / max(abs(prev), eps)


def _safe_block_order(series: pd.Series) -> List[str]:
    """Keep dataset order for blocks (pd.unique) but stringify for keys."""
    return [str(x) for x in pd.unique(series).tolist()]


def _previous_model_path(model_path: str) -> str:
    root, ext = os.path.splitext(model_path)
    return f"{root}_previous{ext or '.pkl'}"


def _align_xy_for_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alinear dominios de etiqueta para métricas binarias si es necesario.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    def is_01(arr):
        u = set(np.unique(arr))
        return u <= {0, 1}

    def to_01(arr):
        vals, counts = np.unique(arr, return_counts=True)
        order = np.argsort(-counts)
        mapping = {vals[order[0]]: 0}
        if len(vals) > 1:
            mapping[vals[order[1]]] = 1
        return np.array([mapping.get(v, 1) for v in arr], dtype=int)

    if is_01(y_pred_arr) and not is_01(y_true_arr):
        return to_01(y_true_arr), y_pred_arr
    if is_01(y_true_arr) and not is_01(y_pred_arr):
        return y_true_arr, to_01(y_pred_arr)
    return y_true_arr, y_pred_arr


# =========================
# Main
# =========================

def check_drift(
    X: pd.DataFrame,
    y: pd.Series,
    logger,
    *,
    model_filename: str,
    output_dir: str,
    model_dir: str,
    block_col: Optional[str] = None,
    # Umbrales de DRIFT (solo estadísticos / degradación relativa)
    drift_thresholds: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Drift check (BLOCK MODE) usando EXCLUSIVAMENTE umbrales de drift:
      • Tests estadísticos por pares de bloques (i<j) con umbrales de drift_thresholds.
      • Si existe modelo ACTUAL:
            - Calcula métricas por bloque (informativas).
            - Si existe modelo PREVIO: compara CURRENT vs PREVIOUS por bloque/metric
              y marca degradación si rel_deg >= decay_ratio (de drift_thresholds).
      • Devuelve: 'train' (primera vez) | 'retrain' | 'no_drift' | 'end_error'

    Parámetros de drift admitidos en `drift_thresholds` (con valores por defecto):
      - alpha (float, 0.05)                : p-value para KS/MWU/CVM
      - psi_threshold (float, 0.10)        : umbral PSI
      - hellinger_threshold (float, 0.10)  : umbral Hellinger
      - emd_threshold (float|None)         : umbral EMD (si None, el detector puede auto-ajustar)
      - mmd_alpha (float|None)             : p-value MMD (si None usa alpha)
      - energy_alpha (float|None)          : p-value Energy (si None usa alpha)
      - decay_ratio (float, 0.30)          : degradación relativa para marcar drift de performance
    """
    # Valores por defecto
    dt = drift_thresholds or {}
    alpha              = float(dt.get("alpha", 0.05))
    psi_threshold      = float(dt.get("psi_threshold", 0.10))
    hellinger_threshold= float(dt.get("hellinger_threshold", 0.10))
    emd_threshold      = dt.get("emd_threshold", None)
    mmd_alpha          = float(dt.get("mmd_alpha", alpha)) if dt.get("mmd_alpha", None) is not None else alpha
    energy_alpha       = float(dt.get("energy_alpha", alpha)) if dt.get("energy_alpha", None) is not None else alpha
    decay_ratio        = float(dt.get("decay_ratio", 0.30))

    os.makedirs(output_dir, exist_ok=True)
    drift_path = os.path.join(output_dir, "drift_results.json")
    model_path = os.path.join(model_dir, model_filename)
    prev_model_path = _previous_model_path(model_path)

    results: Dict[str, Any] = {
        "decision": None,
        "drift": {
            "any_stat_drift": False,
            "any_perf_drift": False,
            "by_test": {},
        },
        "blockwise": {
            "block_col": None,
            "blocks": [],
            "pairs_mode": "all_pairs",
            "pairwise": {},                   # test_name -> { "bi|bj": {...} }
            "by_block_stat_drift": {},        # block -> bool (participa en algún drift estadístico)
            "drifted_blocks_stats": [],       # lista de bj de pares con drift
            "performance": {
                # Solo información; sin umbrales de performance
                "current":   {"per_block": {}},
                "previous":  {"per_block": {}},
                "comparison": {"decay_ratio": decay_ratio, "per_block": {}, "flags": {}},
                "drifted_blocks_perf": [],
            },
        },
        "promoted_model": False,
        "promotion_reason": None,
    }

    try:
        # --------- block column (requerida) ----------
        bc = block_col
        results["blockwise"]["block_col"] = bc
        if not bc or bc not in X.columns:
            logger.error(
                f"[DRIFT] Block column '{block_col}' not found in X. "
                f"Make sure preprocessing keeps it and the pipeline attaches it to X."
            )
            results["decision"] = "end_error"
            _save_results(results, drift_path, logger)
            return "end_error"

        blocks = _safe_block_order(X[bc])
        results["blockwise"]["blocks"] = blocks

        # --------- 1) Tests estadísticos (siempre) ----------
        det = DriftDetector()

        pair_tests = {
            "KS": {}, "MWU": {}, "CVM": {}, "PSI": {}, "Hellinger": {}, "EMD": {}, "MMD": {}, "Energy": {}
        }
        per_block_stat_flag = {b: False for b in blocks}
        drifted_blocks_stats_second = set()

        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                bi, bj = blocks[i], blocks[j]
                key = f"{bi}|{bj}"

                Xi = X[X[bc].astype(str) == bi].copy()
                Xj = X[X[bc].astype(str) == bj].copy()

                # Drop non-feature columns if present (block col & target si estuviera en X)
                if bc in Xi.columns: Xi.drop(columns=[bc], inplace=True)
                if bc in Xj.columns: Xj.drop(columns=[bc], inplace=True)
                tgt_name = getattr(y, "name", None)
                if tgt_name and tgt_name in Xi.columns: Xi.drop(columns=[tgt_name], inplace=True)
                if tgt_name and tgt_name in Xj.columns: Xj.drop(columns=[tgt_name], inplace=True)

                # Common numeric cols
                Xi_num = Xi.select_dtypes(include=[np.number]).copy()
                Xj_num = Xj.select_dtypes(include=[np.number]).copy()
                common_cols = Xi_num.columns.intersection(Xj_num.columns).tolist()
                if not common_cols or len(Xi_num) < 5 or len(Xj_num) < 5:
                    for K in pair_tests.keys():
                        pair_tests[K][key] = {"drift": False, "error": "no common numeric cols or not enough rows"}
                    continue
                XiN, XjN = Xi_num[common_cols], Xj_num[common_cols]

                # Univariantes
                ks_flag, ks_detail   = det.kolmogorov_smirnov_test(XiN, XjN, alpha=alpha)
                mwu_flag, mwu_detail = det.mann_whitney_test(XiN, XjN, alpha=alpha)
                cvm_flag, cvm_detail = det.cramervonmises_test(XiN, XjN, alpha=alpha)

                def pmin(detail):
                    if not isinstance(detail, dict): return None
                    vals = []
                    for v in detail.values():
                        pv = v.get("p_value") if isinstance(v, dict) else None
                        if pv is not None:
                            vals.append(float(pv))
                    return min(vals) if vals else None

                pair_tests["KS"][key]  = {"p_min": pmin(ks_detail),  "alpha": alpha, "drift": bool(ks_flag)}
                pair_tests["MWU"][key] = {"p_min": pmin(mwu_detail), "alpha": alpha, "drift": bool(mwu_flag)}
                pair_tests["CVM"][key] = {"p_min": pmin(cvm_detail), "alpha": alpha, "drift": bool(cvm_flag)}

                # PSI / Hellinger / EMD
                psi_flag, psi_detail = det.population_stability_index_test(XiN, XjN, psi_threshold=psi_threshold, num_bins=10)
                psi_max = 0.0
                if isinstance(psi_detail, dict):
                    for payload in psi_detail.values():
                        v = payload.get("psi")
                        if v is not None:
                            psi_max = max(psi_max, float(v))
                pair_tests["PSI"][key] = {"psi_max": psi_max, "threshold": psi_threshold, "drift": bool(psi_flag)}

                hell_flag, hell_detail = det.hellinger_distance_test(XiN, XjN, num_bins=30, threshold=hellinger_threshold)
                h_max = 0.0
                if isinstance(hell_detail, dict):
                    for payload in hell_detail.values():
                        v = payload.get("hellinger_distance")
                        if v is not None:
                            h_max = max(h_max, float(v))
                pair_tests["Hellinger"][key] = {"distance_max": h_max, "threshold": hellinger_threshold, "drift": bool(hell_flag)}

                emd_flag, emd_detail = det.earth_movers_distance_test(XiN, XjN, threshold=emd_threshold)
                e_max, thr_used = 0.0, None
                if isinstance(emd_detail, dict):
                    for payload in emd_detail.values():
                        v = payload.get("emd_distance")
                        if v is not None:
                            e_max = max(e_max, float(v))
                        thr_used = payload.get("threshold", thr_used)
                pair_tests["EMD"][key] = {"distance_max": e_max, "threshold": thr_used, "drift": bool(emd_flag)}

                # Multivariantes
              

                if any([bool(ks_flag), bool(mwu_flag), bool(cvm_flag), bool(psi_flag), bool(hell_flag),
                        bool(emd_flag)]):
                    per_block_stat_flag[bi] = True
                    per_block_stat_flag[bj] = True
                    drifted_blocks_stats_second.add(bj)

        by_test = {t: any(bool(v.get("drift", False)) for v in mat.values()) for t, mat in pair_tests.items()}
        results["blockwise"]["pairwise"] = pair_tests
        results["blockwise"]["by_block_stat_drift"] = per_block_stat_flag
        results["blockwise"]["drifted_blocks_stats"] = sorted(list(drifted_blocks_stats_second), key=lambda x: str(x))
        results["drift"]["by_test"] = by_test
        results["drift"]["any_stat_drift"] = any(by_test.values())

        # --------- 2) Performance por bloque (informativa) y comparación prev->curr ----------
        has_current = os.path.exists(model_path)
        if not has_current:
            logger.info("[DRIFT] No current model found → will return 'train' but statistical tests have been computed.")
            results["decision"] = "train"
            _save_results(results, drift_path, logger)
            return "train"

        # Cargar modelo actual
        try:
            model = joblib.load(model_path)
            logger.info("[DRIFT] Current model loaded.")
        except Exception as e:
            logger.error(f"[DRIFT] Could not load current model: {e}")
            results["decision"] = "end_error"
            _save_results(results, drift_path, logger)
            return "end_error"

        task = _task_from_model(model)

        def _eval_model_per_block(_model) -> Dict[str, Dict[str, float]]:
            per_metrics: Dict[str, Dict[str, float]] = {}
            for b in blocks:
                mask = (X[bc].astype(str) == b)
                Xb = X.loc[mask].copy()
                if bc in Xb.columns: Xb.drop(columns=[bc], inplace=True)
                tgt = y.name if hasattr(y, "name") else None
                if tgt and tgt in Xb.columns: Xb.drop(columns=[tgt], inplace=True)
                yb = y.loc[Xb.index]

                try:
                    yhat = getattr(_model, "predict")(Xb)
                    y_true_m, y_pred_m = _align_xy_for_metrics(yb.values, yhat)
                    vals = _metrics_classification(y_true_m, y_pred_m) if task == "classification" else _metrics_regression(y_true_m, y_pred_m)
                except Exception as e:
                    vals = {"error": str(e)}

                per_metrics[str(b)] = vals
            return per_metrics

        # Métricas actuales
        curr_metrics = _eval_model_per_block(model)
        results["blockwise"]["performance"]["current"]["per_block"] = curr_metrics

        # Métricas previas (si hay modelo previo) y comparación por degradación relativa
        prev_metrics: Dict[str, Dict[str, float]] = {}
        has_previous = os.path.exists(prev_model_path)

        if has_previous:
            try:
                prev_model = joblib.load(prev_model_path)
                prev_metrics = _eval_model_per_block(prev_model)
                results["blockwise"]["performance"]["previous"]["per_block"] = prev_metrics
            except Exception as e:
                has_previous = False
                logger.warning(f"[DRIFT] Could not load/evaluate previous model: {e}")

        comp_flags_total: Dict[str, Dict[str, bool]] = {}
        comp_detail_total: Dict[str, Dict[str, Dict[str, float]]] = {}
        perf_drift_blocks = set()

        if has_previous:
            for b in blocks:
                b = str(b)
                comp_flags_total[b] = {}
                comp_detail_total[b] = {}
                prev_b = prev_metrics.get(b, {})
                curr_b = curr_metrics.get(b, {})
                for m in set(prev_b.keys()) | set(curr_b.keys()):
                    if m not in prev_b or m not in curr_b:
                        continue
                    prev_val = float(prev_b[m]); curr_val = float(curr_b[m])
                    rel_deg = _relative_degradation(prev_val, curr_val, m, task)
                    flag = bool(rel_deg >= decay_ratio)
                    comp_flags_total[b][m] = flag
                    comp_detail_total[b][m] = {"prev": prev_val, "curr": curr_val, "rel_degradation": rel_deg}
                    if flag:
                        perf_drift_blocks.add(b)

            results["blockwise"]["performance"]["comparison"]["per_block"] = comp_detail_total
            results["blockwise"]["performance"]["comparison"]["flags"] = comp_flags_total

        results["blockwise"]["performance"]["drifted_blocks_perf"] = sorted(list(perf_drift_blocks), key=lambda x: str(x))

        # -------------------------
        # Decisión final
        # -------------------------
        results["drift"]["any_perf_drift"] = bool(len(perf_drift_blocks) > 0)
        drifted_union = set(results["blockwise"]["drifted_blocks_stats"]) | set(results["blockwise"]["performance"]["drifted_blocks_perf"])
        results["drifted_blocks"] = sorted(list(drifted_union), key=lambda x: str(x))

        decision = "retrain" if (results["drift"]["any_stat_drift"] or results["drift"]["any_perf_drift"]) else "no_drift"
        results["decision"] = decision
        _save_results(results, drift_path, logger)
        return decision

    except Exception as e:
        logger.error(f"[DRIFT] Unexpected error: {e}")
        results["decision"] = "end_error"
        _save_results(results, drift_path, logger)
        return "end_error"
