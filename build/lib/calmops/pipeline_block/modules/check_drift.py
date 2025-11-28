# pipeline/modules/check_drift.py
# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from ...Detector.drift_detector import DriftDetector


# =========================
# Utils (serialization & helpers)
# =========================

def _jsonable(obj):
    """Make Python/numpy objects JSON-serializable."""
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _save_results(payload: Dict[str, Any], path: str, logger) -> None:
    """Persist results to JSON with safe serialization."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=4)
        logger.info(f"[DRIFT] Results saved to {path}")
    except Exception as e:
        logger.error(f"[DRIFT] Failed to save results: {e}")


def _detect_block_col(df: pd.DataFrame, cand: Optional[str]) -> Optional[str]:
    """Detect the block column from a candidate or common names."""
    if cand and cand in df.columns:
        return cand
    for c in df.columns:
        lc = str(c).lower()
        if "block" in lc or lc in ("block_id", "chunk"):
            return c
    return None


def _task_from_model(model) -> str:
    """Infer task type from model (classification/regression)."""
    gm = getattr(model, "global_model", model)
    if is_classifier(gm):
        return "classification"
    if is_regressor(gm):
        return "regression"
    # fallback heuristic
    return "classification"


def _metrics_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _relative_degradation(prev: float, curr: float, metric: str, task: str) -> float:
    """
    Relative degradation (positive means worse) from previous → current.
    For classification (↑ is better):    (prev - curr) / max(|prev|, eps)
    For regression errors (↓ is better): (curr - prev) / max(|prev|, eps)
    For r2 (↑ is better):                (prev - curr) / max(|prev|, eps)
    """
    eps = 1e-9
    if task == "classification":
        # higher is better (accuracy, balanced_accuracy, f1)
        return (prev - curr) / max(abs(prev), eps)
    else:
        if metric.lower() in ("r2", "r²", "r_2"):
            return (prev - curr) / max(abs(prev), eps)
        # error metrics: lower is better
        return (curr - prev) / max(abs(prev), eps)


# =========================
# Main
# =========================

def check_drift(
    X: pd.DataFrame,
    y: pd.Series,
    logger,
    perf_thresholds: dict,
    model_filename: str,
    output_dir: str,
    model_dir: str,
    data_dir: str,
    control_dir: str,
    current_filename: str,
    *,
    prediction_only: bool = False,
    block_col: Optional[str] = None,
    # thresholds for data-drift tests (optional)
    alpha: float = 0.05,
    psi_threshold: float = 0.10,
    hellinger_threshold: float = 0.10,
    emd_threshold: Optional[float] = None,
    mmd_alpha: Optional[float] = None,
    energy_alpha: Optional[float] = None,
    # previous vs current comparison
    decay_ratio: float = 0.30,
) -> str:
    """
    Drift check (BLOCK MODE):
      • Always runs pairwise statistical tests across blocks (even on the very first run).
      • Runs per-block performance tests only if a CURRENT model exists.
      • If PREVIOUS exists, compares CURRENT vs PREVIOUS by block+metric.
      • Optional automatic promotion of PREVIOUS (same policy as before).
    Returns: 'train' | 'retrain' | 'no_drift' | 'end_error'
    and persists metrics/drift_results.json for the dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)
    drift_path = os.path.join(output_dir, "drift_results.json")
    model_path = os.path.join(model_dir, model_filename)
    prev_model_path = model_path.replace(".pkl", "_previous.pkl")

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
            "pairwise": {},
            "by_block_stat_drift": {},
            "performance": {
                "thresholds": perf_thresholds or {},
                "current":   {"per_block": {}, "flags": {}},
                "previous":  {"per_block": {}, "flags": {}},
                "comparison": {"decay_ratio": decay_ratio, "per_block": {}, "flags": {}},
            },
        },
        "promoted_model": False,
        "promotion_reason": None,
    }

    if prediction_only:
        logger.info("[DRIFT] Prediction-only mode enabled. Skipping drift detection.")
        return "predict"

    if y is None:
        logger.error("[DRIFT] Target variable 'y' is None, but prediction_only is False. Cannot perform drift detection.")
        results["decision"] = "end_error"
        _save_results(results, drift_path, logger)
        return "end_error"

    try:
        # --------- block discovery ----------
        bc = _detect_block_col(X, block_col)
        results["blockwise"]["block_col"] = bc
        if not bc:
            logger.error("[DRIFT] No block column detected. Aborting drift analysis.")
            results["decision"] = "end_error"
            _save_results(results, drift_path, logger)
            return "end_error"

        blocks = pd.unique(X[bc].astype(str)).tolist()
        results["blockwise"]["blocks"] = blocks

        # --------- 1) Statistical tests (ALWAYS) ----------
        det = DriftDetector()

        pair_tests = {
            "KS": {}, "MWU": {}, "PSI": {}
        }
        per_block_stat_flag = {b: False for b in blocks}

        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                bi, bj = blocks[i], blocks[j]
                key = f"{bi}|{bj}"

                Xi = X[X[bc].astype(str) == bi]
                Xj = X[X[bc].astype(str) == bj]
                # --- build numeric, aligned matrices for Xi and Xj ---
                Xi_num = Xi.select_dtypes(include=[np.number]).copy()
                Xj_num = Xj.select_dtypes(include=[np.number]).copy()

                # columns present in BOTH blocks
                common_cols = Xi_num.columns.intersection(Xj_num.columns).tolist()
                if not common_cols:
                    for K in pair_tests.keys():
                        pair_tests[K][key] = {
                            "drift": False,
                            "error": "no common numeric cols",
                        }
                    continue

                XiN = Xi_num[common_cols]
                XjN = Xj_num[common_cols]

                # optional: skip if too few rows
                if len(XiN) < 5 or len(XjN) < 5:
                    for K in pair_tests.keys():
                        pair_tests[K][key] = {
                            "drift": False,
                            "error": "not enough rows in numeric cols",
                        }
                    continue


                # Univariate tests
                ks_flag, ks_detail = det.kolmogorov_smirnov_test(XiN, XjN, alpha=alpha)
                mwu_flag, mwu_detail = det.mann_whitney_test(XiN, XjN, alpha=alpha)

                pmin = lambda detail: None if not isinstance(detail, dict) else \
                    (min([float(v.get("p_value")) for v in detail.values() if v.get("p_value") is not None], default=None))

                pair_tests["KS"][key]  = {"p_min": pmin(ks_detail),  "alpha": alpha, "drift": bool(ks_flag)}
                pair_tests["MWU"][key] = {"p_min": pmin(mwu_detail), "alpha": alpha, "drift": bool(mwu_flag)}

                psi_flag, psi_detail = det.population_stability_index_test(XiN, XjN, psi_threshold=psi_threshold, num_bins=10)
                psi_max = 0.0
                if isinstance(psi_detail, dict):
                    for payload in psi_detail.values():
                        v = payload.get("psi")
                        if v is not None:
                            psi_max = max(psi_max, float(v))
                pair_tests["PSI"][key] = {"psi_max": psi_max, "threshold": psi_threshold, "drift": bool(psi_flag)}

  

                if any([ks_flag, mwu_flag, psi_flag]):
                    per_block_stat_flag[bi] = True
                    per_block_stat_flag[bj] = True

        by_test = {t: any(bool(v.get("drift", False)) for v in mat.values()) for t, mat in pair_tests.items()}
        results["blockwise"]["pairwise"] = pair_tests
        results["blockwise"]["by_block_stat_drift"] = per_block_stat_flag
        results["drift"]["by_test"] = by_test
        results["drift"]["any_stat_drift"] = any(by_test.values())

        # --------- 2) Performance by block (ONLY if current model exists) ----------
        has_current = os.path.exists(model_path)
        if not has_current:
            logger.info("[DRIFT] No current model found → will return 'train' but statistical tests have been computed.")
            results["decision"] = "train"
            _save_results(results, drift_path, logger)
            return "train"

        # load current
        try:
            model = joblib.load(model_path)
            logger.info("[DRIFT] Current model loaded.")
        except Exception as e:
            logger.error(f"[DRIFT] Could not load current model: {e}")
            results["decision"] = "end_error"
            _save_results(results, drift_path, logger)
            return "end_error"

        task = _task_from_model(model)

        def _eval_model_per_block(_model) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, bool]]]:
            per_metrics: Dict[str, Dict[str, float]] = {}
            per_flags: Dict[str, Dict[str, bool]] = {}
            for b in blocks:
                mask = (X[bc].astype(str) == b)
                Xb = X.loc[mask]
                yb = y.loc[Xb.index]
                flags_b, vals_b = {}, {}
                try:
                    yhat = getattr(_model, "predict")(Xb)
                    if task == "classification":
                        vals_b = _metrics_classification(yb.values, yhat)
                        if "accuracy" in perf_thresholds:
                            flags_b["accuracy"] = bool(vals_b["accuracy"] < float(perf_thresholds["accuracy"]))
                        if "balanced_accuracy" in perf_thresholds:
                            flags_b["balanced_accuracy"] = bool(vals_b["balanced_accuracy"] < float(perf_thresholds["balanced_accuracy"]))
                        if "f1" in perf_thresholds:
                            flags_b["f1"] = bool(vals_b["f1"] < float(perf_thresholds["f1"]))
                    else:
                        vals_b = _metrics_regression(yb.values, yhat)
                        if "rmse" in perf_thresholds:
                            flags_b["rmse"] = bool(vals_b["rmse"] > float(perf_thresholds["rmse"]))
                        if "mae" in perf_thresholds:
                            flags_b["mae"] = bool(vals_b["mae"] > float(perf_thresholds["mae"]))
                        if "mse" in perf_thresholds:
                            flags_b["mse"] = bool(vals_b["mse"] > float(perf_thresholds["mse"]))
                        if "r2" in perf_thresholds:
                            flags_b["r2"] = bool(vals_b["r2"] < float(perf_thresholds["r2"]))
                except Exception:
                    # model might expect a specific feature order/pipeline; keep empty
                    pass
                per_metrics[str(b)] = vals_b
                per_flags[str(b)] = flags_b
            return per_metrics, per_flags

        # current perf
        curr_metrics, curr_flags = _eval_model_per_block(model)
        results["blockwise"]["performance"]["current"]["per_block"] = curr_metrics
        results["blockwise"]["performance"]["current"]["flags"] = curr_flags

        # previous perf (if exists)
        prev_metrics: Dict[str, Dict[str, float]] = {}
        prev_flags: Dict[str, Dict[str, bool]] = {}
        has_previous = os.path.exists(prev_model_path)
        previous_passes = False

        if has_previous:
            try:
                prev_model = joblib.load(prev_model_path)
                prev_metrics, prev_flags = _eval_model_per_block(prev_model)
                results["blockwise"]["performance"]["previous"]["per_block"] = prev_metrics
                results["blockwise"]["performance"]["previous"]["flags"] = prev_flags
                previous_passes = (sum(bool(v) for b in prev_flags.values() for v in b.values()) == 0)
            except Exception as e:
                logger.error(f"[DRIFT] Could not load/evaluate previous model: {e}")
                has_previous = False

        # comparison current vs previous
        comp_flags_total: Dict[str, Dict[str, bool]] = {}
        comp_detail_total: Dict[str, Dict[str, Dict[str, float]]] = {}
        comp_majority = False

        if has_previous:
            worse_flags: List[bool] = []
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
                    worse = bool(rel_deg >= decay_ratio)
                    comp_flags_total[b][m] = worse
                    comp_detail_total[b][m] = {"prev": prev_val, "curr": curr_val, "rel_degradation": rel_deg}
                    worse_flags.append(worse)

            results["blockwise"]["performance"]["comparison"]["per_block"] = comp_detail_total
            results["blockwise"]["performance"]["comparison"]["flags"] = comp_flags_total

            if worse_flags:
                comp_majority = (sum(worse_flags) >= max(1, int(np.ceil(len(worse_flags) / 2))))

        # aggregate perf flags (current)
        all_current_flags = [bool(v) for b in curr_flags.values() for v in b.values()]
        any_perf_flag = (sum(all_current_flags) >= max(1, int(np.ceil(len(all_current_flags) / 2)))) if all_current_flags else False
        results["drift"]["any_perf_drift"] = bool(any_perf_flag)

        # optional promotion policy (unchanged)
        if has_previous and (previous_passes or comp_majority):
            reason = "previous_passed_thresholds" if previous_passes else "current_degraded_vs_previous_30pct"
            try:
                tmp_swap = model_path + ".swap"
                os.replace(model_path, tmp_swap)
                os.replace(prev_model_path, model_path)
                os.replace(tmp_swap, prev_model_path)
                logger.info(f"[DRIFT] Previous model promoted to CURRENT (reason: {reason}).")
                results["promoted_model"] = True
                results["promotion_reason"] = reason
                results["decision"] = "no_drift"
                _save_results(results, drift_path, logger)
                return "no_drift"
            except Exception as e:
                logger.error(f"[DRIFT] Error swapping previous/current: {e}")

        # final decision
        decision = "retrain" if (results["drift"]["any_stat_drift"] or results["drift"]["any_perf_drift"]) else "no_drift"
        results["decision"] = decision
        
        # Add files_compared for dashboard compatibility
        full_file_path = os.path.join(data_dir, current_filename)
        ref_file_path = None
        
        # Try to find the previous file from control_file.txt
        control_file = os.path.join(control_dir, "control_file.txt")
        if os.path.exists(control_file):
            with open(control_file, "r") as f:
                lines = f.readlines()
            if lines:
                # The last line is the most recent processed file (before the current one if not yet appended, 
                # or the current one if already appended. Usually check_drift runs before appending?)
                # Assuming check_drift runs before appending current file to control_file.
                last_ref_filename = lines[-1].strip().split(',')[0]
                ref_file_path = os.path.join(data_dir, last_ref_filename)

        results["files_compared"] = {
            "reference_file": ref_file_path if ref_file_path and os.path.exists(ref_file_path) else None,
            "current_file": full_file_path
        }
        
        _save_results(results, drift_path, logger)
        return decision

    except Exception as e:
        logger.error(f"[DRIFT] Unexpected error: {e}")
        results["decision"] = "end_error"
        _save_results(results, drift_path, logger)
        return "end_error"
