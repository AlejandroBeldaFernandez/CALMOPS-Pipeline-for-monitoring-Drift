# pipeline/modules/check_drift.py
# -*- coding: utf-8 -*-
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor

def check_drift(
    X,
    y,
    detector,
    logger,
    perf_thresholds: dict,
    model_filename: str,
    data_dir: str,      # signature compatibility
    output_dir: str,
    control_dir: str,
    model_dir: str,
):
    """
    One-shot drift checker for the current incoming batch (X, y).

    1) Performance checks on CURRENT model vs thresholds_perf.
    2) If PREVIOUS model exists:
       - Performance checks on PREVIOUS.
       - Comparative suite CURRENT vs PREVIOUS (>=30% relative decay).
       - Auto rollback promote PREVIOUS if passes or CURRENT degrades enough.
    3) Statistical drift tests vs reference dataset if control/previous_data.csv exists.
    4) Return 'train' | 'retrain' | 'end' ('end' = ok or previous_promoted path).

    Writes JSON report at {output_dir}/drift_results.json
    """
    os.makedirs(output_dir, exist_ok=True)
    drift_path = os.path.join(output_dir, "drift_results.json")
    model_path = os.path.join(model_dir, model_filename)
    prev_model_path = model_path.replace(".pkl", "_previous.pkl")
    ref_path = os.path.join(control_dir, "previous_data.csv")

    drift_results = {
        "thresholds": perf_thresholds or {},
        "tests": {},
        "drift": {},
        "metrics": {},
        "decision": None,
    }

    def _serializable(obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray, list, tuple)): return [_serializable(x) for x in obj]
        if isinstance(obj, dict): return {k: _serializable(v) for k, v in obj.items()}
        return obj

    def _save_results():
        try:
            with open(drift_path, "w") as f:
                json.dump(_serializable(drift_results), f, indent=4)
            logger.info(f"[DRIFT] Results saved to {drift_path}")
        except Exception as e:
            logger.error(f"[DRIFT] Failed to save results: {e}")

    if not os.path.exists(model_path):
        logger.info("No current model found → train from scratch.")
        drift_results["info"] = "First model, no drift check performed."
        drift_results["decision"] = "train"
        _save_results()
        return "train"

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading current model: {e}")
        drift_results["error"] = "failed_to_load_current_model"
        drift_results["decision"] = "end_error"
        _save_results()
        return "end"

    def _run_perf_tests(_model):
        """Use detector's performance tests mapped to provided thresholds."""
        results, flags = {}, {}
        if is_classifier(_model):
            if "accuracy" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_accuracy(X, y, _model, threshold=perf_thresholds["accuracy"])
                results["Accuracy"] = {**res, "drift": bool(d)}; flags["accuracy"] = bool(d)
            if "balanced_accuracy" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_balanced_accuracy(X, y, _model, threshold=perf_thresholds["balanced_accuracy"])
                results["Balanced Accuracy"] = {**res, "drift": bool(d)}; flags["balanced_accuracy"] = bool(d)
            if "f1" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_f1(X, y, _model, threshold=perf_thresholds["f1"])
                results["F1 Score"] = {**res, "drift": bool(d)}; flags["f1"] = bool(d)
        elif is_regressor(_model):
            if "rmse" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_rmse(X, y, _model, threshold=perf_thresholds["rmse"])
                results["RMSE"] = {**res, "drift": bool(d)}; flags["rmse"] = bool(d)
            if "r2" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_r2(X, y, _model, threshold=perf_thresholds["r2"])
                results["R2"] = {**res, "drift": bool(d)}; flags["r2"] = bool(d)
            if "mae" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_mae(X, y, _model, threshold=perf_thresholds["mae"])
                results["MAE"] = {**res, "drift": bool(d)}; flags["mae"] = bool(d)
            if "mse" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_mse(X, y, _model, threshold=perf_thresholds["mse"])
                results["MSE"] = {**res, "drift": bool(d)}; flags["mse"] = bool(d)
        return results, flags

    # CURRENT
    perf_results_current, perf_flags_current = _run_perf_tests(model)
    drift_results["tests"]["Performance_Current"] = perf_results_current
    drift_results["drift"].update({f"current::{k}": bool(v) for k, v in perf_flags_current.items()})
    num_total_current = len(perf_flags_current)
    num_drift_current = sum(bool(v) for v in perf_flags_current.values())
    current_perf_drift = (num_total_current > 0 and num_drift_current >= (num_total_current / 2))

    # PREVIOUS
    if os.path.exists(prev_model_path):
        try:
            prev_model = joblib.load(prev_model_path)

            perf_results_prev, perf_flags_prev = _run_perf_tests(prev_model)
            drift_results["tests"]["Performance_Previous"] = perf_results_prev
            drift_results["drift"].update({f"previous::{k}": bool(v) for k, v in perf_flags_prev.items()})
            previous_passes = (sum(bool(v) for v in perf_flags_prev.values()) == 0)

            task = "classification" if is_classifier(model) else "regression"
            comp_drift, comp_results, comp_flags = detector.performance_comparison_suite(
                X, y, prev_model, model, task=task, decay_ratio=0.30, average="macro"
            )
            drift_results["tests"]["Performance_Comparison"] = comp_results
            drift_results["drift"].update({f"comparison::{k}": bool(v) for k, v in comp_flags.items()})

            if previous_passes or bool(comp_drift):
                reason = "previous_passed_thresholds" if previous_passes else "current_degraded_vs_previous_30pct"
                try:
                    tmp_swap = model_path + ".swap"
                    os.replace(model_path, tmp_swap)
                    os.replace(prev_model_path, model_path)
                    os.replace(tmp_swap, prev_model_path)

                    logger.info(f"Previous model promoted to current (reason: {reason}).")
                    drift_results["promoted_model"] = True
                    drift_results["promotion_reason"] = reason
                    drift_results["decision"] = "previous_promoted"
                    _save_results()
                    return "end"
                except Exception as e:
                    logger.error(f"Error swapping previous/current models: {e}")
        except Exception as e:
            logger.error(f"Error loading/evaluating previous model: {e}")

    # Statistical drift tests vs reference
    stats_drift_detected = False
    if os.path.exists(ref_path):
        try:
            df_ref = pd.read_csv(ref_path)
            if y is not None and getattr(y, "name", None) in df_ref.columns:
                X_ref = df_ref.drop(columns=[y.name])
            else:
                X_ref = df_ref

            ks_drift, ks_res   = detector.kolmogorov_smirnov_test(X_ref, X, alpha=0.05)
            psi_drift, psi_res = detector.population_stability_index_test(X_ref, X, psi_threshold=0.10, num_bins=10)
            mw_drift, mw_res   = detector.mann_whitney_test(X_ref, X, alpha=0.05)
            cvm_drift, cvm_res = detector.cramervonmises_test(X_ref, X, alpha=0.05)
            hd_drift, hd_res   = detector.hellinger_distance_test(X_ref, X, num_bins=30, threshold=0.10)
            emd_drift, emd_res = detector.earth_movers_distance_test(X_ref, X, threshold=None)

            mmd_drift, mmd_res = detector.mmd_test(X_ref, X, kernel="rbf", bandwidth="auto", alpha=0.05)
            ed_drift, ed_res   = detector.energy_distance_test(X_ref, X, alpha=0.05)

            drift_results["tests"].update({
                "Kolmogorov-Smirnov": ks_res,
                "PSI": psi_res,
                "Mann-Whitney": mw_res,
                "Cramér-von Mises": cvm_res,
                "Hellinger Distance": hd_res,
                "Earth Mover's Distance": emd_res,
                "MMD": mmd_res,
                "Energy Distance": ed_res,
            })
            drift_results["drift"].update({
                "Kolmogorov-Smirnov": bool(ks_drift),
                "PSI": bool(psi_drift),
                "Mann-Whitney": bool(mw_drift),
                "Cramér-von Mises": bool(cvm_drift),
                "Hellinger Distance": bool(hd_drift),
                "Earth Mover's Distance": bool(emd_drift),
                "MMD": bool(mmd_drift),
                "Energy Distance": bool(ed_drift),
            })

            stats_flags = [
                bool(ks_drift), bool(psi_drift), bool(mw_drift), bool(cvm_drift),
                bool(hd_drift), bool(emd_drift), bool(mmd_drift), bool(ed_drift)
            ]
            stats_drift_detected = (sum(stats_flags) >= (len(stats_flags) / 2))

        except Exception as e:
            logger.error(f"Error running statistical tests: {e}")

    drift_detected = bool(current_perf_drift or stats_drift_detected)
    if drift_detected:
        logger.info("Drift detected → Retrain.")
        drift_results["decision"] = "retrain"
        _save_results()
        return "retrain"
    else:
        logger.info("No drift detected → End.")
        drift_results["decision"] = "no_drift"
        _save_results()
        return "end"
