import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def save_drift_results(results: dict, output_dir: str):
    """
    Serialize and save drift results to JSON.
    Ensures numpy/scalar types are converted to native Python types.
    """
    drift_path = os.path.join(output_dir, "drift_results.json")

    def make_serializable(obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray, list)): return [make_serializable(x) for x in obj]
        if isinstance(obj, dict): return {kk: make_serializable(vv) for kk, vv in obj.items()}
        return obj

    try:
        serializable_results = make_serializable(results)
        with open(drift_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
        print(f"[DRIFT] Results saved to {drift_path}")
    except Exception as e:
        print(f"[ERROR] Could not save drift results: {e}")


def check_drift(X, y, detector, logger, perf_thresholds, model_filename,
                data_dir, output_dir, control_dir, model_dir):
    """
    Drift checker with:
      - Performance tests on CURRENT model vs user thresholds
      - Performance tests on PREVIOUS model vs user thresholds (if exists)
      - Comparative suite PREVIOUS vs CURRENT (>=30% relative degradation rule)
      - Statistical drift via Frouros (univariate: KS, PSI, Mann–Whitney, Cramér–von Mises, Hellinger, EMD)
      - Multivariate drift via Frouros (MMD, Energy Distance)
      - Automatic rollback (promote previous) if:
          A) previous passes all performance checks (no drift), OR
          B) current shows >=30% degradation vs previous (majority of metrics)
      - 'decision' written to drift_results.json:
          'train' | 'previous_promoted' | 'retrain' | 'no_drift' | 'end_error'
    """
    model_path = os.path.join(model_dir, model_filename)
    prev_model_path = model_path.replace(".pkl", "_previous.pkl")
    ref_path = os.path.join(control_dir, "previous_data.csv")

    drift_results = {"thresholds": perf_thresholds, "tests": {}, "drift": {}, "metrics": {}}

    # 0) No current model → request initial training
    if not os.path.exists(model_path):
        logger.info("No current model found → train from scratch")
        drift_results["info"] = "First model, no drift check performed"
        drift_results["decision"] = "train"
        save_drift_results(drift_results, output_dir)
        return "train"

    # 1) Load current model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading current model: {e}")
        drift_results["error"] = "failed_to_load_current_model"
        drift_results["decision"] = "end_error"
        save_drift_results(drift_results, output_dir)
        return "end"

    # Helper: run performance tests for a given model against user thresholds
    def _run_perf_tests(_model):
        results, flags = {}, {}
        if is_classifier(_model):
            if "accuracy" in perf_thresholds:
                d, res = detector.performance_degradation_test_accuracy(X, y, _model, threshold=perf_thresholds["accuracy"])
                results["Accuracy"] = {**res, "drift": d}; flags["accuracy"] = bool(d)
            if "balanced_accuracy" in perf_thresholds:
                d, res = detector.performance_degradation_test_balanced_accuracy(X, y, _model, threshold=perf_thresholds["balanced_accuracy"])
                results["Balanced Accuracy"] = {**res, "drift": d}; flags["balanced_accuracy"] = bool(d)
            if "f1" in perf_thresholds:
                d, res = detector.performance_degradation_test_f1(X, y, _model, threshold=perf_thresholds["f1"])
                results["F1 Score"] = {**res, "drift": d}; flags["f1"] = bool(d)
        elif is_regressor(_model):
            if "rmse" in perf_thresholds:
                d, res = detector.performance_degradation_test_rmse(X, y, _model, threshold=perf_thresholds["rmse"])
                results["RMSE"] = {**res, "drift": d}; flags["rmse"] = bool(d)
            if "r2" in perf_thresholds:
                d, res = detector.performance_degradation_test_r2(X, y, _model, threshold=perf_thresholds["r2"])
                results["R2"] = {**res, "drift": d}; flags["r2"] = bool(d)
            if "mae" in perf_thresholds:
                d, res = detector.performance_degradation_test_mae(X, y, _model, threshold=perf_thresholds["mae"])
                results["MAE"] = {**res, "drift": d}; flags["mae"] = bool(d)
            if "mse" in perf_thresholds:
                d, res = detector.performance_degradation_test_mse(X, y, _model, threshold=perf_thresholds["mse"])
                results["MSE"] = {**res, "drift": d}; flags["mse"] = bool(d)
        return results, flags

    # 2) Performance tests for CURRENT model
    perf_results_current, perf_flags_current = _run_perf_tests(model)
    drift_results["tests"]["Performance_Current"] = perf_results_current
    drift_results["drift"].update({f"current::{k}": v for k, v in perf_flags_current.items()})
    num_total_current = len(perf_flags_current)
    num_drift_current = sum(perf_flags_current.values())
    current_perf_drift = (num_total_current > 0 and num_drift_current >= (num_total_current / 2))

    # 3) If PREVIOUS model exists: run performance + comparative tests (>=30% rule)
    if os.path.exists(prev_model_path):
        try:
            prev_model = joblib.load(prev_model_path)

            # 3.a) Performance tests for PREVIOUS model
            perf_results_prev, perf_flags_prev = _run_perf_tests(prev_model)
            drift_results["tests"]["Performance_Previous"] = perf_results_prev
            drift_results["drift"].update({f"previous::{k}": v for k, v in perf_flags_prev.items()})
            previous_passes = (sum(perf_flags_prev.values()) == 0)  # strict: zero flags

            # 3.b) Comparative suite: CURRENT vs PREVIOUS (>= 30% relative drop/increase)
            task = "classification" if is_classifier(model) else "regression"
            comp_drift, comp_results, comp_flags = detector.performance_comparison_suite(
                X, y, prev_model, model, task=task, decay_ratio=0.30, average="macro"
            )
            drift_results["tests"]["Performance_Comparison"] = comp_results
            drift_results["drift"].update({f"comparison::{k}": bool(v) for k, v in comp_flags.items()})

            # 3.c) Rollback policy: promote previous if it passes OR current degrades >=30%
            if previous_passes or comp_drift:
                reason = "previous_passed_thresholds" if previous_passes else "current_degraded_vs_previous_30pct"
                try:
                    tmp_swap = model_path + ".swap"
                    os.replace(model_path, tmp_swap)
                    os.replace(prev_model_path, model_path)
                    os.replace(tmp_swap, prev_model_path)
                    logger.info(f"📌 Previous model promoted to current (reason: {reason})")
                    drift_results["promoted_model"] = True
                    drift_results["promotion_reason"] = reason
                    drift_results["decision"] = "previous_promoted"
                    save_drift_results(drift_results, output_dir)
                    return "end"  # operational recovery: no retrain needed
                except Exception as e:
                    logger.error(f"Error swapping previous/current models: {e}")
        except Exception as e:
            logger.error(f"Error loading/evaluating previous model: {e}")

    # 4) Statistical drift tests via Frouros (if reference dataset exists)
    stats_drift_detected = False
    if os.path.exists(ref_path):
        try:
            df_ref = pd.read_csv(ref_path)
            X_ref = df_ref.drop(columns=[y.name]) if (y is not None and y.name in df_ref.columns) else df_ref

            # --- Univariate (Frouros) ---
            ks_drift, ks_res   = detector.kolmogorov_smirnov_test(X_ref, X, alpha=0.05)
            psi_drift, psi_res = detector.population_stability_index_test(X_ref, X, psi_threshold=0.10, num_bins=10)
            mw_drift, mw_res   = detector.mann_whitney_test(X_ref, X, alpha=0.05)
            cvm_drift, cvm_res = detector.cramervonmises_test(X_ref, X, alpha=0.05)
            hd_drift, hd_res   = detector.hellinger_distance_test(X_ref, X, num_bins=30, threshold=0.10)
            emd_drift, emd_res = detector.earth_movers_distance_test(X_ref, X, threshold=None)  # adaptive if None

            # --- Multivariate (Frouros) ---
            mmd_drift, mmd_res = detector.mmd_test(X_ref, X, kernel="rbf", bandwidth="auto", alpha=0.05)
            ed_drift, ed_res   = detector.energy_distance_test(X_ref, X, alpha=0.05)

            # Store results
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
                "Kolmogorov-Smirnov": ks_drift,
                "PSI": psi_drift,
                "Mann-Whitney": mw_drift,
                "Cramér-von Mises": cvm_drift,
                "Hellinger Distance": hd_drift,
                "Earth Mover's Distance": emd_drift,
                "MMD": mmd_drift,
                "Energy Distance": ed_drift,
            })

            # Majority rule for statistical tests
            stats_flags = [
                ks_drift, psi_drift, mw_drift, cvm_drift,
                hd_drift, emd_drift, mmd_drift, ed_drift
            ]
            stats_drift_detected = (sum(bool(x) for x in stats_flags) >= (len(stats_flags) / 2))

        except Exception as e:
            logger.error(f"Error running statistical tests (Frouros): {e}")

    # 5) Final decision combining performance and statistical drift
    drift_detected = current_perf_drift or stats_drift_detected
    if drift_detected:
        logger.info("⚠️ Drift Detected → Retrain")
        drift_results["decision"] = "retrain"
    else:
        logger.info("✅ No drift detected → End")
        drift_results["decision"] = "no_drift"

    save_drift_results(drift_results, output_dir)
    return "retrain" if drift_detected else "end"
