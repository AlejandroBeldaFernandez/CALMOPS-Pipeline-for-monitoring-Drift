# modules/check_drift.py
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
    data_dir: str,      # kept for signature compatibility (not used)
    output_dir: str,
    control_dir: str,
    model_dir: str,
    # --- bloques ---
    blocks: pd.Series | None = None,   # ids de bloque alineados con X/y
    block_col: str | None = None,      # nombre lógico (para reporting)
):
    """
    Drift checker global y por bloque.

    Global:
      - Performance del modelo actual vs umbrales.
      - Comparativa con modelo previo (30% decay rule) + posible rollback.
      - Drift estadístico vs control/previous_data.csv si existe.

    Por bloque (si `blocks` disponible):
      - Performance por bloque (umbral).
      - Tests estadísticos por bloque comparando X_bloque vs X_restante.
      - Si un bloque dispara flags → cuenta como drift.
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
        "blockwise": {"block_col": block_col, "by_block": {}},
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

    # 0) No hay modelo → entrenar
    if not os.path.exists(model_path):
        logger.info("No current model found → train from scratch.")
        drift_results["info"] = "First model, no drift check performed."
        drift_results["decision"] = "train"
        _save_results()
        return "train"

    # 1) Cargar modelo actual
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading current model: {e}")
        drift_results["error"] = "failed_to_load_current_model"
        drift_results["decision"] = "end_error"
        _save_results()
        return "end"

    # Helper: performance vs umbrales
    def _run_perf_tests(_model, X_, y_):
        results, flags = {}, {}
        if is_classifier(_model):
            if "accuracy" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_accuracy(X_, y_, _model, threshold=perf_thresholds["accuracy"])
                results["Accuracy"] = {**res, "drift": bool(d)}; flags["accuracy"] = bool(d)
            if "balanced_accuracy" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_balanced_accuracy(X_, y_, _model, threshold=perf_thresholds["balanced_accuracy"])
                results["Balanced Accuracy"] = {**res, "drift": bool(d)}; flags["balanced_accuracy"] = bool(d)
            if "f1" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_f1(X_, y_, _model, threshold=perf_thresholds["f1"])
                results["F1 Score"] = {**res, "drift": bool(d)}; flags["f1"] = bool(d)
        elif is_regressor(_model):
            if "rmse" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_rmse(X_, y_, _model, threshold=perf_thresholds["rmse"])
                results["RMSE"] = {**res, "drift": bool(d)}; flags["rmse"] = bool(d)
            if "r2" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_r2(X_, y_, _model, threshold=perf_thresholds["r2"])
                results["R2"] = {**res, "drift": bool(d)}; flags["r2"] = bool(d)
            if "mae" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_mae(X_, y_, _model, threshold=perf_thresholds["mae"])
                results["MAE"] = {**res, "drift": bool(d)}; flags["mae"] = bool(d)
            if "mse" in (perf_thresholds or {}):
                d, res = detector.performance_degradation_test_mse(X_, y_, _model, threshold=perf_thresholds["mse"])
                results["MSE"] = {**res, "drift": bool(d)}; flags["mse"] = bool(d)
        return results, flags

    # 2) Performance global con modelo actual
    perf_results_current, perf_flags_current = _run_perf_tests(model, X, y)
    drift_results["tests"]["Performance_Current"] = perf_results_current
    drift_results["drift"].update({f"current::{k}": bool(v) for k, v in perf_flags_current.items()})
    current_perf_drift = (len(perf_flags_current) > 0 and sum(perf_flags_current.values()) >= (len(perf_flags_current) / 2))

    # 3) Comparativa con modelo previo + posible rollback
    prev_model_path = model_path.replace(".pkl", "_previous.pkl")
    if os.path.exists(prev_model_path):
        try:
            prev_model = joblib.load(prev_model_path)
            perf_results_prev, perf_flags_prev = _run_perf_tests(prev_model, X, y)
            drift_results["tests"]["Performance_Previous"] = perf_results_prev
            drift_results["drift"].update({f"previous::{k}": bool(v) for k, v in perf_flags_prev.items()})
            previous_passes = (sum(perf_flags_prev.values()) == 0)

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

    # 4) Drift estadístico vs referencia temporal
    stats_drift_detected = False
    ref_path = os.path.join(control_dir, "previous_data.csv")
    if os.path.exists(ref_path):
        try:
            df_ref = pd.read_csv(ref_path)
            X_ref = df_ref.drop(columns=[y.name]) if (y is not None and y.name in df_ref.columns) else df_ref

            ks_drift, ks_res   = detector.kolmogorov_smirnov_test(X_ref, X, alpha=0.05)
            psi_drift, psi_res = detector.population_stability_index_test(X_ref, X, psi_threshold=0.10, num_bins=10)
            mw_drift, mw_res   = detector.mann_whitney_test(X_ref, X, alpha=0.05)
            cvm_drift, cvm_res = detector.cramervonmises_test(X_ref, X, alpha=0.05)
            hd_drift, hd_res   = detector.hellinger_distance_test(X_ref, X, num_bins=30, threshold=0.10)
            emd_drift, emd_res = detector.earth_movers_distance_test(X_ref, X, threshold=None)
            mmd_drift, mmd_res = detector.mmd_test(X_ref, X, kernel="rbf", bandwidth="auto", alpha=0.05)
            ed_drift, ed_res   = detector.energy_distance_test(X_ref, X, alpha=0.05)

            drift_results["tests"].update({
                "Kolmogorov-Smirnov": ks_res, "PSI": psi_res, "Mann-Whitney": mw_res,
                "Cramér-von Mises": cvm_res, "Hellinger Distance": hd_res,
                "Earth Mover's Distance": emd_res, "MMD": mmd_res, "Energy Distance": ed_res,
            })
            drift_results["drift"].update({
                "Kolmogorov-Smirnov": bool(ks_drift), "PSI": bool(psi_drift),
                "Mann-Whitney": bool(mw_drift), "Cramér-von Mises": bool(cvm_drift),
                "Hellinger Distance": bool(hd_drift), "Earth Mover's Distance": bool(emd_drift),
                "MMD": bool(mmd_drift), "Energy Distance": bool(ed_drift),
            })
            flags = [bool(ks_drift), bool(psi_drift), bool(mw_drift), bool(cvm_drift),
                     bool(hd_drift), bool(emd_drift), bool(mmd_drift), bool(ed_drift)]
            stats_drift_detected = (sum(flags) >= len(flags) / 2)
        except Exception as e:
            logger.error(f"Error running statistical tests: {e}")

    # 5) Drift INTRA-DATASET por bloque
    block_drift_detected = False
    if blocks is not None:
        try:
            blocks = pd.Series(blocks).loc[X.index]  # alinear
            unique_blocks = list(blocks.dropna().unique())
            for bid in unique_blocks:
                mask = (blocks == bid)
                X_b, y_b = X.loc[mask], (y.loc[mask] if y is not None else None)
                X_rest   = X.loc[~mask]

                # a) Performance del modelo en este bloque
                perf_b, flags_b = _run_perf_tests(model, X_b, y_b)
                n = len(flags_b); bad = sum(flags_b.values())
                perf_block_fail = (n > 0 and bad >= n/2)

                # b) Deriva estadística del bloque vs resto
                ks_d, _   = detector.kolmogorov_smirnov_test(X_rest, X_b, alpha=0.05)
                psi_d, _  = detector.population_stability_index_test(X_rest, X_b, psi_threshold=0.10, num_bins=10)
                mw_d, _   = detector.mann_whitney_test(X_rest, X_b, alpha=0.05)
                cvm_d, _  = detector.cramervonmises_test(X_rest, X_b, alpha=0.05)
                hd_d, _   = detector.hellinger_distance_test(X_rest, X_b, num_bins=30, threshold=0.10)
                emd_d, _  = detector.earth_movers_distance_test(X_rest, X_b, threshold=None)
                mmd_d, _  = detector.mmd_test(X_rest, X_b, kernel="rbf", bandwidth="auto", alpha=0.05)
                ed_d, _   = detector.energy_distance_test(X_rest, X_b, alpha=0.05)
                stat_flags = [bool(ks_d), bool(psi_d), bool(mw_d), bool(cvm_d),
                              bool(hd_d), bool(emd_d), bool(mmd_d), bool(ed_d)]
                stats_block_fail = (sum(stat_flags) >= len(stat_flags)/2)

                drift_results["blockwise"]["by_block"][str(bid)] = {
                    "size": int(len(X_b)),
                    "performance": perf_b,
                    "performance_flags": {k: bool(v) for k, v in flags_b.items()},
                    "stats_flags": {
                        "KS": bool(ks_d), "PSI": bool(psi_d), "MW": bool(mw_d),
                        "CvM": bool(cvm_d), "Hellinger": bool(hd_d),
                        "EMD": bool(emd_d), "MMD": bool(mmd_d), "ED": bool(ed_d),
                    },
                    "block_drift": bool(perf_block_fail or stats_block_fail),
                }
                if perf_block_fail or stats_block_fail:
                    block_drift_detected = True
        except Exception as e:
            logger.error(f"Error computing blockwise drift: {e}")

    # 6) Decisión
    drift_detected = bool(current_perf_drift or stats_drift_detected or block_drift_detected)
    if drift_detected:
        logger.info("Drift detected → Retrain.")
        drift_results["decision"] = "retrain"
    else:
        logger.info("No drift detected → End.")
        drift_results["decision"] = "no_drift"

    _save_results()
    return "retrain" if drift_detected else "end"
