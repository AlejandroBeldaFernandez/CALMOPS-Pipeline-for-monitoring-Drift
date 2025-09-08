import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
import sys
from pathlib import Path

# Add config path to import defaults
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.defaults import DRIFT_DETECTION

def save_drift_results(results: dict, output_dir: str, logger=None):
    """
    Serialize and save drift detection results to JSON format.
    
    Recursively converts numpy arrays and scalar types to native Python types
    to ensure proper JSON serialization. This function handles the complete
    drift analysis results including statistical tests, performance metrics,
    and decision outcomes.
    
    Args:
        results (dict): Complete drift analysis results containing test outcomes,
                       performance metrics, thresholds, and final decision
        output_dir (str): Directory path where drift_results.json will be saved
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
        if logger:
            logger.info(f"Drift analysis results saved to {drift_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save drift analysis results: {e}")


def check_drift(X, y, detector, logger, perf_thresholds, model_filename,
                data_dir, output_dir, control_dir, model_dir):
    """
    Comprehensive drift detection system with automatic model management.
    
    This function implements a multi-layered drift detection approach combining:
    1. Performance-based drift detection using user-defined thresholds
    2. Statistical drift detection using Frouros library implementations
    3. Comparative analysis between current and previous model versions
    4. Automatic rollback/promotion logic for operational stability
    
    Performance Detection:
        - Tests current model against absolute performance thresholds
        - Tests previous model (if available) against same thresholds
        - Implements majority voting: drift flagged if >=50% metrics fail
        
    Statistical Detection (Frouros Integration):
        - Univariate tests: Kolmogorov-Smirnov, PSI, Mann-Whitney, 
          Cramér-von Mises, Hellinger Distance, Earth Mover's Distance
        - Uses alpha=0.05 for hypothesis tests, configurable thresholds for others
        - Majority voting across all statistical tests for final decision
        
    Rollback Logic:
        - Promotes previous model if it passes all performance checks
        - Promotes previous model if current model shows >=30% relative degradation
        - Uses atomic file swapping for safe model replacement
        
    Decision Outcomes:
        - 'train': No current model exists, initial training required
        - 'previous_promoted': Previous model restored due to better performance
        - 'retrain': Drift detected, model retraining recommended
        - 'no_drift': No significant drift detected, current model maintained
        - 'end_error': Critical error prevented drift analysis completion
        
    Args:
        X: Feature matrix for drift analysis
        y: Target vector for performance evaluation
        detector: Frouros detector instance with statistical test methods
        logger: Logging instance for operational messages
        perf_thresholds: Dictionary of performance metric thresholds
        model_filename: Name of current model file
        data_dir: Directory containing current dataset
        output_dir: Directory for saving drift analysis results
        control_dir: Directory containing reference/baseline data
        model_dir: Directory containing model files
        
    Returns:
        str: Decision code indicating required action
    """
    model_path = os.path.join(model_dir, model_filename)
    prev_model_path = model_path.replace(".pkl", "_previous.pkl")
    ref_path = os.path.join(control_dir, "previous_data.csv")

    drift_results = {"thresholds": perf_thresholds, "tests": {}, "drift": {}, "metrics": {}}

    # Initial training scenario: no current model exists
    if not os.path.exists(model_path):
        logger.info("No current model found, requesting initial training")
        drift_results["info"] = "First model deployment, drift analysis skipped"
        drift_results["decision"] = "train"
        save_drift_results(drift_results, output_dir, logger)
        return "train"

    # Load current model for analysis
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load current model from {model_path}: {e}")
        drift_results["error"] = "failed_to_load_current_model"
        drift_results["decision"] = "end_error"
        save_drift_results(drift_results, output_dir, logger)
        return "end"

    def _run_perf_tests(_model):
        """
        Execute performance degradation tests against user-defined thresholds.
        
        Runs appropriate metric tests based on model type (classifier/regressor).
        Each test compares current performance against absolute thresholds,
        flagging degradation when performance drops below acceptable levels.
        
        For classifiers: accuracy, balanced_accuracy, f1_score
        For regressors: rmse, r2, mae, mse
        
        Returns:
            tuple: (detailed_results_dict, binary_flags_dict)
        """
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

    # Performance evaluation: current model against thresholds
    perf_results_current, perf_flags_current = _run_perf_tests(model)
    drift_results["tests"]["Performance_Current"] = perf_results_current
    drift_results["drift"].update({f"current::{k}": v for k, v in perf_flags_current.items()})
    num_total_current = len(perf_flags_current)
    num_drift_current = sum(perf_flags_current.values())
    # Apply majority voting: drift detected if >=50% of metrics fail thresholds
    # This reduces false positives from single metric fluctuations
    perf_threshold = DRIFT_DETECTION["majority_voting"]["performance_threshold"]
    current_perf_drift = (num_total_current > 0 and num_drift_current >= (num_total_current * perf_threshold))

    # Comparative analysis: previous model evaluation and rollback logic
    if os.path.exists(prev_model_path):
        try:
            prev_model = joblib.load(prev_model_path)

            # Performance evaluation: previous model against thresholds
            perf_results_prev, perf_flags_prev = _run_perf_tests(prev_model)
            drift_results["tests"]["Performance_Previous"] = perf_results_prev
            drift_results["drift"].update({f"previous::{k}": v for k, v in perf_flags_prev.items()})
            # Strict evaluation: previous model must pass all threshold checks
            previous_passes = (sum(perf_flags_prev.values()) == 0)

            # Relative performance comparison with configurable degradation threshold
            # This suite compares current vs previous model performance directly
            task = "classification" if is_classifier(model) else "regression"
            degradation_ratio = DRIFT_DETECTION["comparative_analysis"]["degradation_ratio"]
            comp_drift, comp_results, comp_flags = detector.performance_comparison_suite(
                X, y, prev_model, model, task=task, decay_ratio=degradation_ratio, average="macro"
            )
            drift_results["tests"]["Performance_Comparison"] = comp_results
            drift_results["drift"].update({f"comparison::{k}": bool(v) for k, v in comp_flags.items()})

            # Automatic rollback decision logic using two-tier safety check:
            # Tier 1: Previous model meets all performance thresholds (strict validation)
            # Tier 2: Current model shows >=30% relative performance degradation vs previous
            # Either condition triggers immediate model rollback for system stability
            if previous_passes or comp_drift:
                reason = "previous_passed_thresholds" if previous_passes else "current_degraded_vs_previous_30pct"
                try:
                    # Atomic model file swapping for safe replacement
                    tmp_swap = model_path + ".swap"
                    os.replace(model_path, tmp_swap)
                    os.replace(prev_model_path, model_path)
                    os.replace(tmp_swap, prev_model_path)
                    logger.info(f"Previous model promoted to current - Reason: {reason}")
                    drift_results["promoted_model"] = True
                    drift_results["promotion_reason"] = reason
                    drift_results["decision"] = "previous_promoted"
                    save_drift_results(drift_results, output_dir, logger)
                    return "end"  # Operational recovery completed, no retraining needed
                except Exception as e:
                    logger.error(f"Failed to swap previous/current models: {e}")
        except Exception as e:
            logger.error(f"Error during previous model evaluation: {e}")

    # Statistical drift detection using Frouros library implementations
    stats_drift_detected = False
    if os.path.exists(ref_path):
        try:
            # Load reference dataset and prepare feature matrix
            df_ref = pd.read_csv(ref_path)
            X_ref = df_ref.drop(columns=[y.name]) if (y is not None and y.name in df_ref.columns) else df_ref

            # Univariate statistical tests for distribution drift detection
            # Each test compares reference vs current data distributions
            
            # Kolmogorov-Smirnov: Non-parametric test for distribution equality
            # H0: samples come from same continuous distribution
            alpha = DRIFT_DETECTION["statistical_tests"]["alpha"]
            ks_drift, ks_res = detector.kolmogorov_smirnov_test(X_ref, X, alpha=alpha)
            
            # Population Stability Index: Measures distribution shift via binned comparison
            # Threshold 0.10: moderate drift, >0.25 indicates significant shift
            psi_config = DRIFT_DETECTION["statistical_tests"]
            psi_drift, psi_res = detector.population_stability_index_test(
                X_ref, X, 
                psi_threshold=psi_config["psi_threshold"], 
                num_bins=psi_config["psi_num_bins"]
            )
            
            # Mann-Whitney U: Non-parametric test for median differences
            # Tests whether distributions have same location parameter
            mw_drift, mw_res = detector.mann_whitney_test(X_ref, X, alpha=alpha)
            
            # Cramér-von Mises: Tests goodness-of-fit between distributions
            # More sensitive to tail differences than KS test
            cvm_drift, cvm_res = detector.cramervonmises_test(X_ref, X, alpha=alpha)
            
            # Hellinger Distance: Measures overlap between probability distributions
            # Threshold 0.10: significant divergence in distribution shapes
            hd_config = DRIFT_DETECTION["statistical_tests"]
            hd_drift, hd_res = detector.hellinger_distance_test(
                X_ref, X, 
                num_bins=hd_config["hellinger_num_bins"], 
                threshold=hd_config["hellinger_threshold"]
            )
            
            # Earth Mover's Distance (Wasserstein): Optimal transport cost between distributions
            # Adaptive threshold when None: uses data-driven threshold estimation
            emd_drift, emd_res = detector.earth_movers_distance_test(X_ref, X, threshold=None)

            # Store detailed test results for analysis
            drift_results["tests"].update({
                "Kolmogorov-Smirnov": ks_res,
                "PSI": psi_res,
                "Mann-Whitney": mw_res,
                "Cramér-von Mises": cvm_res,
                "Hellinger Distance": hd_res,
                "Earth Mover's Distance": emd_res,
            })
            drift_results["drift"].update({
                "Kolmogorov-Smirnov": ks_drift,
                "PSI": psi_drift,
                "Mann-Whitney": mw_drift,
                "Cramér-von Mises": cvm_drift,
                "Hellinger Distance": hd_drift,
                "Earth Mover's Distance": emd_drift,
            })

            # Majority voting across statistical tests for robust drift detection
            # Reduces false positives from individual test sensitivity
            # Requires configurable percentage of tests to agree on drift detection
            stats_flags = [
                ks_drift, psi_drift, mw_drift, cvm_drift,
                hd_drift, emd_drift
            ]
            stats_threshold = DRIFT_DETECTION["majority_voting"]["statistical_threshold"]
            stats_drift_detected = (sum(bool(x) for x in stats_flags) >= (len(stats_flags) * stats_threshold))

        except Exception as e:
            logger.error(f"Statistical drift analysis failed: {e}")

    # Final decision: combine performance and statistical drift indicators
    # Uses logical OR: drift detected if either performance OR statistical tests indicate drift
    drift_detected = current_perf_drift or stats_drift_detected
    
    if drift_detected:
        logger.info("Drift detected - Model retraining recommended")
        drift_results["decision"] = "retrain"
    else:
        logger.info("No significant drift detected - Current model maintained")
        drift_results["decision"] = "no_drift"

    save_drift_results(drift_results, output_dir)
    return "retrain" if drift_detected else "end"
