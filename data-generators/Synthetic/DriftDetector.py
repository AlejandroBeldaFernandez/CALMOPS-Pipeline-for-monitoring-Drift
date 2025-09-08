import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score

# pip install frouros
from frouros.detectors.data_drift import MMD  # batch MMD
from frouros.detectors.data_drift.batch.distance_based import (
    PSI as PSI_Detector,       # univariante
    HellingerDistance as HellingerDet,
    EMD as EMD_Detector,
)
from frouros.detectors.data_drift.batch.statistical_test import (
    KSTest, MannWhitneyUTest, CVMTest,
)


@dataclass
class DetectorConfig:
    """Configuration class for drift detection parameters"""
    # Statistical test parameters
    alpha: float = 0.05
    
    # PSI parameters
    psi_threshold: float = 0.10
    psi_num_bins: int = 10
    
    # Hellinger distance parameters
    hellinger_threshold: float = 0.10
    hellinger_num_bins: int = 30
    
    # EMD parameters
    emd_threshold: Optional[float] = None  # If None, uses adaptive threshold
    
    # Performance degradation parameters
    accuracy_threshold: float = 0.9
    balanced_accuracy_threshold: float = 0.9
    f1_threshold: float = 0.9
    rmse_threshold: float = 1.2
    r2_threshold: float = 0.5
    
    # Comparative performance parameters
    decay_ratio: float = 0.30
    f1_average: str = "macro"  # "macro" or "binary"


def _to_float_array(x: pd.Series | np.ndarray) -> np.ndarray:
    """Return a 1D float vector with NaNs removed."""
    if isinstance(x, pd.Series):
        x = x.dropna().values
    x = np.asarray(x)
    return x[~np.isnan(x)].astype(float)


class DriftDetector:
    """
    Drift tests implemented using Frouros.
    All methods return:
        (drift_detected: bool, results: dict)
    where 'results' is a per-feature mapping (for univariate) or a global dict (for multivariate).
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize drift detector with configuration"""
        self.config = config if config is not None else DetectorConfig()

    # =========================
    # UNIVARIATE TESTS
    # =========================

    def kolmogorov_smirnov_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: Optional[float] = None):
        """Column-wise KS; drift if p < alpha."""
        alpha = alpha if alpha is not None else self.config.alpha
        drift_detected, results = False, {}
        det = KSTest()
        for col in X_ref.columns:
            if col not in X_new.columns:
                continue
            try:
                x1 = _to_float_array(X_ref[col])
                x2 = _to_float_array(X_new[col])
                if x1.size == 0 or x2.size == 0:
                    results[col] = {"error": "empty data", "drift": False}
                    continue
                det.reset()
                det.fit(X=x1)
                res, _ = det.compare(X=x2)
                p = float(res.p_value)
                stat = float(res.statistic)
                d = p < alpha
                results[col] = {"statistic": stat, "p_value": p, "alpha": alpha, "drift": d}
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def mann_whitney_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: Optional[float] = None):
        """Column-wise Mann–Whitney; drift if p < alpha."""
        alpha = alpha if alpha is not None else self.config.alpha
        drift_detected, results = False, {}
        det = MannWhitneyUTest()
        for col in X_ref.columns:
            if col not in X_new.columns:
                continue
            try:
                x1 = _to_float_array(X_ref[col])
                x2 = _to_float_array(X_new[col])
                if x1.size == 0 or x2.size == 0:
                    results[col] = {"error": "empty data", "drift": False}
                    continue
                det.reset()
                det.fit(X=x1)
                res, _ = det.compare(X=x2)
                p = float(res.p_value)
                stat = float(res.statistic)
                d = p < alpha
                results[col] = {"statistic": stat, "p_value": p, "alpha": alpha, "drift": d}
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def cramervonmises_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: Optional[float] = None):
        """Column-wise Cramér–von Mises; drift if p < alpha."""
        alpha = alpha if alpha is not None else self.config.alpha
        drift_detected, results = False, {}
        det = CVMTest()
        for col in X_ref.columns:
            if col not in X_new.columns:
                continue
            try:
                x1 = _to_float_array(X_ref[col])
                x2 = _to_float_array(X_new[col])
                if x1.size == 0 or x2.size == 0:
                    results[col] = {"error": "empty data", "drift": False}
                    continue
                det.reset()
                det.fit(X=x1)
                res, _ = det.compare(X=x2)
                p = float(res.p_value)
                stat = float(res.statistic)
                d = p < alpha
                results[col] = {"statistic": stat, "p_value": p, "alpha": alpha, "drift": d}
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def population_stability_index_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, 
                                      psi_threshold: Optional[float] = None, 
                                      num_bins: Optional[int] = None):
        """Column-wise PSI; drift if PSI > psi_threshold."""
        psi_threshold = psi_threshold if psi_threshold is not None else self.config.psi_threshold
        num_bins = num_bins if num_bins is not None else self.config.psi_num_bins
        drift_detected, results = False, {}
        det = PSI_Detector(num_bins=num_bins)
        for col in X_ref.columns:
            if col not in X_new.columns:
                continue
            try:
                x1 = _to_float_array(X_ref[col])
                x2 = _to_float_array(X_new[col])
                if x1.size == 0 or x2.size == 0:
                    results[col] = {"error": "empty data", "drift": False}
                    continue
                det.reset()
                det.fit(X=x1)
                res, _ = det.compare(X=x2)  # res.distance = PSI
                psi_val = float(res.distance)
                d = psi_val > psi_threshold
                results[col] = {"psi": psi_val, "threshold": psi_threshold, "num_bins": num_bins, "drift": d}
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def hellinger_distance_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, 
                               num_bins: Optional[int] = None, 
                               threshold: Optional[float] = None):
        """Column-wise Hellinger; drift if distance > threshold."""
        num_bins = num_bins if num_bins is not None else self.config.hellinger_num_bins
        threshold = threshold if threshold is not None else self.config.hellinger_threshold
        drift_detected, results = False, {}
        det = HellingerDet(num_bins=num_bins)
        for col in X_ref.columns:
            if col not in X_new.columns:
                continue
            try:
                x1 = _to_float_array(X_ref[col])
                x2 = _to_float_array(X_new[col])
                if x1.size == 0 or x2.size == 0:
                    results[col] = {"error": "empty data", "drift": False}
                    continue
                det.reset()
                det.fit(X=x1)
                res, _ = det.compare(X=x2)  # res.distance
                h = float(res.distance)
                d = h > threshold
                results[col] = {"hellinger_distance": h, "threshold": threshold, "num_bins": num_bins, "drift": d}
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def earth_movers_distance_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, 
                                  threshold: Optional[float] = None):
        """Column-wise EMD; drift if EMD > threshold (if None, adaptive threshold 0.1*std_ref)."""
        drift_detected, results = False, {}
        det = EMD_Detector()
        for col in X_ref.columns:
            if col not in X_new.columns:
                continue
            try:
                x1 = _to_float_array(X_ref[col])
                x2 = _to_float_array(X_new[col])
                if x1.size == 0 or x2.size == 0:
                    results[col] = {"error": "empty data", "drift": False}
                    continue
                det.reset()
                det.fit(X=x1)
                res, _ = det.compare(X=x2)  # res.distance
                emd = float(res.distance)
                thr = (0.1 * np.std(x1)) if threshold is None else float(threshold)
                d = emd > thr
                results[col] = {"emd_distance": emd, "threshold": thr, "drift": d}
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    # =========================
    # PERFORMANCE DEGRADATION TESTS
    # =========================

    def performance_degradation_test_balanced_accuracy(self, X: pd.DataFrame, y: pd.Series, model, 
                                                      threshold: Optional[float] = None):
        """Performance degradation test using balanced accuracy."""
        threshold = threshold if threshold is not None else self.config.balanced_accuracy_threshold
        try:
            predictions = model.predict(X)
            new_acc = balanced_accuracy_score(y, predictions)
            return new_acc < threshold, {'balanced_accuracy': new_acc, 'threshold': threshold}
        except Exception as e:
            return False, {'error': str(e)}

    def performance_degradation_test_accuracy(self, X: pd.DataFrame, y: pd.Series, model, 
                                            threshold: Optional[float] = None):
        """Performance degradation test using accuracy."""
        threshold = threshold if threshold is not None else self.config.accuracy_threshold
        try:
            predictions = model.predict(X)
            new_acc = accuracy_score(y, predictions)
            return new_acc < threshold, {'accuracy': new_acc, 'threshold': threshold}
        except Exception as e:
            return False, {'error': str(e)}
        
    def performance_degradation_test_f1(self, X: pd.DataFrame, y: pd.Series, model, 
                                       threshold: Optional[float] = None):
        """Performance degradation test using F1 score."""
        threshold = threshold if threshold is not None else self.config.f1_threshold
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, average='binary')  # Or 'macro' for multi-class
        return f1 < threshold,  {'F1': f1, 'threshold': threshold}

    def performance_degradation_test_rmse(self, X: pd.DataFrame, y: pd.Series, model, 
                                         threshold: Optional[float] = None):
        """Performance degradation test using RMSE (Root Mean Squared Error)."""
        threshold = threshold if threshold is not None else self.config.rmse_threshold
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        return rmse > threshold, {'RMSE': rmse, 'threshold': threshold}
    
    def performance_degradation_test_r2(self, X: pd.DataFrame, y: pd.Series, model, 
                                       threshold: Optional[float] = None):
        """Performance degradation test using R2 score."""
        threshold = threshold if threshold is not None else self.config.r2_threshold
        r2 = r2_score(y, model.predict(X))
        return r2 < threshold, {'R2': r2, 'threshold': threshold}

    # ===========================
    # Comparative performance tests (previous vs current)
    # Drift if current degrades >= decay_ratio (default 0.30)
    # ===========================
    @staticmethod
    def _safe_rel_drop(prev_value: float, curr_value: float, higher_is_better: bool, decay_ratio: float, eps: float = 1e-12):
        """
        Compute relative change (drop or increase) robustly.
        Returns:
            drift (bool), rel_change (float or None)
        rel_change is expressed as a positive fraction (e.g., 0.32 means 32% drop/increase).
        """
        # For metrics where higher is better (accuracy, f1, r2):
        #   drop = (prev - curr) / max(|prev|, eps)  -> drift if drop >= decay_ratio
        # For metrics where lower is better (rmse, mae, mse):
        #   increase = (curr - prev) / max(|prev|, eps)  -> drift if increase >= decay_ratio
        denom = max(abs(prev_value), eps)

        if higher_is_better:
            rel_change = (prev_value - curr_value) / denom
            drift = rel_change >= decay_ratio
        else:
            rel_change = (curr_value - prev_value) / denom
            drift = rel_change >= decay_ratio

        # If prev is ~0 (e.g., r2 near 0), the relative change can be noisy; still return the computed value.
        return drift, rel_change

    def compare_accuracy_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, 
                             decay_ratio: Optional[float] = None):
        """Drift if accuracy(current) dropped >= decay_ratio vs previous."""
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        acc_prev = accuracy_score(y, y_prev)
        acc_curr = accuracy_score(y, y_curr)
        drift, rel = self._safe_rel_drop(acc_prev, acc_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": "accuracy", "prev": acc_prev, "current": acc_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_balanced_accuracy_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, 
                                      decay_ratio: Optional[float] = None):
        """Drift if balanced_accuracy(current) dropped >= decay_ratio vs previous."""
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        b_prev = balanced_accuracy_score(y, y_prev)
        b_curr = balanced_accuracy_score(y, y_curr)
        drift, rel = self._safe_rel_drop(b_prev, b_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": "balanced_accuracy", "prev": b_prev, "current": b_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_f1_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, 
                       decay_ratio: Optional[float] = None, average: Optional[str] = None):
        """Drift if F1(current) dropped >= decay_ratio vs previous. Use average='macro' (multi-class) or 'binary'."""
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        average = average if average is not None else self.config.f1_average
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        f_prev = f1_score(y, y_prev, average=average)
        f_curr = f1_score(y, y_curr, average=average)
        drift, rel = self._safe_rel_drop(f_prev, f_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": f"f1_{average}", "prev": f_prev, "current": f_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_r2_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, 
                       decay_ratio: Optional[float] = None):
        """Drift if R2(current) dropped >= decay_ratio vs previous."""
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        r_prev = r2_score(y, model_prev.predict(X))
        r_curr = r2_score(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(r_prev, r_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": "r2", "prev": r_prev, "current": r_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_rmse_increase(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, 
                             decay_ratio: Optional[float] = None):
        """Drift if RMSE(current) increased >= decay_ratio vs previous."""
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        rmse_prev = mean_squared_error(y, model_prev.predict(X), squared=False)
        rmse_curr = mean_squared_error(y, model_curr.predict(X), squared=False)
        drift, rel = self._safe_rel_drop(rmse_prev, rmse_curr, higher_is_better=False, decay_ratio=decay_ratio)
        return drift, {"metric": "rmse", "prev": rmse_prev, "current": rmse_curr, "relative_increase": rel, "threshold": decay_ratio}

    def compare_mae_increase(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, 
                            decay_ratio: Optional[float] = None):
        """Drift if MAE(current) increased >= decay_ratio vs previous."""
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        from sklearn.metrics import mean_absolute_error
        mae_prev = mean_absolute_error(y, model_prev.predict(X))
        mae_curr = mean_absolute_error(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(mae_prev, mae_curr, higher_is_better=False, decay_ratio=decay_ratio)
        return drift, {"metric": "mae", "prev": mae_prev, "current": mae_curr, "relative_increase": rel, "threshold": decay_ratio}

    def compare_mse_increase(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, 
                            decay_ratio: Optional[float] = None):
        """Drift if MSE(current) increased >= decay_ratio vs previous."""
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        mse_prev = mean_squared_error(y, model_prev.predict(X))
        mse_curr = mean_squared_error(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(mse_prev, mse_curr, higher_is_better=False, decay_ratio=decay_ratio)
        return drift, {"metric": "mse", "prev": mse_prev, "current": mse_curr, "relative_increase": rel, "threshold": decay_ratio}

    def performance_comparison_suite(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr,
                                     task: str = "classification",
                                     decay_ratio: Optional[float] = None,
                                     average: Optional[str] = None):
        """
        Run a set of comparative tests (previous vs current).
        Returns:
            drift_detected (bool), results (dict per metric), flags (dict[metric]=bool)
        """
        decay_ratio = decay_ratio if decay_ratio is not None else self.config.decay_ratio
        average = average if average is not None else self.config.f1_average
        
        results = {}
        flags = {}

        if task == "classification":
            d, r = self.compare_accuracy_drop(X, y, model_prev, model_curr, decay_ratio); results["accuracy"] = r; flags["accuracy"] = d
            d, r = self.compare_balanced_accuracy_drop(X, y, model_prev, model_curr, decay_ratio); results["balanced_accuracy"] = r; flags["balanced_accuracy"] = d
            d, r = self.compare_f1_drop(X, y, model_prev, model_curr, decay_ratio, average=average); results[f"f1_{average}"] = r; flags["f1"] = d
        elif task == "regression":
            d, r = self.compare_r2_drop(X, y, model_prev, model_curr, decay_ratio); results["r2"] = r; flags["r2"] = d
            d, r = self.compare_rmse_increase(X, y, model_prev, model_curr, decay_ratio); results["rmse"] = r; flags["rmse"] = d
            d, r = self.compare_mae_increase(X, y, model_prev, model_curr, decay_ratio); results["mae"] = r; flags["mae"] = d
            d, r = self.compare_mse_increase(X, y, model_prev, model_curr, decay_ratio); results["mse"] = r; flags["mse"] = d
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        # Majority rule: drift if >= 50% of selected metrics flag drift
        n = len(flags)
        drift_detected = (n > 0 and sum(bool(v) for v in flags.values()) >= (n / 2.0))
        return drift_detected, results, flags


def compare_detectors(detectors: List[Tuple[str, DriftDetector]], 
                     X_ref: pd.DataFrame, 
                     X_new: pd.DataFrame,
                     methods: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Compare multiple drift detectors on the same data
    
    Args:
        detectors: List of (name, detector) tuples
        X_ref: Reference dataset
        X_new: New dataset to test for drift
        methods: List of methods to test. If None, tests all available methods
        
    Returns:
        Dictionary with results for each detector and method combination
    """
    if methods is None:
        methods = [
            'kolmogorov_smirnov_test',
            'mann_whitney_test', 
            'cramervonmises_test',
            'population_stability_index_test',
            'hellinger_distance_test',
            'earth_movers_distance_test'
        ]
    
    results = {}
    
    for detector_name, detector in detectors:
        results[detector_name] = {}
        
        for method in methods:
            if hasattr(detector, method):
                try:
                    drift_detected, method_results = getattr(detector, method)(X_ref, X_new)
                    results[detector_name][method] = {
                        'drift_detected': drift_detected,
                        'results': method_results
                    }
                except Exception as e:
                    results[detector_name][method] = {
                        'drift_detected': False,
                        'error': str(e)
                    }
    
    return results