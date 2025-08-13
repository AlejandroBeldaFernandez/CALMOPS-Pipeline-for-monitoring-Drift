import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score

# pip install frouros
import numpy as np
import pandas as pd

# Quita: from frouros.detectors.data_drift.batch.multivariate import ( MMD, EnergyDistance )
from functools import partial

from frouros.detectors.data_drift import MMD  # batch MMD
from frouros.detectors.data_drift.batch.distance_based import (
    EnergyDistance,            # dist multivariante
    PSI as PSI_Detector,       # univariante
    HellingerDistance as HellingerDet,
    EMD as EMD_Detector,
)
from frouros.detectors.data_drift.batch.statistical_test import (
    KSTest, MannWhitneyUTest, CVMTest,
)
from frouros.callbacks.batch import PermutationTestDistanceBased
from frouros.utils.kernels import rbf_kernel


def _to_float_array(x: pd.Series | np.ndarray) -> np.ndarray:
    """Return a 1D float vector with NaNs removed."""
    if isinstance(x, pd.Series):
        x = x.dropna().values
    x = np.asarray(x)
    return x[~np.isnan(x)].astype(float)


def _to_float_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return a 2D float matrix with numeric common cols and rows without NaNs."""
    X = df.copy()
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    # Drop rows containing NaNs
    X = X.dropna(axis=0, how="any")
    return X.values.astype(float)


class DriftDetector:
    """
    Drift tests implemented using Frouros.
    All methods return:
        (drift_detected: bool, results: dict)
    where 'results' is a per-feature mapping (for univariate) or a global dict (for multivariate).
    """

    # =========================
    # UNIVARIATE TESTS
    # =========================

    def kolmogorov_smirnov_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: float = 0.05):
        """Column-wise KS; drift if p < alpha."""
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

    def mann_whitney_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: float = 0.05):
        """Column-wise Mann–Whitney; drift if p < alpha."""
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

    def cramervonmises_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: float = 0.05):
        """Column-wise Cramér–von Mises; drift if p < alpha."""
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

    def population_stability_index_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, psi_threshold: float = 0.10, num_bins: int = 10):
        """Column-wise PSI; drift if PSI > psi_threshold."""
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

    def hellinger_distance_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, num_bins: int = 30, threshold: float = 0.10):
        """Column-wise Hellinger; drift if distance > threshold."""
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

    def earth_movers_distance_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame, threshold: float | None = None):
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
    # MULTIVARIATE TESTS
    # =========================

    def mmd_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame,
             alpha: float = 0.05,
             kernel: str = "rbf",
             bandwidth: float | str = "auto",
             num_permutations: int = 1000):
        """
        Batch MMD with permutation test for p-value; drift if p < alpha.
        Supports kernel='rbf' (default) and 'linear'. Any other value falls back to 'rbf'.
        """
        try:
            X1 = _to_float_matrix(X_ref)
            X2 = _to_float_matrix(X_new)
            if X1.size == 0 or X2.size == 0:
                return False, {"error": "empty data", "drift": False}

            # Select kernel
            kernel_used = kernel.lower() if isinstance(kernel, str) else "rbf"

            if kernel_used == "rbf":
                # estimate sigma if requested
                if bandwidth == "auto":
                    from sklearn.metrics import pairwise_distances
                    sample = np.vstack([X1[:min(200, len(X1))], X2[:min(200, len(X2))]])
                    med = np.median(pairwise_distances(sample, sample))
                    sigma = med if med > 0 else 1.0
                else:
                    sigma = float(bandwidth)
                kfun = partial(rbf_kernel, sigma=sigma)
                kernel_info = {"kernel": "rbf", "bandwidth": sigma}

            elif kernel_used == "linear":
                # simple dot-product kernel
                def linear_kernel(a, b):
                    return a @ b.T
                kfun = linear_kernel
                kernel_info = {"kernel": "linear", "bandwidth": None}

            else:
                # fallback to rbf to avoid breaking existing calls
                if bandwidth == "auto":
                    from sklearn.metrics import pairwise_distances
                    sample = np.vstack([X1[:min(200, len(X1))], X2[:min(200, len(X2))]])
                    med = np.median(pairwise_distances(sample, sample))
                    sigma = med if med > 0 else 1.0
                else:
                    sigma = float(bandwidth)
                kfun = partial(rbf_kernel, sigma=sigma)
                kernel_info = {"kernel": f"fallback_rbf_from_{kernel_used}", "bandwidth": sigma}

            detector = MMD(
                kernel=kfun,
                callbacks=[
                    PermutationTestDistanceBased(
                        num_permutations=num_permutations,
                        random_state=31,
                        num_jobs=-1,
                        method="approx",   # usa "exact" solo con n pequeño
                        name="permutation_test",
                    )
                ],
            )

            _ = detector.fit(X=X1)
            _, logs = detector.compare(X=X2)
            p_value = logs["permutation_test"]["p_value"]
            stat = logs["permutation_test"].get("statistic")
            drift = p_value < alpha

            return drift, {
                "statistic": float(stat) if stat is not None else None,
                "p_value": float(p_value),
                "alpha": alpha,
                **kernel_info,
                "drift": drift,
            }

        except Exception as e:
            return False, {"error": str(e), "drift": False}


    def energy_distance_test(self, X_ref: pd.DataFrame, X_new: pd.DataFrame,
                         alpha: float = 0.05, num_permutations: int = 1000):
        """
        Energy Distance (global). Se añade permutación para obtener p-value; drift si p < alpha.
        """
        try:
            X1 = _to_float_matrix(X_ref)
            X2 = _to_float_matrix(X_new)
            if X1.size == 0 or X2.size == 0:
                return False, {"error": "empty data", "drift": False}

            detector = EnergyDistance(
                callbacks=[
                    PermutationTestDistanceBased(
                        num_permutations=num_permutations,
                        random_state=31,
                        num_jobs=-1,
                        method="approx",
                        name="permutation_test",
                    )
                ]
            )
            _ = detector.fit(X=X1)
            res, logs = detector.compare(X=X2)  # res.distance
            p_value = logs["permutation_test"].get("p_value")
            if p_value is not None:
                drift = p_value < alpha
                out = {"distance": float(getattr(res, "distance", np.nan)),
                    "p_value": float(p_value), "alpha": alpha, "drift": drift}
            else:
                # fallback sin p-value
                out = {"distance": float(getattr(res, "distance", np.nan)),
                    "p_value": None, "alpha": None, "drift": False}
                drift = False
            return drift, out
        except Exception as e:
            return False, {"error": str(e), "drift": False}


    def performance_degradation_test_balanced_accuracy(self, X: pd.DataFrame, y: pd.Series, model, threshold=0.9):
        """Performance degradation test using balanced accuracy."""
        try:
            predictions = model.predict(X)
            new_acc = balanced_accuracy_score(y, predictions)
            return new_acc < threshold, {'balanced_accuracy': new_acc, 'threshold': threshold}
        except Exception as e:
            return False, {'error': str(e)}

    def performance_degradation_test_accuracy(self, X: pd.DataFrame, y: pd.Series, model, threshold=0.9):
        """Performance degradation test using accuracy."""
        try:
            predictions = model.predict(X)
            new_acc = accuracy_score(y, predictions)
            return new_acc < threshold, {'accuracy': new_acc, 'threshold': threshold}
        except Exception as e:
            return False, {'error': str(e)}
        
    def performance_degradation_test_f1(self, X: pd.DataFrame, y: pd.Series, model, threshold=0.9):
        """Performance degradation test using F1 score."""
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, average='binary')  # Or 'macro' for multi-class
        return f1 < threshold,  {'F1': f1, 'threshold': threshold}

    def performance_degradation_test_rmse(self, X: pd.DataFrame, y: pd.Series, model, threshold=1.2):
        """Performance degradation test using RMSE (Root Mean Squared Error)."""
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        return rmse > threshold, {'RMSE': rmse, 'threshold': threshold}
    
    def performance_degradation_test_r2(self, X: pd.DataFrame, y: pd.Series, model, threshold=0.5):
        """Performance degradation test using R2 score."""
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

    def compare_accuracy_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, decay_ratio: float = 0.30):
        """Drift if accuracy(current) dropped >= decay_ratio vs previous."""
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        acc_prev = accuracy_score(y, y_prev)
        acc_curr = accuracy_score(y, y_curr)
        drift, rel = self._safe_rel_drop(acc_prev, acc_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": "accuracy", "prev": acc_prev, "current": acc_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_balanced_accuracy_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, decay_ratio: float = 0.30):
        """Drift if balanced_accuracy(current) dropped >= decay_ratio vs previous."""
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        b_prev = balanced_accuracy_score(y, y_prev)
        b_curr = balanced_accuracy_score(y, y_curr)
        drift, rel = self._safe_rel_drop(b_prev, b_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": "balanced_accuracy", "prev": b_prev, "current": b_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_f1_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, decay_ratio: float = 0.30, average: str = "macro"):
        """Drift if F1(current) dropped >= decay_ratio vs previous. Use average='macro' (multi-class) or 'binary'."""
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        f_prev = f1_score(y, y_prev, average=average)
        f_curr = f1_score(y, y_curr, average=average)
        drift, rel = self._safe_rel_drop(f_prev, f_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": f"f1_{average}", "prev": f_prev, "current": f_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_r2_drop(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, decay_ratio: float = 0.30):
        """Drift if R2(current) dropped >= decay_ratio vs previous."""
        r_prev = r2_score(y, model_prev.predict(X))
        r_curr = r2_score(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(r_prev, r_curr, higher_is_better=True, decay_ratio=decay_ratio)
        return drift, {"metric": "r2", "prev": r_prev, "current": r_curr, "relative_drop": rel, "threshold": decay_ratio}

    def compare_rmse_increase(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, decay_ratio: float = 0.30):
        """Drift if RMSE(current) increased >= decay_ratio vs previous."""
        rmse_prev = mean_squared_error(y, model_prev.predict(X), squared=False)
        rmse_curr = mean_squared_error(y, model_curr.predict(X), squared=False)
        drift, rel = self._safe_rel_drop(rmse_prev, rmse_curr, higher_is_better=False, decay_ratio=decay_ratio)
        return drift, {"metric": "rmse", "prev": rmse_prev, "current": rmse_curr, "relative_increase": rel, "threshold": decay_ratio}

    def compare_mae_increase(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, decay_ratio: float = 0.30):
        """Drift if MAE(current) increased >= decay_ratio vs previous."""
        from sklearn.metrics import mean_absolute_error
        mae_prev = mean_absolute_error(y, model_prev.predict(X))
        mae_curr = mean_absolute_error(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(mae_prev, mae_curr, higher_is_better=False, decay_ratio=decay_ratio)
        return drift, {"metric": "mae", "prev": mae_prev, "current": mae_curr, "relative_increase": rel, "threshold": decay_ratio}

    def compare_mse_increase(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr, decay_ratio: float = 0.30):
        """Drift if MSE(current) increased >= decay_ratio vs previous."""
        mse_prev = mean_squared_error(y, model_prev.predict(X))
        mse_curr = mean_squared_error(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(mse_prev, mse_curr, higher_is_better=False, decay_ratio=decay_ratio)
        return drift, {"metric": "mse", "prev": mse_prev, "current": mse_curr, "relative_increase": rel, "threshold": decay_ratio}

    def performance_comparison_suite(self, X: pd.DataFrame, y: pd.Series, model_prev, model_curr,
                                     task: str = "classification",
                                     decay_ratio: float = 0.30,
                                     average: str = "macro"):
        """
        Run a set of comparative tests (previous vs current).
        Returns:
            drift_detected (bool), results (dict per metric), flags (dict[metric]=bool)
        """
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
