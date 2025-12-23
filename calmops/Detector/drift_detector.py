import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from pandas.api.types import is_numeric_dtype

# Frouros imports (drift detection)
from frouros.detectors.data_drift import MMD  # batch MMD
from frouros.detectors.data_drift.batch.distance_based import PSI as PSI_Detector
from frouros.detectors.data_drift.batch.statistical_test import (
    KSTest,
    MannWhitneyUTest,
    ChiSquareTest,
    CVMTest,
)

# Optional: Suppress Tensorflow warnings if TF is installed
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel("ERROR")
except ImportError:
    pass


def _get_logger(name: str = "DriftDetector") -> logging.Logger:
    return logging.getLogger(name)


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

    # =========================
    # UNIVARIATE TESTS
    # =========================

    def kolmogorov_smirnov_test(
        self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: float = 0.05
    ):
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
                results[col] = {
                    "statistic": stat,
                    "p_value": p,
                    "alpha": alpha,
                    "drift": d,
                }
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def mann_whitney_test(
        self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: float = 0.05
    ):
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
                results[col] = {
                    "statistic": stat,
                    "p_value": p,
                    "alpha": alpha,
                    "drift": d,
                }
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def population_stability_index_test(
        self,
        X_ref: pd.DataFrame,
        X_new: pd.DataFrame,
        psi_threshold: float = 0.10,
        num_bins: int = 10,
    ):
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
                results[col] = {
                    "psi": psi_val,
                    "threshold": psi_threshold,
                    "num_bins": num_bins,
                    "drift": d,
                }
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def chi_squared_test(
        self, X_ref: pd.DataFrame, X_new: pd.DataFrame, alpha: float = 0.05
    ):
        """Column-wise Chi-squared test; drift if p < alpha. For categorical features."""
        drift_detected, results = False, {}
        det = ChiSquareTest()
        for col in X_ref.columns:
            if col not in X_new.columns:
                continue
            try:
                # For categorical data, we pass the series directly
                x1 = X_ref[col]
                x2 = X_new[col]
                if x1.empty or x2.empty:
                    results[col] = {"error": "empty data", "drift": False}
                    continue
                det.reset()
                det.fit(X=x1)
                res, _ = det.compare(X=x2)
                p = float(res.p_value)
                stat = float(res.statistic)
                d = p < alpha
                results[col] = {
                    "statistic": stat,
                    "p_value": p,
                    "alpha": alpha,
                    "drift": d,
                }
                drift_detected |= d
            except Exception as e:
                results[col] = {"error": str(e), "drift": False}
        return drift_detected, results

    def data_drift_suite(
        self,
        X_ref: pd.DataFrame,
        X_new: pd.DataFrame,
        alpha: float = 0.05,
        psi_threshold: float = 0.10,
        num_bins: int = 10,
    ):
        """Run a suite of univariate data drift tests, adapting to column types."""
        drift_detected, results, flags = False, {}, {}

        for col in X_ref.columns:
            if col not in X_new.columns:
                results[col] = {"error": "column missing in new data", "drift": True}
                flags[col] = True
                drift_detected = True
                continue

            # Determine column type
            if is_numeric_dtype(X_ref[col]):
                # Numerical tests
                d, r = self.kolmogorov_smirnov_test(
                    X_ref[[col]], X_new[[col]], alpha=alpha
                )
                results[f"{col}_ks"] = r[col]
                flags[f"{col}_ks"] = d
                drift_detected |= d

                d, r = self.mann_whitney_test(X_ref[[col]], X_new[[col]], alpha=alpha)
                results[f"{col}_mw"] = r[col]
                flags[f"{col}_mw"] = d
                drift_detected |= d

                d, r = self.population_stability_index_test(
                    X_ref[[col]],
                    X_new[[col]],
                    psi_threshold=psi_threshold,
                    num_bins=num_bins,
                )
                results[f"{col}_psi"] = r[col]
                flags[f"{col}_psi"] = d
                drift_detected |= d
            else:
                # Categorical tests
                d, r = self.chi_squared_test(X_ref[[col]], X_new[[col]], alpha=alpha)
                results[f"{col}_chi2"] = r[col]
                flags[f"{col}_chi2"] = d
                drift_detected |= d

        return drift_detected, results, flags

    def absolute_performance_degradation_suite(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        task: str = "classification",
        balanced_accuracy_threshold: float = 0.9,
        accuracy_threshold: float = 0.9,
        f1_threshold: float = 0.9,
        rmse_threshold: float = 1.2,
        r2_threshold: float = 0.5,
        mae_threshold: float = 0.2,
        mse_threshold: float = 0.4,
        average: str = "macro",
    ):
        """
        Run a suite of absolute performance degradation tests.
        Returns:
            drift_detected (bool), results (dict per metric), flags (dict[metric]=bool)
        """
        results = {}
        flags = {}
        drift_detected = False

        if task == "classification":
            d, r = self.performance_degradation_test_balanced_accuracy(
                X, y, model, balanced_accuracy_threshold
            )
            results["balanced_accuracy"] = r
            flags["balanced_accuracy"] = d
            drift_detected |= d

            d, r = self.performance_degradation_test_accuracy(
                X, y, model, accuracy_threshold
            )
            results["accuracy"] = r
            flags["accuracy"] = d
            drift_detected |= d

            d, r = self.performance_degradation_test_f1(X, y, model, f1_threshold)
            results["F1"] = r
            flags["F1"] = d
            drift_detected |= d

        elif task == "regression":
            d, r = self.performance_degradation_test_rmse(X, y, model, rmse_threshold)
            results["rmse"] = r
            flags["rmse"] = d
            drift_detected |= d

            d, r = self.performance_degradation_test_r2(X, y, model, r2_threshold)
            results["r2"] = r
            flags["r2"] = d
            drift_detected |= d

            d, r = self.performance_degradation_test_mae(X, y, model, mae_threshold)
            results["mae"] = r
            flags["mae"] = d
            drift_detected |= d

            d, r = self.performance_degradation_test_mse(X, y, model, mse_threshold)
            results["mse"] = r
            flags["mse"] = d
            drift_detected |= d
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        # Majority rule: drift if >= 50% of selected metrics flag drift
        n = len(flags)
        drift_detected = n > 0 and sum(bool(v) for v in flags.values()) >= (n / 2.0)
        return drift_detected, results, flags

    def performance_degradation_test_balanced_accuracy(
        self, X: pd.DataFrame, y: pd.Series, model, threshold=0.9
    ):
        """Performance degradation test using balanced accuracy."""
        try:
            predictions = model.predict(X)
            new_acc = balanced_accuracy_score(y, predictions)
            return new_acc < threshold, {
                "balanced_accuracy": new_acc,
                "threshold": threshold,
            }
        except Exception as e:
            return False, {"error": str(e)}

    def performance_degradation_test_accuracy(
        self, X: pd.DataFrame, y: pd.Series, model, threshold=0.9
    ):
        """Performance degradation test using accuracy."""
        try:
            predictions = model.predict(X)
            new_acc = accuracy_score(y, predictions)
            return new_acc < threshold, {"accuracy": new_acc, "threshold": threshold}
        except Exception as e:
            return False, {"error": str(e)}

    def performance_degradation_test_f1(
        self, X: pd.DataFrame, y: pd.Series, model, threshold=0.9
    ):
        """Performance degradation test using F1 score."""
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, average="binary")  # Or 'macro' for multi-class
        return f1 < threshold, {"F1": f1, "threshold": threshold}

    def performance_degradation_test_rmse(
        self, X: pd.DataFrame, y: pd.Series, model, threshold=1.2
    ):
        """Performance degradation test using RMSE (Root Mean Squared Error)."""
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        return rmse > threshold, {"RMSE": rmse, "threshold": threshold}

    def performance_degradation_test_r2(
        self, X: pd.DataFrame, y: pd.Series, model, threshold=0.5
    ):
        """Performance degradation test using R2 score."""
        r2 = r2_score(y, model.predict(X))
        return r2 < threshold, {"R2": r2, "threshold": threshold}

    def performance_degradation_test_mae(
        self, X: pd.DataFrame, y: pd.Series, model, threshold=0.2
    ):
        """Performance degradation test using MAE (Mean Absolute Error)."""
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        return mae > threshold, {"MAE": mae, "threshold": threshold}

    def performance_degradation_test_mse(
        self, X: pd.DataFrame, y: pd.Series, model, threshold=0.4
    ):
        """Performance degradation test using MSE (Mean Squared Error)."""
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return mse > threshold, {"MSE": mse, "threshold": threshold}

    # ===========================
    # Comparative performance tests (previous vs current)
    # Drift if current degrades >= decay_ratio (default 0.30)
    # ===========================
    @staticmethod
    def _safe_rel_drop(
        prev_value: float,
        curr_value: float,
        higher_is_better: bool,
        decay_ratio: float,
        eps: float = 1e-12,
    ):
        """
        Compute relative change (drop or increase) robustly.
        Returns:
            drift (bool), rel_change (float or None)
        rel_change is expressed as a positive fraction (e.g., 0.32 means 32% drop/increase).
        """
        # For metrics where higher is better (accuracy, F1, r2):
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

    def compare_accuracy_drop(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        decay_ratio: float = 0.30,
    ):
        """Drift if accuracy(current) dropped >= decay_ratio vs previous."""
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        acc_prev = accuracy_score(y, y_prev)
        acc_curr = accuracy_score(y, y_curr)
        drift, rel = self._safe_rel_drop(
            acc_prev, acc_curr, higher_is_better=True, decay_ratio=decay_ratio
        )
        return drift, {
            "metric": "accuracy",
            "prev": acc_prev,
            "current": acc_curr,
            "relative_drop": rel,
            "threshold": decay_ratio,
        }

    def compare_balanced_accuracy_drop(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        decay_ratio: float = 0.30,
    ):
        """Drift if balanced_accuracy(current) dropped >= decay_ratio vs previous."""
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        b_prev = balanced_accuracy_score(y, y_prev)
        b_curr = balanced_accuracy_score(y, y_curr)
        drift, rel = self._safe_rel_drop(
            b_prev, b_curr, higher_is_better=True, decay_ratio=decay_ratio
        )
        return drift, {
            "metric": "balanced_accuracy",
            "prev": b_prev,
            "current": b_curr,
            "relative_drop": rel,
            "threshold": decay_ratio,
        }

    def compare_f1_drop(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        decay_ratio: float = 0.30,
        average: str = "macro",
    ):
        """Drift if F1(current) dropped >= decay_ratio vs previous. Use average='macro' (multi-class) or 'binary'."""
        y_prev = model_prev.predict(X)
        y_curr = model_curr.predict(X)
        f_prev = f1_score(y, y_prev, average=average)
        f_curr = f1_score(y, y_curr, average=average)
        drift, rel = self._safe_rel_drop(
            f_prev, f_curr, higher_is_better=True, decay_ratio=decay_ratio
        )
        return drift, {
            "metric": f"F1_{average}",
            "prev": f_prev,
            "current": f_curr,
            "relative_drop": rel,
            "threshold": decay_ratio,
        }

    def compare_r2_drop(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        decay_ratio: float = 0.30,
    ):
        """Drift if R2(current) dropped >= decay_ratio vs previous."""
        r_prev = r2_score(y, model_prev.predict(X))
        r_curr = r2_score(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(
            r_prev, r_curr, higher_is_better=True, decay_ratio=decay_ratio
        )
        return drift, {
            "metric": "r2",
            "prev": r_prev,
            "current": r_curr,
            "relative_drop": rel,
            "threshold": decay_ratio,
        }

    def compare_rmse_increase(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        decay_ratio: float = 0.30,
    ):
        """Drift if RMSE(current) increased >= decay_ratio vs previous."""
        rmse_prev = mean_squared_error(y, model_prev.predict(X), squared=False)
        rmse_curr = mean_squared_error(y, model_curr.predict(X), squared=False)
        drift, rel = self._safe_rel_drop(
            rmse_prev, rmse_curr, higher_is_better=False, decay_ratio=decay_ratio
        )
        return drift, {
            "metric": "rmse",
            "prev": rmse_prev,
            "current": rmse_curr,
            "relative_increase": rel,
            "threshold": decay_ratio,
        }

    def compare_mae_increase(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        decay_ratio: float = 0.30,
    ):
        """Drift if MAE(current) increased >= decay_ratio vs previous."""
        from sklearn.metrics import mean_absolute_error

        mae_prev = mean_absolute_error(y, model_prev.predict(X))
        mae_curr = mean_absolute_error(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(
            mae_prev, mae_curr, higher_is_better=False, decay_ratio=decay_ratio
        )
        return drift, {
            "metric": "mae",
            "prev": mae_prev,
            "current": mae_curr,
            "relative_increase": rel,
            "threshold": decay_ratio,
        }

    def compare_mse_increase(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        decay_ratio: float = 0.30,
    ):
        """Drift if MSE(current) increased >= decay_ratio vs previous."""
        mse_prev = mean_squared_error(y, model_prev.predict(X))
        mse_curr = mean_squared_error(y, model_curr.predict(X))
        drift, rel = self._safe_rel_drop(
            mse_prev, mse_curr, higher_is_better=False, decay_ratio=decay_ratio
        )
        return drift, {
            "metric": "mse",
            "prev": mse_prev,
            "current": mse_curr,
            "relative_increase": rel,
            "threshold": decay_ratio,
        }

    def performance_comparison_suite(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_prev,
        model_curr,
        task: str = "classification",
        decay_ratio: float = 0.30,
        average: str = "macro",
    ):
        """
        Run a set of comparative tests (previous vs current).
        Returns:
            drift_detected (bool), results (dict per metric), flags (dict[metric]=bool)
        """
        results = {}
        flags = {}

        if task == "classification":
            d, r = self.compare_accuracy_drop(X, y, model_prev, model_curr, decay_ratio)
            results["accuracy"] = r
            flags["accuracy"] = d
            d, r = self.compare_balanced_accuracy_drop(
                X, y, model_prev, model_curr, decay_ratio
            )
            results["balanced_accuracy"] = r
            flags["balanced_accuracy"] = d
            d, r = self.compare_f1_drop(
                X, y, model_prev, model_curr, decay_ratio, average=average
            )
            results[f"F1_{average}"] = r
            flags["f1"] = d
        elif task == "regression":
            d, r = self.compare_r2_drop(X, y, model_prev, model_curr, decay_ratio)
            results["r2"] = r
            flags["r2"] = d
            d, r = self.compare_rmse_increase(X, y, model_prev, model_curr, decay_ratio)
            results["rmse"] = r
            flags["rmse"] = d
            d, r = self.compare_mae_increase(X, y, model_prev, model_curr, decay_ratio)
            results["mae"] = r
            flags["mae"] = d
            d, r = self.compare_mse_increase(X, y, model_prev, model_curr, decay_ratio)
            results["mse"] = r
            flags["mse"] = d
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        # Majority rule: drift if >= 50% of selected metrics flag drift
        n = len(flags)
        drift_detected = n > 0 and sum(bool(v) for v in flags.values()) >= (n / 2.0)
        return drift_detected, results, flags
