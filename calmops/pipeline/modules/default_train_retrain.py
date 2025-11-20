import joblib
import os
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    classification_report, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import is_classifier, is_regressor
from scikeras.wrappers import KerasClassifier
from torch.nn import Module
from sklearn.base import is_classifier, is_regressor, clone
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
def _merge_train_results_retrain(results_base: dict, extra_info: dict) -> dict:
    # We do not allow default_train to overwrite retrain tags
    extra_info = dict(extra_info)  # defensive copy
    extra_info.pop("type", None)   # <- key of the problem
    # (Optional) prevents 'file' from changing it if you don't want to
    # extra_info.pop("file", None)

    results_base.update(extra_info)
    results_base["type"] = "retrain"  # we ensure the correct label
    return results_base

def save_train_results(results: dict, output_dir: str, logger=None):
    """Save training results in JSON format."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_results.json")

    def make_serializable(obj):
        """Convert non-serializable objects into serializable types."""
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray, list)): return [make_serializable(x) for x in obj]
        if isinstance(obj, dict): return {kk: make_serializable(vv) for kk, vv in obj.items()}
        return obj

    serializable_results = make_serializable(results)
    with open(train_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    if logger:
        logger.info(f"Training results saved at {train_path}")


def prepare_X_y(model, X, y):
    """Prepare data for models like Skorch or PyTorchNet."""
    if isinstance(model, Module) or "skorch" in str(type(model)).lower():
        X = X.values if hasattr(X, "values") else np.array(X)
        y = y.values if hasattr(y, "values") else np.array(y)
    return X, y


def calculate_metrics(model, X_train, y_train):
    """Calculate and return metrics based on the model type (classifier or regressor)."""
    y_pred = model.predict(X_train)
    metrics = {}

    if is_classifier(model):
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_train, y_pred)
        metrics["classification_report"] = classification_report(
            y_train, y_pred, output_dict=True
        )
    elif is_regressor(model):
        metrics.update({
            "R2": r2_score(y_train, y_pred),
            "RMSE": mean_squared_error(y_train, y_pred, squared=False),
            "MAE": mean_absolute_error(y_train, y_pred),
            "MSE": mean_squared_error(y_train, y_pred),
        })

    return metrics


def default_train(X, y, last_processed_file, model_instance, random_state, logger, output_dir, param_grid=None, cv=None):
    """Train the model and return the trained model and evaluation metrics."""
    logger.info("Starting default training process")

    # Split data
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=random_state)
    if "torch" in str(type(model_instance)).lower() or "skorch" in str(type(model_instance)).lower():
        X_train = X_train.astype(np.float32)
        X_eval = X_eval.astype(np.float32)
        y_train = y_train.astype(np.int64)  # classification
        y_eval = y_eval.astype(np.int64)
        X_train, y_train = prepare_X_y(model_instance, X_train, y_train)
        X_eval, y_eval = prepare_X_y(model_instance, X_eval, y_eval)

    grid_info = None
    try:
        # --- If using GridSearch ---
        if param_grid:
            grid_search = GridSearchCV(model_instance, param_grid, cv=cv, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            grid_info = {"best_params": grid_search.best_params_, "cv": cv}
        else:
            model = model_instance
            model.fit(X_train, y_train)

        # Metrics
        metrics = calculate_metrics(model, X_train, y_train)

        # Results
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "train",
            "model": str(model),
            "gridsearch": grid_info if grid_info else "Not used",
            "file": last_processed_file,
            **metrics
        }

        save_train_results(results, output_dir, logger)
        return model, X_eval, y_eval, results

    except Exception as e:
        logger.error(f"Training process failed: {e}")
        raise


def default_retrain(
    X, y, last_processed_file, model_path, mode, random_state, logger,
    output_dir, param_grid=None, cv=None, window_size=None,
    replay_frac_old: float = 0.4,  # for mode 5
):
    """Retrain the model based on the given mode (0..6)."""
    logger.info(f"Starting retraining process with mode {mode}")
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "retrain",
        "file": last_processed_file,
        "mode": mode,
        "fallback": False,
    }

    def _safe_train_return(_model, _X_train, _y_train, _X_eval, _y_eval, extra_info=None):
        metrics = calculate_metrics(_model, _X_train, _y_train)
        results.update({
            "model": str(type(_model).__name__),
            "gridsearch": "Not used",
            **(extra_info or {}),
            **metrics
        })
        save_train_results(results, output_dir, logger)
        return _model, _X_eval, _y_eval, results

    def _split(X_, y_):
        try:
            base = joblib.load(model_path)
            strat = y_ if is_classifier(base) else None
        except Exception:
            strat = None
        return train_test_split(X_, y_, test_size=0.2, random_state=random_state, stratify=strat)

    try:
        # === Mode 0: Full retraining ===
        if mode == 0:
            results["strategy"] = "full retraining"
            model, X_eval, y_eval, extra_info = default_train(
                X, y, last_processed_file,
                model_instance=joblib.load(model_path),
                random_state=random_state,
                logger=logger,
                output_dir=output_dir,
                param_grid=param_grid,
                cv=cv
            )
            results = _merge_train_results_retrain(results, extra_info)
            save_train_results(results, output_dir, logger)
            return model, X_eval, y_eval, results

        # === Mode 1: Incremental (partial_fit) ===
        elif mode == 1:
            base = joblib.load(model_path)
            results["strategy"] = "incremental"
            inc = base
            if hasattr(base, "steps"):  # sklearn Pipeline
                inc = base.steps[-1][1]

            if hasattr(inc, "partial_fit"):
                X_train, X_eval, y_train, y_eval = _split(X, y)
                if is_classifier(inc):
                    classes = np.unique(y_train)
                    try:
                        inc.partial_fit(X_train, y_train, classes=classes)
                    except TypeError:
                        inc.partial_fit(X_train, y_train)
                else:
                    inc.partial_fit(X_train, y_train)
                model = base  # if pipeline, step is updated in place
                return _safe_train_return(model, X_train, y_train, X_eval, y_eval)
            else:
                results["strategy"] = "fallback full retraining"
                results["fallback"] = True
                model, X_eval, y_eval, extra_info = default_train(
                    X, y, last_processed_file,
                    model_instance=base,
                    random_state=random_state,
                    logger=logger,
                    output_dir=output_dir,
                    param_grid=param_grid,
                    cv=cv
                )
                results = _merge_train_results_retrain(results, extra_info)
                save_train_results(results, output_dir, logger)
                return model, X_eval, y_eval, results

        # === Mode 2: Retrain with window (last N samples) ===
        elif mode == 2:
            if not window_size or window_size <= 0:
                raise ValueError("window_size must be a positive integer for mode 2.")
            results["strategy"] = f"retrain with window (last {window_size} samples)"
            X_window = X.iloc[-window_size:] if window_size < len(X) else X
            y_window = y.iloc[-window_size:] if window_size < len(y) else y

            model, X_eval, y_eval, extra_info = default_train(
                X_window, y_window, last_processed_file,
                model_instance=joblib.load(model_path),
                random_state=random_state,
                logger=logger,
                output_dir=output_dir,
                param_grid=param_grid,
                cv=cv
            )
            results = _merge_train_results_retrain(results, extra_info)
            save_train_results(results, output_dir, logger)
            return model, X_eval, y_eval, results



        # === Mode 4: Stacking (old + cloned(old)) ===
        elif mode == 4:
            results["strategy"] = "stacking old + cloned(old)"
            old_model = joblib.load(model_path)
            new_model = clone(old_model)
            new_model.fit(X, y)

            if is_classifier(old_model):
                final_est = LogisticRegression(max_iter=200)
                model = StackingClassifier(
                    estimators=[("old", old_model), ("new", new_model)],
                    final_estimator=final_est,
                    passthrough=False,
                    cv=5
                )
            elif is_regressor(old_model):
                model = StackingRegressor(
                    estimators=[("old", old_model), ("new", new_model)],
                    final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0)),
                    passthrough=False,
                    cv=5
                )
            else:
                raise ValueError("The model is neither a classifier nor a regressor")

            X_train, X_eval, y_train, y_eval = _split(X, y)
            model.fit(X_train, y_train)
            return _safe_train_return(model, X_train, y_train, X_eval, y_eval)

        # === Mode 5: Replay mix (blend previous_data.csv with current) ===
        elif mode == 5:
            results["strategy"] = f"replay mix (prev fraction={replay_frac_old:.2f})"
            control_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), "control")
            prev_csv = os.path.join(control_dir, "previous_data.csv")
            if not os.path.exists(prev_csv):
                logger.warning("Previous training data not found; falling back to full retraining on current data")
                return default_retrain(
                    X, y, last_processed_file, model_path, 0, random_state, logger,
                    output_dir, param_grid=param_grid, cv=cv, window_size=window_size
                )

            try:
                df_prev = pd.read_csv(prev_csv)
                tgt = y.name if (hasattr(y, "name") and y.name in df_prev.columns) else None
                if tgt is None:
                    raise ValueError("Target column not found in previous_data.csv for replay mix.")
                X_prev = df_prev.drop(columns=[tgt])
                y_prev = df_prev[tgt]

                n_old = int(len(X) * replay_frac_old)
                n_old = max(0, min(n_old, len(X_prev)))
                X_old_samp = X_prev.sample(n=n_old, random_state=random_state)
                y_old_samp = y_prev.loc[X_old_samp.index]

                X_blend = pd.concat([X_old_samp, X], axis=0)
                y_blend = pd.concat([y_old_samp, y], axis=0)

                model, X_eval, y_eval, extra_info = default_train(
                    X_blend, y_blend, last_processed_file,
                    model_instance=joblib.load(model_path),
                    random_state=random_state,
                    logger=logger,
                    output_dir=output_dir,
                    param_grid=param_grid,
                    cv=cv
                )
                results = _merge_train_results_retrain(results, extra_info)
                save_train_results(results, output_dir, logger)
                return model, X_eval, y_eval, results
            except Exception as e:
                logger.error(f"Replay mix failed: {e}")
                results["fallback"] = True
                if window_size and window_size > 0:
                    return default_retrain(
                        X, y, last_processed_file, model_path, 2, random_state, logger,
                        output_dir, param_grid=param_grid, cv=cv, window_size=window_size
                    )
                return default_retrain(
                    X, y, last_processed_file, model_path, 0, random_state, logger,
                    output_dir, param_grid=param_grid, cv=cv
                )

        # === Mode 6: Recalibration only (classification) ===
        elif mode == 6:
            base = joblib.load(model_path)
            if not is_classifier(base):
                raise ValueError("Mode 6 (recalibration) is only valid for classification models.")

            results["strategy"] = "probability calibration (Platt/Isotonic) with base prefit"
            X_cal, X_eval, y_cal, y_eval = _split(X, y)

            def _make_calibrator(_base, method):
                try:
                    return CalibratedClassifierCV(estimator=_base, method=method, cv="prefit")
                except TypeError:
                    return CalibratedClassifierCV(base_estimator=_base, method=method, cv="prefit")

            try:
                calibrator = _make_calibrator(base, method="sigmoid")
                calibrator.fit(X_cal, y_cal)
                return _safe_train_return(
                    calibrator, X_cal, y_cal, X_eval, y_eval,
                    extra_info={"calibration": "sigmoid"}
                )
            except Exception as e:
                logger.warning(f"Sigmoid calibration failed ({e}); trying isotonic.")
                calibrator = _make_calibrator(base, method="isotonic")
                calibrator.fit(X_cal, y_cal)
                return _safe_train_return(
                    calibrator, X_cal, y_cal, X_eval, y_eval,
                    extra_info={"calibration": "isotonic"}
                )

        else:
            results["strategy"] = "invalid_mode"
            results["error"] = f"Unknown mode {mode}"
            save_train_results(results, output_dir, logger)
            raise ValueError(f"Unknown mode {mode}")

    except Exception as e:
        results["error"] = str(e)
        results["strategy"] = results.get("strategy", "error")
        results["fallback"] = True
        save_train_results(results, output_dir, logger)
        raise
