# modules/default_train_retrain.py
from __future__ import annotations

import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone, is_classifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error
)


# =====================================================================================
# Utils
# =====================================================================================

def _serializable(obj):
    import numpy as _np
    if isinstance(obj, (bool, _np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, _np.integer)):
        return int(obj)
    if isinstance(obj, (float, _np.floating)):
        return float(obj)
    if isinstance(obj, (list, tuple, _np.ndarray)):
        return [_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    return obj


def _sorted_blocks(values: pd.Series) -> List[str]:
    vals = values.dropna().astype(str).unique().tolist()
    try:
        nums = [float(v) for v in vals]
        return [x for _, x in sorted(zip(nums, vals))]
    except Exception:
        pass
    try:
        dt = pd.to_datetime(vals, errors="raise")
        return [x for _, x in sorted(zip(dt, vals))]
    except Exception:
        pass
    return sorted(vals, key=lambda x: str(x))


def _global_metrics(y_true: pd.Series, y_pred: np.ndarray, model) -> Dict[str, float]:
    # IMPORTANT: detect using underlying estimator (global_model) if present
    base_est = getattr(model, "global_model", model)
    if is_classifier(base_est):
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro"),
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        out = {
            "r2": r2_score(y_true, y_pred),
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": float(mse),
        }
    return {k: float(v) for k, v in out.items()}


def _per_block_metrics(y_true: pd.Series, y_pred: np.ndarray, blocks: pd.Series, model) -> Dict[str, Dict[str, float]]:
    # Detect kind on underlying estimator
    base_est = getattr(model, "global_model", model)
    is_cls = is_classifier(base_est)

    df = pd.DataFrame({"y": y_true, "pred": y_pred, "_b": blocks}).dropna(subset=["_b"])
    out: Dict[str, Dict[str, float]] = {}
    for bid, g in df.groupby("_b"):
        if is_cls:
            out[str(bid)] = {
                "accuracy": float(accuracy_score(g["y"], g["pred"])),
                "balanced_accuracy": float(balanced_accuracy_score(g["y"], g["pred"])),
                "f1": float(f1_score(g["y"], g["pred"], average="macro")),
                "n": int(g.shape[0]),
            }
        else:
            mse = mean_squared_error(g["y"], g["pred"])
            out[str(bid)] = {
                "r2": float(r2_score(g["y"], g["pred"])),
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(g["y"], g["pred"])),
                "mse": float(mse),
                "n": int(g.shape[0]),
            }
    return out


# =====================================================================================
# Block-wise model
# =====================================================================================

class BlockWiseModel:
    """
    - global_model: trained with all TRAIN blocks
    - per_block_models: dict {block_id -> model trained only with that block}
    """
    def __init__(self, base_estimator, block_col: str, fallback_strategy: str = 'global'):
        self.base_estimator = base_estimator
        self.block_col = block_col
        self.fallback_strategy = fallback_strategy
        self.global_model = None
        self.per_block_models: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, train_blocks: List[str], *, min_rows_per_block: int = 1):
        # Train global model with all train blocks
        mask = X[self.block_col].astype(str).isin([str(b) for b in train_blocks])
        Xg, yg = X.loc[mask].drop(columns=[self.block_col], errors="ignore"), y.loc[mask]
        if Xg.shape[0] == 0:
            raise RuntimeError("No rows available to train the global model. Check train_blocks selection.")
        self.global_model = clone(self.base_estimator)
        self.global_model.fit(Xg, yg)

        # Per-block models (skip tiny blocks)
        self.per_block_models = {}
        skipped_small: List[str] = []
        for b in train_blocks:
            Xb_full = X[X[self.block_col].astype(str) == str(b)]
            if Xb_full.shape[0] < int(min_rows_per_block):
                skipped_small.append(str(b))
                continue
            Xb = Xb_full.drop(columns=[self.block_col], errors="ignore")
            yb = y.loc[Xb.index]
            est = clone(self.base_estimator)
            est.fit(Xb, yb)
            self.per_block_models[str(b)] = est

        # Keep info for reporting usage (optionally)
        self._skipped_small_on_fit = skipped_small
        return self

    def retrain_blocks(self, X: pd.DataFrame, y: pd.Series, blocks_to_retrain: List[str], *, min_rows_per_block: int = 1):
        """Retrains ONLY the per-block models listed (does not touch global)."""
        skipped_small: List[str] = []
        for b in blocks_to_retrain:
            Xb_full = X[X[self.block_col].astype(str) == str(b)]
            if Xb_full.shape[0] < int(min_rows_per_block):
                skipped_small.append(str(b))
                continue
            Xb = Xb_full.drop(columns=[self.block_col], errors="ignore")
            yb = y.loc[Xb.index]
            est = clone(self.base_estimator)
            est.fit(Xb, yb)
            self.per_block_models[str(b)] = est

        self._skipped_small_on_retrain = skipped_small
        return self

    def set_block_model(self, block_id: str, model) -> None:
        self.per_block_models[str(block_id)] = model

    def get_block_model(self, block_id: str):
        return self.per_block_models.get(str(block_id), None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_model is None:
            raise RuntimeError("Model not fitted.")

        # Predict per block when available; fallback to global or ensemble
        if self.block_col in X.columns:
            out = np.empty(len(X), dtype=object)
            blocks = X[self.block_col].astype(str)
            base_index = X.index
            for b, idx_labels in blocks.groupby(blocks).groups.items():
                idx_labels = pd.Index(idx_labels)
                pos = base_index.get_indexer(idx_labels)
                Xb = X.loc[idx_labels].drop(columns=[self.block_col], errors="ignore")

                model_to_use = self.per_block_models.get(str(b))
                yb = None

                if model_to_use:
                    yb = model_to_use.predict(Xb)
                else:
                    # Fallback logic
                    if self.fallback_strategy == 'ensemble' and self.per_block_models:
                        all_preds = [m.predict(Xb) for m in self.per_block_models.values()]
                        
                        base_est = self.global_model or self.base_estimator
                        if is_classifier(base_est):
                            # Majority vote for classification
                            from scipy.stats import mode
                            preds_array = np.array(all_preds).T
                            yb, _ = mode(preds_array, axis=1)
                            yb = yb.ravel()
                        else:
                            # Average for regression
                            yb = np.mean(all_preds, axis=0)
                    
                    # If ensemble fails or strategy is 'global', use global model
                    if yb is None:
                        yb = self.global_model.predict(Xb)
                
                out[pos] = yb

            try:
                return out.astype(float)
            except Exception:
                return out

        return self.global_model.predict(X)


# =====================================================================================
# Helpers for training / per-block cache
# =====================================================================================

def _fit_with_optional_gs(base_model, Xtr, ytr, param_grid=None, cv=None, logger=None):
    model = clone(base_model)
    if param_grid and cv:
        scoring = "f1_macro" if is_classifier(model) else "r2"
        gs = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=scoring, refit=True)
        gs.fit(Xtr, ytr)
        best_model = gs.best_estimator_
        if logger:
            logger.info(f"[TRAIN] GridSearchCV best params: {gs.best_params_}")
        return best_model, {"gridsearch": {"best_params": gs.best_params_, "cv": cv}}
    else:
        model.fit(Xtr, ytr)
        return model, {}


def _save_block_training_data(control_dir: Optional[str], block_id: str, Xb: pd.DataFrame, yb: pd.Series):
    if not control_dir:
        return
    os.makedirs(os.path.join(control_dir, "per_block_train_data"), exist_ok=True)
    dfb = Xb.copy()
    dfb["__target__"] = yb
    dfb.to_parquet(os.path.join(control_dir, "per_block_train_data", f"{str(block_id)}.parquet"), index=True)


def _load_block_training_data(control_dir: Optional[str], block_id: str) -> Optional[pd.DataFrame]:
    if not control_dir:
        return None
    path = os.path.join(control_dir, "per_block_train_data", f"{str(block_id)}.parquet")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# =====================================================================================
# Report
# =====================================================================================

def _train_report(
    *,
    kind: str,
    file: str,
    model: BlockWiseModel,
    block_col: str,
    train_blocks: List[str],
    eval_blocks: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    extra: Dict[str, Any] | None = None,
    X_train_data: Optional[pd.DataFrame] = None,
    y_train_data: Optional[pd.Series] = None,
    X_eval_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    report: Dict[str, Any] = {
        "type": kind,
        "file": file,
        "timestamp": ts,
        "model": model.base_estimator.__class__.__name__,
        "fallback_strategy": model.fallback_strategy,
        "block_col": block_col,
        "train_blocks": [str(b) for b in train_blocks],
        "eval_blocks": [str(b) for b in eval_blocks],
    }

    # Use specific train/eval data if provided
    X_tr = X_train_data if X_train_data is not None else X[X[block_col].astype(str).isin(train_blocks)]
    y_tr = y_train_data if y_train_data is not None else y.loc[X_tr.index]

    report["train_size"] = int(X_tr.shape[0])
    if X_eval_data is not None:
        report["eval_size"] = int(X_eval_data.shape[0])
    else:
        report["eval_size"] = int(X[X[block_col].astype(str).isin(eval_blocks)].shape[0])

    # In-sample metrics per train block
    per_block_train: Dict[str, Dict[str, float]] = {}
    for b in train_blocks:
        Xb = X_tr[X_tr[block_col].astype(str) == str(b)]
        if Xb.empty:
            continue
        yb = y_tr.loc[Xb.index]
        yhat = model.predict(Xb)
        per_block_train[str(b)] = _global_metrics(yb, yhat, model)
        per_block_train[str(b)]["n"] = int(Xb.shape[0])
    report["per_block_train"] = per_block_train

    # If we tracked skips on fit/retrain, add them
    if hasattr(model, "_skipped_small_on_fit"):
        report["skipped_small_blocks_on_fit"] = list(getattr(model, "_skipped_small_on_fit") or [])
    if hasattr(model, "_skipped_small_on_retrain"):
        report["skipped_small_blocks_on_retrain"] = list(getattr(model, "_skipped_small_on_retrain") or [])

    if extra:
        report.update(extra)
    return _serializable(report)


# =====================================================================================
# TRAIN (block_wise only)
# =====================================================================================

def default_train(
    X: pd.DataFrame,
    y: pd.Series,
    last_processed_file: str,
    *,
    model_instance,
    random_state: int,
    logger,
    param_grid: dict = None,
    cv: int = None,
    output_dir: str = None,
    blocks: Optional[pd.Series] = None,
    block_col: Optional[str] = None,
    # only using `test_blocks` to define eval blocks; others kept for API compatibility
    split_mode: Optional[str] = None,
    test_size: Optional[float] = None,
    n_test_blocks: Optional[int] = None,
    test_blocks: Optional[List[str]] = None,
    block_selection: Optional[str] = None,
    # NEW: per-block cache (replay mode) + small-block threshold
    control_dir: Optional[str] = None,
    min_rows_per_block: int = 1,          # skip training per-block models smaller than this
    auto_init_new_blocks: bool = True,    # not used on train (all seen are "current"), kept for symmetry
    split_within_blocks: bool = False,
    train_percentage: float = 0.8,
    fallback_strategy: str = 'global',
) -> Tuple[BlockWiseModel, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    TRAIN (BLOCK_WISE ONLY):
      - eval_blocks = test_blocks if provided; otherwise last block
      - train_blocks = the rest; if empty (1-block dataset), use the same block also as train (safe fallback)
      - trains global + per-block (skips tiny blocks < min_rows_per_block)
      - stores per-block cache for replay
    """
    assert block_col is not None, "block_col is required in block_wise mode."
    assert blocks is not None and blocks.size > 0, "A blocks Series is required."

    os.makedirs(output_dir or ".", exist_ok=True)
    train_json = os.path.join(output_dir or ".", "train_results.json")

    all_blocks_sorted = _sorted_blocks(blocks.astype(str))
    if not all_blocks_sorted:
        raise RuntimeError("No block ids found.")

    # Initialize report_params
    report_params = {
        "X": X,
        "y": y,
    }

    if split_within_blocks:
        logger.info(f"Splitting within blocks for training and evaluation ({train_percentage * 100}% train)." )
        train_indices = []
        eval_indices = []
        for block_id in all_blocks_sorted:
            block_indices = X[X[block_col].astype(str) == block_id].index
            n_samples = len(block_indices)
            n_train = int(n_samples * train_percentage)
            # Shuffle indices before splitting to avoid ordered data issues
            shuffled_block_indices = np.random.permutation(block_indices)
            train_indices.extend(shuffled_block_indices[:n_train])
            eval_indices.extend(shuffled_block_indices[n_train:])

        X_train_data = X.loc[train_indices]
        y_train_data = y.loc[train_indices]
        X_eval = X.loc[eval_indices]
        y_eval = y.loc[eval_indices]

        train_blocks = all_blocks_sorted
        eval_blocks = all_blocks_sorted

        # Update report_params for split mode
        report_params["X_train_data"] = X_train_data
        report_params["y_train_data"] = y_train_data
        report_params["X_eval_data"] = X_eval


        # Best base estimator (optional GridSearch) with all TRAIN rows
        Xg = X_train_data.drop(columns=[block_col], errors="ignore")
        yg = y_train_data
        best_est, extra = _fit_with_optional_gs(model_instance, Xg, yg, param_grid=param_grid, cv=cv, logger=logger)

        # Block-wise model trained on the training split of the data
        model = BlockWiseModel(best_est, block_col=block_col, fallback_strategy=fallback_strategy).fit(X_train_data, y_train_data, train_blocks=train_blocks, min_rows_per_block=min_rows_per_block)

    else:
        if test_blocks:
            eval_blocks = [str(b) for b in test_blocks if str(b) in all_blocks_sorted]
            if not eval_blocks:
                eval_blocks = [all_blocks_sorted[-1]]
        else:
            eval_blocks = [all_blocks_sorted[-1]]

        train_blocks = [b for b in all_blocks_sorted if b not in set(eval_blocks)]
        if len(train_blocks) == 0:
            # Safe fallback for single-block datasets
            train_blocks = list(eval_blocks)

        # Best base estimator (optional GridSearch) with all TRAIN rows
        Xg = X[X[block_col].astype(str).isin(train_blocks)].drop(columns=[block_col], errors="ignore")
        yg = y.loc[Xg.index]
        best_est, extra = _fit_with_optional_gs(model_instance, Xg, yg, param_grid=param_grid, cv=cv, logger=logger)

        # Block-wise model
        model = BlockWiseModel(best_est, block_col=block_col, fallback_strategy=fallback_strategy).fit(X, y, train_blocks=train_blocks, min_rows_per_block=min_rows_per_block)

        # Eval subset
        eval_mask = X[block_col].astype(str).isin(eval_blocks)
        X_eval = X.loc[eval_mask]
        y_eval = y.loc[eval_mask]

    # Save per-block training cache (for replay)
    for b in train_blocks:
        Xb_full = X[X[block_col].astype(str) == str(b)].drop(columns=[block_col], errors="ignore")
        yb_full = y.loc[Xb_full.index]
        if Xb_full.shape[0] >= int(min_rows_per_block):
            _save_block_training_data(control_dir, b, Xb_full, yb_full)

    # Report
    report = _train_report(
        kind="train",
        file=last_processed_file,
        model=model,
        block_col=block_col,
        train_blocks=train_blocks,
        eval_blocks=eval_blocks,
        extra=extra if 'extra' in locals() else {},
        **report_params
    )
    try:
        with open(train_json, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

    meta = {"test_blocks": pd.Series(X_eval[block_col]) if block_col in X_eval else None}

    return model, X_eval, y_eval, meta


# =====================================================================================
# RETRAIN (selective block-wise with modes 0–6)
# =====================================================================================

class FrozenEstimator:
    """Wrapper to freeze a fitted estimator (fit is a no-op)."""
    def __init__(self, fitted_estimator):
        self.est = fitted_estimator

    def fit(self, X, y):
        return self  # no-op

    def predict(self, X):
        return self.est.predict(X)

    def predict_proba(self, X):
        if hasattr(self.est, "predict_proba"):
            return self.est.predict_proba(X)
        raise AttributeError("Underlying estimator has no predict_proba")


def _retrain_block_strategy(
    *,
    mode: int,
    prev_model,
    base_estimator,
    Xb: pd.DataFrame,
    yb: pd.Series,
    random_state: int,
    is_clf: bool,
    control_dir: Optional[str],
    block_id: str
):
    """
    Returns the NEW per-block model according to 'mode'.
    """
    # 0) full
    if mode == 0:
        new_model = clone(base_estimator)
        new_model.fit(Xb, yb)
        return new_model

    # 1) incremental (partial_fit)
    if mode == 1 and hasattr(prev_model, "partial_fit"):
        try:
            if is_clf:
                classes = np.unique(yb)
                prev_model.partial_fit(Xb, yb, classes=classes)
            else:
                prev_model.partial_fit(Xb, yb)
            return prev_model
        except Exception:
            # fallback to full if partial_fit fails
            new_model = clone(base_estimator)
            new_model.fit(Xb, yb)
            return new_model

    # 2) windowed: Xb already trimmed outside if needed
    if mode == 2:
        new_model = clone(base_estimator)
        new_model.fit(Xb, yb)
        return new_model

    # 3) ensemble old + new (stacking)
    if mode == 3:
        from sklearn.linear_model import LogisticRegression, Ridge
        if is_clf:
            from sklearn.ensemble import StackingClassifier
            new_est = clone(base_estimator)
            new_est.fit(Xb, yb)
            estimators = [("old", clone(prev_model)), ("new", new_est)]
            meta = LogisticRegression(max_iter=1000)
            model = StackingClassifier(estimators=estimators, final_estimator=meta, n_jobs=-1)
            model.fit(Xb, yb)
            return model
        else:
            from sklearn.ensemble import StackingRegressor
            new_est = clone(base_estimator)
            new_est.fit(Xb, yb)
            estimators = [("old", clone(prev_model)), ("new", new_est)]
            meta = Ridge(random_state=random_state)
            model = StackingRegressor(estimators=estimators, final_estimator=meta, n_jobs=-1)
            model.fit(Xb, yb)
            return model

    # 4) stacking frozen(old) + cloned(old) retrained
    if mode == 4:
        from sklearn.linear_model import LogisticRegression, Ridge
        if is_clf:
            from sklearn.ensemble import StackingClassifier
            frozen_old = FrozenEstimator(prev_model)
            cloned_old = clone(prev_model)
            cloned_old.fit(Xb, yb)
            estimators = [("old_frozen", frozen_old), ("old_clone", cloned_old)]
            meta = LogisticRegression(max_iter=1000)
            model = StackingClassifier(estimators=estimators, final_estimator=meta, n_jobs=-1, passthrough=False)
            model.fit(Xb, yb)
            return model
        else:
            from sklearn.ensemble import StackingRegressor
            frozen_old = FrozenEstimator(prev_model)
            cloned_old = clone(prev_model)
            cloned_old.fit(Xb, yb)
            estimators = [("old_frozen", frozen_old), ("old_clone", cloned_old)]
            meta = Ridge(random_state=random_state)
            model = StackingRegressor(estimators=estimators, final_estimator=meta, n_jobs=-1, passthrough=False)
            model.fit(Xb, yb)
            return model

    # 5) replay mix (previous cached training data + current block data)
    if mode == 5:
        df_prev = _load_block_training_data(control_dir, block_id)
        if df_prev is not None and "__target__" in df_prev.columns:
            X_prev = df_prev.drop(columns=["__target__"])
            y_prev = df_prev["__target__"]
            X_mix = pd.concat([X_prev, Xb], axis=0)
            y_mix = pd.concat([y_prev, yb], axis=0)
        else:
            X_mix, y_mix = Xb, yb
        new_model = clone(base_estimator)
        new_model.fit(X_mix, y_mix)
        return new_model

    # 6) calibration (classification)
    if mode == 6 and is_clf:
        from sklearn.calibration import CalibratedClassifierCV
        try:
            calib = CalibratedClassifierCV(base_estimator=prev_model, method="isotonic", cv=3)
            calib.fit(Xb, yb)
            return calib
        except Exception:
            # fallback to full
            new_model = clone(base_estimator)
            new_model.fit(Xb, yb)
            return new_model

    # fallback → full
    new_model = clone(base_estimator)
    new_model.fit(Xb, yb)
    return new_model

def default_retrain(
    X: pd.DataFrame,
    y: pd.Series,
    last_processed_file: str,
    *,
    model_path: str,
    mode: int,
    random_state: int,
    logger,
    param_grid: dict = None,     # unused in block-wise selective retrain
    cv: int = None,              # unused in block-wise selective retrain
    output_dir: str = None,
    window_size: int = None,
    blocks: Optional[pd.Series] = None,
    block_col: Optional[str] = None,
    split_mode: Optional[str] = None,         # ignored
    test_size: Optional[float] = None,        # ignored
    n_test_blocks: Optional[int] = None,      # ignored
    test_blocks: Optional[List[str]] = None,  # eval blocks
    block_selection: Optional[str] = None,    # ignored
    drifted_blocks: Optional[List[str]] = None,
    control_dir: Optional[str] = None,
    # NEW
    min_rows_per_block: int = 1,          # do not train per-block if block has fewer rows
    auto_init_new_blocks: bool = True,    # automatically create models for brand-new blocks
    split_within_blocks: bool = False,
    train_percentage: float = 0.8,
    fallback_strategy: str = 'global',
) -> Tuple[BlockWiseModel, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    SELECTIVE RETRAIN (BLOCK_WISE):
      - Retrains only `drifted_blocks` using the selected mode.
      - If `auto_init_new_blocks` is True, also initializes models for brand-new blocks.
      - Does NOT touch the global model.
      - Updates per-block training cache when applicable (replay).
    """
    assert block_col is not None, "block_col is required in block_wise mode."
    assert blocks is not None and blocks.size > 0, "A blocks Series is required."
    import joblib

    os.makedirs(output_dir or ".", exist_ok=True)
    train_json = os.path.join(output_dir or ".", "train_results.json")

    model: BlockWiseModel = joblib.load(model_path)
    model.fallback_strategy = fallback_strategy # Update strategy on loaded model
    all_blocks_sorted = _sorted_blocks(blocks.astype(str))

    report_params = {
        "X": X,
        "y": y,
    }

    if split_within_blocks:
        logger.info(f"Splitting within blocks for retraining and evaluation ({train_percentage * 100}% train).")
        train_indices = []
        eval_indices = []
        for block_id in all_blocks_sorted:
            block_indices = X[X[block_col].astype(str) == block_id].index
            n_samples = len(block_indices)
            n_train = int(n_samples * train_percentage)
            shuffled_block_indices = np.random.permutation(block_indices)
            train_indices.extend(shuffled_block_indices[:n_train])
            eval_indices.extend(shuffled_block_indices[n_train:])

        X_train_data = X.loc[train_indices]
        y_train_data = y.loc[train_indices]
        X_eval = X.loc[eval_indices]
        y_eval = y.loc[eval_indices]
        train_blocks = all_blocks_sorted
        eval_blocks = all_blocks_sorted

        report_params["X_train_data"] = X_train_data
        report_params["y_train_data"] = y_train_data
        report_params["X_eval_data"] = X_eval

    else:
        if test_blocks:
            eval_blocks = [str(b) for b in test_blocks if str(b) in all_blocks_sorted]
            if not eval_blocks:
                eval_blocks = [all_blocks_sorted[-1]]
        else:
            eval_blocks = [all_blocks_sorted[-1]]
        train_blocks = [b for b in all_blocks_sorted if b not in set(eval_blocks)]
        eval_mask = X[block_col].astype(str).isin(eval_blocks)
        X_eval = X.loc[eval_mask]
        y_eval = y.loc[eval_mask]

    # Compute brand-new blocks vs known per-block models
    known_blocks = set(model.per_block_models.keys())
    current_blocks = set(all_blocks_sorted)
    new_blocks = sorted(list(current_blocks - known_blocks))

    # Build the final list of blocks to train/update
    drifted_blocks = [str(b) for b in (drifted_blocks or []) if str(b) in train_blocks]
    new_blocks_init: List[str] = []
    if auto_init_new_blocks:
        # Only consider as trainable those new blocks that belong to TRAIN side
        for b in new_blocks:
            if str(b) in train_blocks:
                # must also have enough rows
                if X[X[block_col].astype(str) == str(b)].shape[0] >= int(min_rows_per_block):
                    if str(b) not in drifted_blocks:
                        drifted_blocks.append(str(b))
                    new_blocks_init.append(str(b))

    # Nature of the problem (classification/regression) based on global model
    is_clf = is_classifier(model.global_model) if model.global_model is not None else True

    # Windowing helper (mode 2)
    def _window_block_df(Xb: pd.DataFrame, yb: pd.Series, win: Optional[int]):
        if win is None or win <= 0 or len(Xb) <= win:
            return Xb, yb
        return Xb.tail(win), yb.tail(win)

    # Apply strategy block-by-block
    skipped_small: List[str] = []
    if drifted_blocks:
        logger.info(f"Retraining per-block models for: {drifted_blocks}")
        for b in drifted_blocks:
            if split_within_blocks:
                # If splitting, train only on the training part of the drifted block
                block_indices = X[X[block_col].astype(str) == str(b)].index
                n_samples = len(block_indices)
                n_train = int(n_samples * train_percentage)
                train_block_indices = np.random.permutation(block_indices)[:n_train]
                Xb_full = X.loc[train_block_indices].drop(columns=[block_col], errors="ignore")
                yb_full = y.loc[train_block_indices]
            else:
                Xb_full = X[X[block_col].astype(str) == str(b)].drop(columns=[block_col], errors="ignore")
                yb_full = y.loc[Xb_full.index]

            if Xb_full.shape[0] < int(min_rows_per_block):
                skipped_small.append(str(b))
                continue

            prev_m = model.get_block_model(b)
            if prev_m is None:
                prev_m = clone(model.base_estimator)
                df_prev = _load_block_training_data(control_dir, b)
                if df_prev is not None and "__target__" in df_prev.columns:
                    X_prev, y_prev = df_prev.drop(columns=["__target__"]), df_prev["__target__"]
                    try:
                        prev_m.fit(X_prev, y_prev)
                    except Exception:
                        pass

            Xb_use, yb_use = (Xb_full, yb_full)
            if mode == 2 and not split_within_blocks: # Windowing only makes sense when not splitting
                Xb_use, yb_use = _window_block_df(Xb_full, yb_full, window_size)

            new_block_model = _retrain_block_strategy(
                mode=mode,
                prev_model=prev_m,
                base_estimator=model.base_estimator,
                Xb=Xb_use,
                yb=yb_use,
                random_state=random_state,
                is_clf=is_clf,
                control_dir=control_dir,
                block_id=b,
            )
            model.set_block_model(b, new_block_model)
            _save_block_training_data(control_dir, b, Xb_use, yb_use)

    # Always retrain the global model with the latest full training data
    logger.info("Retraining the global model with all current training data...")
    if split_within_blocks:
        Xg = X_train_data.drop(columns=[block_col], errors="ignore")
        yg = y_train_data
    else:
        Xg = X[X[block_col].astype(str).isin(train_blocks)].drop(columns=[block_col], errors="ignore")
        yg = y.loc[Xg.index]

    if Xg.shape[0] > 0:
        new_global_model = clone(model.base_estimator)
        new_global_model.fit(Xg, yg)
        model.global_model = new_global_model
        logger.info("Global model has been retrained.")
    else:
        logger.warning("No training data available to retrain the global model.")


    # Report
    extra_report = {
        "retrained_blocks": drifted_blocks,
        "mode": mode,
        "new_blocks_initialized": new_blocks_init,
        "skipped_small_blocks": skipped_small,
    }
    report = _train_report(
        kind="retrain",
        file=last_processed_file,
        model=model,
        block_col=block_col,
        train_blocks=train_blocks,
        eval_blocks=eval_blocks,
        extra=extra_report,
        **report_params
    )
    try:
        with open(train_json, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

    meta = {"test_blocks": pd.Series(X_eval[block_col]) if block_col in X_eval else None}

    return model, X_eval, y_eval, meta
