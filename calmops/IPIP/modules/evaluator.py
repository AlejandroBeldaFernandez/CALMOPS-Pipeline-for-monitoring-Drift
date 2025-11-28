# pipeline/modules/evaluator.py
from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.base import is_classifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from ..ipip_model import IpipModel

# ------------------------------- utils -------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _save_json(
    payload: Dict[str, Any], out_dir: str, name: str = "eval_results.json"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=4)
    return path


def _detect_block_col(block_col: Optional[str], X: pd.DataFrame) -> Optional[str]:
    if block_col and block_col in X.columns:
        return block_col
    for c in X.columns:
        lc = str(c).lower()
        if "block" in lc or lc in ("block_id", "chunk"):
            return c
    return None


def _sorted_blocks(vals: pd.Series | List[Any]) -> List[str]:
    series = pd.Series(pd.unique(pd.Series(vals).astype(str))).dropna()
    v = series.astype(str).tolist()
    # numeric order if possible
    try:
        num = [float(x) for x in v]
        return [x for _, x in sorted(zip(num, v))]
    except Exception:
        pass
    # datetime order if possible
    try:
        dt = pd.to_datetime(v, errors="raise")
        return [x for _, x in sorted(zip(dt, v))]
    except Exception:
        pass
    # lexicographic fallback
    return sorted(v, key=lambda x: str(x))


# ------------------------------- metrics & approval -------------------------------

_HIGHER_BETTER = {"accuracy", "balanced_accuracy", "f1", "r2"}
_LOWER_BETTER = {"rmse", "mae", "mse"}


def _metrics_cls(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _metrics_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def _approved(
    metrics: Dict[str, float], thresholds: Optional[Dict[str, float]]
) -> bool:
    """
    Global approval using the same thresholds semantics as training:
    - For 'higher is better' metrics: metric >= threshold -> pass
    - For 'lower is better' metrics: metric <= threshold -> pass
    If thresholds is None/empty, approval is True.
    """
    if not thresholds:
        return True
    ok = True
    for k, thr in thresholds.items():
        if k in _HIGHER_BETTER:
            ok = ok and (metrics.get(k) is not None and float(metrics[k]) >= float(thr))
        elif k in _LOWER_BETTER:
            ok = ok and (metrics.get(k) is not None and float(metrics[k]) <= float(thr))
        else:
            ok = ok and True
    return bool(ok)


def _pick_rank_metric(
    task: str, thresholds: Optional[Dict[str, float]], rank_by: Optional[str]
) -> Tuple[str, int]:
    """
    Returns (metric_name, sign) used to rank worst blocks.
    sign = +1 means higher is worse (e.g., errors); sign = -1 means lower is worse (e.g., accuracy).
    """
    if rank_by:
        m = rank_by
    else:
        m = "balanced_accuracy" if task == "classification" else "r2"
    if m in _LOWER_BETTER:
        # lower better -> higher is worse, so sign=+1
        return m, +1
    # higher better -> lower is worse, so sign=-1
    return m, -1


# ------------------------------- main API -------------------------------


def evaluate_model(
    model_or_path: str | Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    X_full: Optional[pd.DataFrame] = None,
    logger: logging.Logger = None,
    metrics_dir: str,
    control_dir: str,
    data_file: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    block_col: Optional[str] = None,
    evaluated_blocks: Optional[List[str]] = None,  # restrict eval to these blocks
    reference_blocks: Optional[
        List[str]
    ] = None,  # blocks used for training (for reporting / new-blocks detection)
    include_predictions: bool = True,
    max_pred_examples: int = 100,
    mtime: Optional[float] = None,
    # new options
    include_confusion_by_block: bool = False,
    rank_by: Optional[str] = None,  # metric to rank worst blocks
    top_k_worst: int = 5,
    dir_predictions: Optional[str] = None,
) -> bool:
    """
    Evaluate a model (router or plain sklearn) in block-wise mode.

    - If a block column is present (or provided), compute per-block metrics.
    - `evaluated_blocks` restricts the evaluation subset.
    - `reference_blocks` are used to flag brand-new blocks in evaluation.
    - Global approval is computed over the entire evaluation subset using `thresholds`.
    - Per-block approval is also reported using the same thresholds (helps triage).
    - Saves eval_results.json in `metrics_dir`. If approved and `data_file` is given,
      append "<file>,<mtime_float>\n" to control/control_file.txt.

    Returns:
        bool -> global approved flag
    """
    # Load model
    model = (
        joblib.load(model_or_path)
        if isinstance(model_or_path, (str, os.PathLike))
        else model_or_path
    )

    is_cls = True
    task = "classification"
    base_model_name = "IpipModel"

    if isinstance(model, IpipModel):
        if model.ensembles_ and model.ensembles_[0]:
            base_model = model.ensembles_[0][0]
            is_cls = is_classifier(base_model)
            task = "classification" if is_cls else "regression"
            base_model_name = type(base_model).__name__
    else:
        base_model = getattr(model, "global_model", model)
        is_cls = is_classifier(base_model)
        task = "classification" if is_cls else "regression"
        base_model_name = type(base_model).__name__

    # Block handling
    bc = _detect_block_col(block_col, X_eval)
    X_in = X_eval.copy()
    y_in = y_eval.copy()

    if bc and evaluated_blocks:
        mask = X_in[bc].astype(str).isin([str(x) for x in evaluated_blocks])
        X_in = X_in.loc[mask]
        y_in = y_in.loc[X_in.index]

    # Empty subset guard
    if len(X_in) == 0 or len(y_in) == 0:
        if logger:
            logger.warning(
                "[evaluator] Empty evaluation subset (no rows in selected blocks)."
            )
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "approved": False,
            "reason": "empty_evaluation_subset",
            "metrics": {},
            "thresholds": thresholds or {},
            "blocks": {
                "block_col": bc,
                "evaluated_blocks": list(map(str, evaluated_blocks))
                if evaluated_blocks
                else [],
                "reference_blocks": list(map(str, reference_blocks))
                if reference_blocks
                else [],
                "new_blocks_detected": [],
                "missing_reference_blocks": [],
                "per_block_metrics_full": {},
                "per_block_approved": {},
                "approved_blocks_count": 0,
                "rejected_blocks_count": 0,
                "top_worst_blocks": [],
            },
            "predictions": [],
        }
        _save_json(results, metrics_dir, "eval_results.json")
        return False

    # Predict
    if not hasattr(model, "predict"):
        raise RuntimeError("Model does not implement predict().")

    X_drop = X_in.drop(columns=[bc], errors="ignore") if bc else X_in
    y_pred = model.predict(X_drop)

    # Global metrics
    if is_cls:
        metrics_global: Dict[str, Any] = _metrics_cls(y_in, y_pred)
        try:
            metrics_global["classification_report"] = classification_report(
                y_in, y_pred, output_dict=True, zero_division=0
            )
        except Exception:
            pass
    else:
        metrics_global = _metrics_reg(y_in, y_pred)

    # Per-block metrics + approvals
    blocks_payload: Dict[str, Any] = {
        "block_col": bc,
        "evaluated_blocks": [],
        "reference_blocks": list(map(str, reference_blocks))
        if reference_blocks
        else [],
        "new_blocks_detected": [],
        "missing_reference_blocks": [],
        "per_block_metrics_full": {},
        "per_block_approved": {},
        "approved_blocks_count": 0,
        "rejected_blocks_count": 0,
        "top_worst_blocks": [],
    }

    # Compute block lists / novelty if block column exists
    if bc:
        eval_block_ids = _sorted_blocks(X_in[bc].astype(str))
        blocks_payload["evaluated_blocks"] = eval_block_ids

        ref_set = set(map(str, reference_blocks)) if reference_blocks else set()
        eval_set = set(eval_block_ids)
        # New blocks present in eval that were not in reference
        blocks_payload["new_blocks_detected"] = (
            sorted(list(eval_set - ref_set)) if ref_set else []
        )
        # Reference blocks not present in the current eval subset
        blocks_payload["missing_reference_blocks"] = (
            sorted(list(ref_set - eval_set)) if ref_set else []
        )

        # Align indexes to avoid mismatches
        y_pred_s = pd.Series(y_pred, index=y_in.index)

        # Compute per-block metrics
        for b, idx in X_in[bc].groupby(X_in[bc]).groups.items():
            idx = pd.Index(idx)
            yb_true = y_in.loc[idx]
            yb_pred = y_pred_s.loc[idx].values
            m = (
                _metrics_cls(yb_true, yb_pred)
                if is_cls
                else _metrics_reg(yb_true, yb_pred)
            )
            blocks_payload["per_block_metrics_full"][str(b)] = m
            blocks_payload["per_block_approved"][str(b)] = _approved(m, thresholds)

        # Counts
        blocks_payload["approved_blocks_count"] = sum(
            1 for v in blocks_payload["per_block_approved"].values() if v
        )
        blocks_payload["rejected_blocks_count"] = sum(
            1 for v in blocks_payload["per_block_approved"].values() if not v
        )

        # Optional confusion matrices by block (classification only)
        if include_confusion_by_block and is_cls:
            cm_dict: Dict[str, Any] = {}
            for b, idx in X_in[bc].groupby(X_in[bc]).groups.items():
                idx = pd.Index(idx)
                yb_true = y_in.loc[idx]
                yb_pred = y_pred_s.loc[idx].values
                try:
                    cm = confusion_matrix(yb_true, yb_pred)
                    cm_dict[str(b)] = cm.tolist()
                except Exception:
                    cm_dict[str(b)] = None
            blocks_payload["per_block_confusion_matrix"] = cm_dict

        # Rank worst blocks
        metric_to_rank, sign = _pick_rank_metric(task, thresholds, rank_by)
        worst_rows = []
        for b, m in blocks_payload["per_block_metrics_full"].items():
            if metric_to_rank in m and m[metric_to_rank] is not None:
                val = float(m[metric_to_rank])
                # score = val if higher is worse, else -val (we chose sign above)
                score = sign * val
                worst_rows.append((b, val, score))
        if worst_rows:
            worst_rows.sort(key=lambda x: x[2], reverse=True)  # higher score = worse
            blocks_payload["top_worst_blocks"] = [
                {"block": b, metric_to_rank: v}
                for b, v, _ in worst_rows[: int(max(1, top_k_worst))]
            ]

    # Global approval
    approved_flag = _approved(metrics_global, thresholds)

    # Prediction examples
    predictions = None
    if include_predictions:
        try:
            dfp = pd.DataFrame(
                {"y_true": y_in.values, "y_pred": y_pred}, index=y_in.index
            )
            if bc and bc in X_in.columns:
                dfp["block"] = X_in[bc].astype(str)
            predictions = (
                dfp.head(int(max_pred_examples))
                .reset_index(drop=True)
                .to_dict(orient="records")
            )
        except Exception:
            predictions = None

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "approved": bool(approved_flag),
        "metrics": metrics_global,
        "thresholds": thresholds or {},
        "blocks": blocks_payload,
        "predictions": predictions or [],
        "model_info": {
            "type": base_model_name,
            "is_ipip": isinstance(model, IpipModel),
        },
    }
    _save_json(results, metrics_dir, "eval_results.json")
    if logger:
        logger.info("[evaluator] eval_results.json saved.")

    # Save a copy to history
    try:
        history_dir = os.path.join(metrics_dir, "eval_history")
        os.makedirs(history_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_fname = f"eval_results_{ts}.json"
        _save_json(results, history_dir, history_fname)
        if logger:
            logger.info(f"[evaluator] Saved historical eval to {history_fname}")
    except Exception as e:
        if logger:
            logger.warning(f"[evaluator] Could not save historical eval: {e}")

    # Update control_file.txt if globally approved
    if approved_flag and data_file:
        try:
            os.makedirs(control_dir, exist_ok=True)
            control_file = os.path.join(control_dir, "control_file.txt")
            with open(control_file, "a", encoding="utf-8") as f:
                if mtime is not None:
                    f.write(f"{data_file},{float(mtime)}\n")
                else:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{data_file},{ts}\n")
            if logger:
                logger.info("[evaluator] control_file.txt updated (approved).")
        except Exception as e:
            if logger:
                logger.warning(f"[evaluator] Could not update control_file.txt: {e}")

        # Save predictions on the full dataset
        if X_full is not None:
            try:
                if logger:
                    logger.info("Saving predictions on the full dataset.")
                full_predictions = model.predict(
                    X_full.drop(columns=[bc], errors="ignore") if bc else X_full
                )
                predictions_df = pd.DataFrame(full_predictions, columns=["prediction"])
                predictions_path = os.path.join(
                    metrics_dir,
                    f"predictions_{os.path.splitext(os.path.basename(data_file))[0]}.csv",
                )
                predictions_df.to_csv(predictions_path, index=False)
                if logger:
                    logger.info(f"Full predictions saved to {predictions_path}")

                if dir_predictions:
                    try:
                        os.makedirs(dir_predictions, exist_ok=True)
                        extra_pred_path = os.path.join(
                            dir_predictions,
                            f"predictions_{os.path.splitext(os.path.basename(data_file))[0]}.csv",
                        )
                        predictions_df.to_csv(extra_pred_path, index=False)
                        if logger:
                            logger.info(
                                f"Full predictions also saved to {extra_pred_path}"
                            )
                    except Exception as e:
                        if logger:
                            logger.warning(
                                f"Could not save predictions to dir_predictions ({dir_predictions}): {e}"
                            )

            except Exception as e:
                if logger:
                    logger.error(f"Could not save full predictions: {e}")

    return bool(approved_flag)
