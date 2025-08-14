import os
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.base import is_classifier, is_regressor

# --- Se asume que ya tienes importados en tu proyecto:
# from yourmodule.training import default_train, default_retrain
# from yourmodule.drift import check_drift
# from yourmodule.loader import data_loader

# ---------------------------
# Helpers
# ---------------------------
def _sorted_block_ids(series: pd.Series) -> List:
    """Try numeric, then datetime, else lexicographic ordering of block ids."""
    vals = series.dropna().unique().tolist()
    # numeric
    try:
        nums = [float(v) for v in vals]
        return [x for _, x in sorted(zip(nums, vals))]
    except Exception:
        pass
    # datetime
    try:
        dt = pd.to_datetime(vals, errors="raise")
        return [x for _, x in sorted(zip(dt, vals))]
    except Exception:
        pass
    # lexicographic
    return sorted(vals, key=lambda x: str(x))

def _persist_model(model, model_dir: str, model_filename: str, logger):
    """
    Persist the model to disk keeping a previous copy:
      - current -> _previous.pkl
      - new model -> current .pkl
    This matches your check_drift rollback expectations.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    prev_model_path = model_path.replace(".pkl", "_previous.pkl")

    try:
        # Move current to previous (if exists)
        if os.path.exists(model_path):
            # Overwrite previous if exists
            if os.path.exists(prev_model_path):
                os.remove(prev_model_path)
            os.replace(model_path, prev_model_path)

        # Dump new model
        joblib.dump(model, model_path)
        logger.info(f"[MODEL] Saved current model at {model_path}")
        if os.path.exists(prev_model_path):
            logger.info(f"[MODEL] Previous model available at {prev_model_path}")
    except Exception as e:
        logger.error(f"[MODEL] Failed to persist model: {e}")
        raise

def _write_previous_data_csv(control_dir: str, df_ref: pd.DataFrame, y_col: str, logger):
    """
    Write control/previous_data.csv with the reference window (features + target).
    This is used by check_drift() for statistical tests (Frouros).
    """
    out_dir = Path(control_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "previous_data.csv"
    try:
        df_ref.to_csv(path, index=False)
        logger.info(f"[REF] Updated previous_data.csv with {len(df_ref)} rows at {path}")
    except Exception as e:
        logger.error(f"[REF] Failed to write previous_data.csv: {e}")

def _select_initial_blocks(block_ids: List, initial_k: int) -> List:
    """Pick first K blocks for initial training (bounded)."""
    initial_k = max(1, int(initial_k))
    return block_ids[:min(len(block_ids), initial_k)]

def _build_ref_window(df: pd.DataFrame, block_col: str, y_col: str, ref_blocks: List) -> pd.DataFrame:
    """Concatenate reference blocks, returning a DataFrame with features + target."""
    if not ref_blocks:
        return pd.DataFrame(columns=[*df.columns])
    ref_df = df[df[block_col].isin(ref_blocks)].copy()
    # Keep features + target (drop block_col)
    return ref_df.drop(columns=[block_col])

# ---------------------------
# Main driver
# ---------------------------
def train_retrain_on_blocked_dataset(
    *,
    logger,
    detector,
    perf_thresholds: dict,
    model_instance,            # e.g., RandomForestClassifier(...)
    model_dir: str,
    model_filename: str,       # e.g., "model.pkl"
    output_dir: str,
    data_dir: str,
    control_dir: str,
    y_col: str,
    block_col: str,
    # data loader options
    target_file: Optional[str] = None,
    delimiter: Optional[str] = None,
    # initial training window
    initial_k_blocks: int = 1,
    # reference policy for drift stats
    reference_mode: str = "rolling",   # "first" | "rolling"
    rolling_k: int = 3,
    # retraining policy
    retrain_mode: int = 0,             # your modes 0..6
    window_size: Optional[int] = None, # used in mode 2
    replay_frac_old: float = 0.4,      # used in mode 5
    # default_train grid
    random_state: int = 42,
    param_grid: Optional[dict] = None,
    cv: Optional[int] = None,
):
    """
    End-to-end routine:
      1) Load FULL dataset (no slicing at loader).
      2) Sort block ids and train on first K blocks.
      3) For each subsequent block:
         - Update previous_data.csv based on reference_mode (first/rolling).
         - Run check_drift on the block.
         - If 'retrain' → default_retrain with retrain_mode.
      4) Persist model at each successful (re)train.

    Returns:
      dict with per-block decisions and a compact summary.
    """
    # 1) Load FULL dataset (and snapshot blocks via data_loader if ya integrado)
    from yourmodule.loader import data_loader  # <-- ajusta el import a tu proyecto real
    df, last_file, last_mtime = data_loader(
        logger, data_dir, control_dir, delimiter=delimiter,
        target_file=target_file, block_col=block_col
    )
    if df.empty:
        logger.warning("[PIPELINE] No dataset returned by loader. Nothing to do.")
        return {"summary": {"status": "no_data"}}

    if y_col not in df.columns:
        raise ValueError(f"[PIPELINE] y_col='{y_col}' not found in dataset.")

    if block_col not in df.columns:
        logger.warning(f"[PIPELINE] block_col='{block_col}' not found. Treating as a single block.")
        df[block_col] = "ALL"

    # 2) Identify and sort blocks
    block_ids = _sorted_block_ids(df[block_col])
    if not block_ids:
        logger.warning("[PIPELINE] No blocks found. Treating as a single block 'ALL'.")
        block_ids = ["ALL"]

    # Split features/target by block for reuse
    by_block = {}
    for bid in block_ids:
        part = df[df[block_col] == bid]
        yb = part[y_col]
        Xb = part.drop(columns=[y_col, block_col])
        by_block[bid] = (Xb, yb)

    # 3) Initial training on first K blocks
    init_blocks = _select_initial_blocks(block_ids, initial_k_blocks)
    df_init = df[df[block_col].isin(init_blocks)].copy()
    X_init = df_init.drop(columns=[y_col, block_col])
    y_init = df_init[y_col]

    from yourmodule.training import default_train, default_retrain  # ajusta import
    logger.info(f"[PIPELINE] Initial training on blocks={init_blocks} (rows={len(df_init)}).")
    model, X_eval, y_eval, train_info = default_train(
        X_init, y_init,
        last_processed_file=last_file,
        model_instance=model_instance,
        random_state=random_state,
        logger=logger,
        output_dir=output_dir,
        param_grid=param_grid,
        cv=cv
    )
    _persist_model(model, model_dir, model_filename, logger)

    # Prepare reference list for drift stats
    if reference_mode not in ("first", "rolling"):
        raise ValueError("reference_mode must be 'first' or 'rolling'")
    ref_blocks = list(init_blocks)  # start with initial

    # 4) Iterate over the remaining blocks
    from yourmodule.drift import check_drift  # ajusta import
    decisions = []
    for bid in block_ids:
        if bid in init_blocks:
            decisions.append({"block_id": bid, "decision": "trained_as_reference"})
            continue

        Xb, yb = by_block[bid]

        # Update previous_data.csv per policy
        if reference_mode == "first":
            # keep the initial blocks as reference forever
            df_ref = _build_ref_window(df, block_col, y_col, ref_blocks)
        else:
            # rolling: use last rolling_k processed blocks (including most recent non-current)
            # Ensure current block not in reference
            recent = [b for b in decisions if b.get("decision") != "trained_as_reference"]
            done_bids = [d["block_id"] for d in recent if d["block_id"] in by_block]
            # Use last K from (init_blocks + done_bids) excluding current
            hist = [*init_blocks, *done_bids]
            ref_sel = hist[-rolling_k:] if rolling_k > 0 else hist
            df_ref = _build_ref_window(df, block_col, y_col, ref_sel)
            ref_blocks = ref_sel  # keep tracking

        _write_previous_data_csv(control_dir, df_ref, y_col, logger)

        # Run drift check for the current block
        logger.info(f"[PIPELINE] Drift check on block '{bid}' with |X|={len(Xb)}.")
        decision = check_drift(
            X=Xb, y=yb,
            detector=detector,
            logger=logger,
            perf_thresholds=perf_thresholds,
            model_filename=model_filename,
            data_dir=data_dir,          # kept for signature compatibility
            output_dir=output_dir,
            control_dir=control_dir,
            model_dir=model_dir,
        )

        # Handle decision
        if decision == "retrain":
            logger.info(f"[PIPELINE] Retraining triggered for block '{bid}' (mode={retrain_mode}).")
            model, X_eval, y_eval, retrain_info = default_retrain(
                X=df.drop(columns=[y_col, block_col]),   # retrain on *current dataset* (simple policy)
                y=df[y_col],
                last_processed_file=last_file,
                model_path=os.path.join(model_dir, model_filename),
                mode=retrain_mode,
                random_state=random_state,
                logger=logger,
                output_dir=output_dir,
                param_grid=param_grid,
                cv=cv,
                window_size=window_size,
                replay_frac_old=replay_frac_old
            )
            _persist_model(model, model_dir, model_filename, logger)
            decisions.append({"block_id": bid, "decision": "retrain"})
        elif decision in ("no_drift", "end"):
            logger.info(f"[PIPELINE] No retrain needed for block '{bid}'.")
            decisions.append({"block_id": bid, "decision": "no_retrain"})
        elif decision == "previous_promoted":
            logger.info(f"[PIPELINE] Previous model promoted during block '{bid}'.")
            decisions.append({"block_id": bid, "decision": "previous_promoted"})
        elif decision == "train":
            # This should not happen after initial training, but we handle it defensively.
            logger.info(f"[PIPELINE] 'train' requested at block '{bid}'. Performing full training.")
            model, X_eval, y_eval, train_info = default_train(
                X_init, y_init, last_processed_file=last_file,
                model_instance=model_instance, random_state=random_state,
                logger=logger, output_dir=output_dir,
                param_grid=param_grid, cv=cv
            )
            _persist_model(model, model_dir, model_filename, logger)
            decisions.append({"block_id": bid, "decision": "trained_again"})
        else:
            logger.warning(f"[PIPELINE] Unexpected decision '{decision}' at block '{bid}'. Proceeding.")
            decisions.append({"block_id": bid, "decision": str(decision)})

    # 5) Persist a compact report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "file": last_file,
        "y_col": y_col,
        "block_col": block_col,
        "reference_mode": reference_mode,
        "initial_blocks": init_blocks,
        "rolling_k": rolling_k,
        "decisions": decisions,
        "summary": {
            "num_blocks": len(block_ids),
            "num_retrains": sum(1 for d in decisions if d["decision"] == "retrain"),
            "num_promotions": sum(1 for d in decisions if d["decision"] == "previous_promoted"),
        }
    }
    try:
        with open(os.path.join(output_dir, "blocks_training_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[PIPELINE] Blocks training report saved at {os.path.join(output_dir, 'blocks_training_report.json')}")
    except Exception as e:
        logger.error(f"[PIPELINE] Failed to save blocks report: {e}")

    return report
