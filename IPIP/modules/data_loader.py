# pipeline/modules/data_loader.py
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from pathlib import Path
from scipy.io import arff
import logging

def clear_logs(logs_dir, logger):
    """Clears the content of all log files without deleting them."""
    try:
        if os.path.exists(logs_dir):
            for file in os.listdir(logs_dir):
                file_path = Path(logs_dir) / file
                if file_path.is_file():
                    with open(file_path, "w"):
                        pass
            logger.info(f"🧹 Logs cleared in {logs_dir}")
        else:
            logger.warning(f"Logs directory {logs_dir} not found.")
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")

def load_file(path, delimiter=None):
    """Loads a file based on its extension."""
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path, delimiter=delimiter) if delimiter else pd.read_csv(path)
        elif path.suffix == ".arff":
            data, meta = arff.loadarff(path)
            return pd.DataFrame(data)
        elif path.suffix == ".json":
            return pd.read_json(path)
        elif path.suffix in (".xls", ".xlsx"):
            return pd.read_excel(path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".txt":
            # Try CSV-like reading, delimiter may be None
            return pd.read_csv(path, sep=delimiter if delimiter else None, engine="python")
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    except Exception as e:
        raise ValueError(f"Error loading file {path}: {e}")

def _sorted_block_ids(series: pd.Series):
    """Try numeric, then datetime, else lexicographic ordering of block ids."""
    vals = series.dropna().unique().tolist()

    # Convert numpy scalars to Python types for stability
    def _to_py(v):
        try:
            return v.item()
        except Exception:
            return v
    vals = [_to_py(v) for v in vals]

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

def _write_blocks_snapshot(control_dir: Path, *, file_name: str, block_col: str, df: pd.DataFrame, mtime: float, logger):
    """
    Persist a simple snapshot of blocks present in the dataset right now:
    {
      "<file>": {
        "block_col": "...",
        "blocks": [...],         # ordered list
        "counts": {"id": n, ...},
        "n_rows": N,
        "mtime": <float>
      }
    }
    """
    snap_path = control_dir / "blocks_snapshot.json"
    snapshot = {}
    if snap_path.exists():
        try:
            with open(snap_path, "r") as f:
                snapshot = json.load(f)
        except Exception:
            snapshot = {}

    if block_col and block_col in df.columns:
        blocks = _sorted_block_ids(df[block_col])
        counts_raw = df[block_col].value_counts(dropna=False)
        counts = {str(k): int(v) for k, v in counts_raw.items()}
    else:
        blocks, counts = [], {}

    snapshot[file_name] = {
        "block_col": block_col,
        "blocks": blocks,
        "counts": counts,
        "n_rows": int(len(df)),
        "mtime": float(mtime)
    }

    try:
        control_dir.mkdir(parents=True, exist_ok=True)
        with open(snap_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        logger.info(f"🧭 Blocks snapshot updated for '{file_name}': {len(blocks)} block(s).")
    except Exception as e:
        logger.error(f"Failed to write blocks snapshot: {e}")

def data_loader(logger, data_dir, control_dir, delimiter=None, target_file=None, *, block_col: str = None):
    """
    Loads data from a specific file (target_file) or checks the entire directory.
    Supports multiple formats: .csv, .arff, .json, .xlsx, .parquet, .txt

    Behavior:
    - Returns the FULL dataset as a DataFrame (no per-block slicing).
    - If `block_col` is provided and exists, writes/updates a blocks snapshot JSON.
    - Clears logs when a new or modified dataset is detected.
    """
    control_file = Path(control_dir) / "control_file.txt"
    logs_dir = Path(os.getcwd()) / "pipelines" / "my_pipeline_watchdog" / "logs"

    # Read processed mtimes
    records = {}
    if control_file.exists():
        try:
            with open(control_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        records[parts[0]] = float(parts[1])
        except Exception as e:
            logger.error(f"Error reading control file: {e}")

    # Target file path
    if target_file:
        file_path = Path(data_dir) / target_file
        if not file_path.exists():
            logger.error(f"The file {target_file} does not exist in {data_dir}")
            return pd.DataFrame(), None, None
        try:
            df = load_file(file_path, delimiter=delimiter)
            last_mtime = os.path.getmtime(file_path)
            logger.info(f"File {target_file} loaded successfully.")
            clear_logs(logs_dir, logger)
            _write_blocks_snapshot(Path(control_dir), file_name=target_file, block_col=block_col, df=df, mtime=last_mtime, logger=logger)
            return df, target_file, last_mtime
        except Exception as e:
            logger.error(f"Error loading file {target_file}: {e}")
            return pd.DataFrame(), None, None

    # Directory scan
    if not os.path.exists(data_dir):
        logger.error(f"The directory {data_dir} does not exist.")
        return pd.DataFrame(), None, None

    formats = (".csv", ".arff", ".json", ".xls", ".xlsx", ".parquet", ".txt")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(formats)]
    if not files:
        logger.warning("No files found in the data directory.")
        return pd.DataFrame(), None, None

    # Return the first new/modified file
    for file in files:
        file_path = Path(data_dir) / file
        mtime = os.path.getmtime(file_path)

        if file not in records or mtime > records.get(file, 0):
            try:
                df = load_file(file_path, delimiter=delimiter)
                logger.info(f"File {file} loaded successfully.")
                clear_logs(logs_dir, logger)
                _write_blocks_snapshot(Path(control_dir), file_name=file, block_col=block_col, df=df, mtime=mtime, logger=logger)
                return df, file, mtime
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")

    logger.info("No new files found to process.")
    return pd.DataFrame(), None, None
