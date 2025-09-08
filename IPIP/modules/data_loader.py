# pipeline/modules/data_loader.py
# -*- coding: utf-8 -*-
"""
Robust data loader for IPIP pipelines (READ-ONLY control file).
- Supports CSV/ARFF/JSON/Excel/Parquet/TXT.
- Reads 'control_file.txt' (if present) to know last processed mtimes,
  but DOES NOT write/update it here.
- Optionally clears logs when a new/modified dataset is detected.
- Writes a blocks snapshot JSON if `block_col` is provided.
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from scipy.io import arff


# ---------------------------
# File readers
# ---------------------------
def _decode_bytes_df(df: pd.DataFrame) -> pd.DataFrame:
    """Decode bytes/bytearray values to str (common when loading ARFF)."""
    if df.empty:
        return df
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce object columns to numeric where possible."""
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
    return df


def load_file(path: Path, delimiter: Optional[str] = None) -> pd.DataFrame:
    """Load a file based on its extension; tolerant to delimiter issues."""
    try:
        suf = path.suffix.lower()
        if suf == ".csv":
            # If delimiter is None, rely on default ','; if 'infer', let pandas sniff
            if delimiter == "infer":
                return pd.read_csv(path, sep=None, engine="python", low_memory=False)
            return pd.read_csv(path, delimiter=delimiter, low_memory=False) if delimiter else pd.read_csv(path, low_memory=False)

        elif suf == ".arff":
            data, _ = arff.loadarff(path)
            df = pd.DataFrame(data)
            df = _decode_bytes_df(df)
            df = _coerce_numeric(df)
            return df

        elif suf == ".json":
            return pd.read_json(path)

        elif suf in (".xls", ".xlsx"):
            return pd.read_excel(path)

        elif suf == ".parquet":
            return pd.read_parquet(path)

        elif suf == ".txt":
            # Try CSV-like reading; if delimiter is None, let pandas infer
            if delimiter:
                return pd.read_csv(path, sep=delimiter, engine="python", low_memory=False)
            return pd.read_csv(path, sep=None, engine="python", low_memory=False)

        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    except Exception as e:
        raise ValueError(f"Error loading file {path}: {e}") from e


# ---------------------------
# Blocks utilities
# ---------------------------
def _sorted_block_ids(series: pd.Series):
    """Try numeric, then datetime, else lexicographic ordering of block ids."""
    vals = series.dropna().unique().tolist()

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


def _write_blocks_snapshot(
    control_dir: Path,
    *,
    file_name: str,
    block_col: Optional[str],
    df: pd.DataFrame,
    mtime: float,
    logger: logging.Logger
):
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
    snapshot: Dict[str, dict] = {}
    if snap_path.exists():
        try:
            with open(snap_path, "r", encoding="utf-8") as f:
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
        "mtime": float(mtime),
    }

    try:
        control_dir.mkdir(parents=True, exist_ok=True)
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        logger.info("🧭 Blocks snapshot updated for '%s': %d block(s).", file_name, len(blocks))
    except Exception as e:
        logger.error("Failed to write blocks snapshot: %s", e)


# ---------------------------
# Control file (READ-ONLY)
# ---------------------------
def _read_control_records(control_file: Path, logger: logging.Logger) -> Dict[str, float]:
    """Read last processed mtimes from control file; never writes here."""
    records: Dict[str, float] = {}
    if control_file.exists():
        try:
            with open(control_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) == 2:
                        records[parts[0]] = float(parts[1])
        except Exception as e:
            logger.error("Error reading control file: %s", e)
    return records


# ---------------------------
# Logs cleanup
# ---------------------------
def _clear_logs(logs_dir: Path, logger: logging.Logger) -> None:
    """Delete files inside logs directory (non-recursive)."""
    try:
        if logs_dir.exists():
            for p in logs_dir.glob("*"):
                if p.is_file():
                    p.unlink(missing_ok=True)
        logger.info("🧹 Cleared logs in %s", logs_dir)
    except Exception as e:
        logger.warning("Could not clear logs in %s: %s", logs_dir, e)


# ---------------------------
# Public API
# ---------------------------
def data_loader(logger, data_dir: str | Path, control_dir: str | Path, delimiter: Optional[str] = None, target_file: Optional[str] = None, *, block_col: str = None) -> Tuple[pd.DataFrame, Optional[str], Optional[float]]:
    """
    Load data from a specific file (target_file) or scan the directory.

    Returns:
    -------
    (df, file_name, mtime)
      - df: pandas.DataFrame (FULL dataset; no per-block slicing here)
      - file_name: str or None
      - mtime: float or None

    Side effects:
      - Reads (but does NOT write) control_file.txt.
      - Updates blocks_snapshot.json if block_col is provided.
      - Clears logs when a new or modified dataset is detected.
    """
    data_dir = Path(data_dir)
    control_dir = Path(control_dir)

    control_file = control_dir / "control_file.txt"
    # logs dir coherente con la estructura de la pipeline: pipelines/<name>/logs
    logs_dir = control_dir.parent / "logs"

    # Load processed mtimes (read-only)
    records = _read_control_records(control_file, logger)

    # ---- Target file path (explicit) ----
    if target_file:
        file_path = data_dir / target_file
        if not file_path.exists():
            logger.error("The file %s does not exist in %s", target_file, data_dir)
            return pd.DataFrame(), None, None
        try:
            df = load_file(file_path, delimiter=delimiter)
            last_mtime = os.path.getmtime(file_path)

            _write_blocks_snapshot(control_dir, file_name=target_file, block_col=block_col, df=df, mtime=last_mtime, logger=logger)
            _clear_logs(logs_dir, logger)

            logger.info("File %s loaded successfully.", target_file)
            return df, target_file, last_mtime

        except Exception as e:
            logger.error("Error loading file %s: %s", target_file, e)
            return pd.DataFrame(), None, None

    # ---- Directory scan ----
    if not data_dir.exists():
        logger.error("The directory %s does not exist.", data_dir)
        return pd.DataFrame(), None, None

    formats = (".csv", ".arff", ".json", ".xls", ".xlsx", ".parquet", ".txt")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(formats)]
    if not files:
        logger.warning("No files found in the data directory.")
        return pd.DataFrame(), None, None

    # Sort by mtime (newest first) so we pick the latest updated candidate
    files_sorted = sorted(files, key=lambda nm: os.path.getmtime(data_dir / nm), reverse=True)

    for file in files_sorted:
        file_path = data_dir / file
        mtime = os.path.getmtime(file_path)

        # Decide "new/modified" only using read-only records
        if file not in records or mtime > records.get(file, 0.0):
            try:
                df = load_file(file_path, delimiter=delimiter)

                _write_blocks_snapshot(control_dir, file_name=file, block_col=block_col, df=df, mtime=mtime, logger=logger)
                _clear_logs(logs_dir, logger)

                logger.info("File %s loaded successfully.", file)
                return df, file, mtime
            except Exception as e:
                logger.error("Error loading file %s: %s", file, e)

    logger.info("No new files found to process.")
    return pd.DataFrame(), None, None
