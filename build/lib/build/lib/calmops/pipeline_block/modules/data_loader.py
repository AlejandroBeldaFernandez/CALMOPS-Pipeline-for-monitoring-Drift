# modules/data_loader.py
import os
import json
import pandas as pd
from pathlib import Path
from scipy.io import arff
from typing import Optional, Dict, Tuple
import logging


def _decode_bytes_df(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte strings in object columns of a DataFrame."""
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].apply(lambda v: isinstance(v, (bytes, bytearray))).any():
            df[c] = df[c].apply(
                lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            )
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to coerce object columns to numeric types."""
    for col in df.select_dtypes(include=["object"]).columns:
        # Try to convert to numeric, if fails, keep original
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def load_file(
    path: Path,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    file_type: str = "csv",
) -> pd.DataFrame:
    """Load a file based on its extension or explicit file_type; tolerant to delimiter issues."""
    try:
        # Determine effective suffix/type
        suf = path.suffix.lower()
        # If file_type is provided and not 'auto', it overrides the suffix check logic
        # (though typically we still check suffix for specific handlers, here we can map type->handler)

        # Mapping common types to suffixes for logic reuse
        type_map = {
            "csv": ".csv",
            "arff": ".arff",
            "json": ".json",
            "excel": ".xlsx",
            "parquet": ".parquet",
            "txt": ".txt",
        }

        effective_type = type_map.get(file_type.lower(), suf)

        if effective_type == ".csv":
            # If delimiter is None, rely on default ','; if 'infer', let pandas sniff
            if delimiter == "infer":
                return pd.read_csv(
                    path, sep=None, engine="python", low_memory=False, encoding=encoding
                )
            return (
                pd.read_csv(
                    path, delimiter=delimiter, low_memory=False, encoding=encoding
                )
                if delimiter
                else pd.read_csv(path, low_memory=False, encoding=encoding)
            )

        elif effective_type == ".arff":
            # arff.loadarff usually handles encoding internally or assumes simple ascii/utf-8
            # If needed, we might need to read as text with encoding and parse, but standard lib doesn't support encoding arg easily.
            # For now, we assume arff lib handles it or we decode after.
            data, _ = arff.loadarff(path)
            df = pd.DataFrame(data)
            df = _decode_bytes_df(df)
            df = _coerce_numeric(df)
            return df

        elif effective_type == ".json":
            return pd.read_json(path, encoding=encoding)

        elif effective_type in (".xls", ".xlsx"):
            return pd.read_excel(path)

        elif effective_type == ".parquet":
            return pd.read_parquet(path)

        elif effective_type == ".txt":
            # Try CSV-like reading; if delimiter is None, let pandas infer
            if delimiter:
                return pd.read_csv(
                    path,
                    sep=delimiter,
                    engine="python",
                    low_memory=False,
                    encoding=encoding,
                )
            return pd.read_csv(
                path, sep=None, engine="python", low_memory=False, encoding=encoding
            )

        else:
            raise ValueError(f"Unsupported format: {effective_type}")

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
    logger: logging.Logger,
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
        logger.info(
            "🧭 Blocks snapshot updated for '%s': %d block(s).", file_name, len(blocks)
        )
    except Exception as e:
        logger.error("Failed to write blocks snapshot: %s", e)


# ---------------------------
# Control file (READ-ONLY)
# ---------------------------
def _read_control_records(
    control_file: Path, logger: logging.Logger
) -> Dict[str, float]:
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
def data_loader(
    logger: logging.Logger,
    data_dir: str | Path,
    control_dir: str | Path,
    delimiter: Optional[str] = None,
    target_file: Optional[str] = None,
    *,
    block_col: Optional[str] = None,
    encoding: str = "utf-8",
    file_type: str = "csv",
) -> Tuple[pd.DataFrame, Optional[str], Optional[float]]:
    """
    Loads data from a specific file (target_file) or scans the directory.
    Returns:
      - df FULL (not split by blocks),
      - last_processed_file (str),
      - last_mtime (float)
    """
    control_file = Path(control_dir) / "control_file.txt"

    # records of processed mtimes
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

    # target file path
    if target_file:
        file_path = Path(data_dir) / target_file
        if not file_path.exists():
            logger.error(f"The file {target_file} does not exist in {data_dir}")
            return pd.DataFrame(), None, None
        try:
            df = load_file(
                file_path, delimiter=delimiter, encoding=encoding, file_type=file_type
            )
            last_mtime = os.path.getmtime(file_path)
            logger.info(f"File {target_file} loaded successfully.")
            _write_blocks_snapshot(
                Path(control_dir),
                file_name=target_file,
                block_col=block_col,
                df=df,
                mtime=last_mtime,
                logger=logger,
            )
            return df, target_file, last_mtime
        except Exception as e:
            logger.error(f"Error loading file {target_file}: {e}")
            return pd.DataFrame(), None, None

    # scan directory
    if not os.path.exists(data_dir):
        logger.error(f"The directory {data_dir} does not exist.")
        return pd.DataFrame(), None, None

    formats = (".csv", ".arff", ".json", ".xls", ".xlsx", ".parquet", ".txt")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(formats)]
    if not files:
        logger.warning("No files found in the data directory.")
        return pd.DataFrame(), None, None

        # sort by mtime DESC to deterministically pick the most recent "new/modified" one
    files = sorted(
        files, key=lambda fn: os.path.getmtime(Path(data_dir) / fn), reverse=True
    )

    for file in files:
        file_path = Path(data_dir) / file
        mtime = os.path.getmtime(file_path)

        if file not in records or mtime > records.get(file, 0):
            try:
                df = load_file(file_path, delimiter=delimiter)
                logger.info(f"File {file} loaded successfully.")
                _write_blocks_snapshot(
                    Path(control_dir),
                    file_name=file,
                    block_col=block_col,
                    df=df,
                    mtime=mtime,
                    logger=logger,
                )
                return df, file, mtime
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")

    logger.info("No new files found to process.")
    return pd.DataFrame(), None, None
