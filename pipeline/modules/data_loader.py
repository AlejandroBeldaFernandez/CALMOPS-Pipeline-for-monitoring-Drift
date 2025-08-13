import os
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
                        pass  # Clear the file
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
            return pd.read_csv(path, sep=delimiter if delimiter else None, engine="python")
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    except Exception as e:
        raise ValueError(f"Error loading file {path}: {e}")

def data_loader(logger, data_dir, control_dir, delimiter=None, target_file=None):
    """
    Loads data from a specific file (target_file) or checks the entire directory.
    Supports multiple formats: .csv, .arff, .json, .xlsx, .parquet, .txt
    
    - If target_file is defined, loads that file.
    - If not, it checks the directory for the most recent unprocessed file.
    - Clears logs when a new or modified dataset is detected.
    """

    control_file = Path(control_dir) / "control_file.txt"
    logs_dir = Path(os.getcwd()) / "pipelines" / "my_pipeline_watchdog" / "logs"

    # --- Helper function to load dataset ---
    if target_file:
        file_path = Path(data_dir) / target_file
        if not file_path.exists():
            logger.error(f"The file {target_file} does not exist in {data_dir}")
            return pd.DataFrame(), None, None
        try:
            df = load_file(file_path)
            last_mtime = os.path.getmtime(file_path)
            logger.info(f"File {target_file} loaded successfully.")
            clear_logs(logs_dir, logger)
            return df, target_file, last_mtime
        except Exception as e:
            logger.error(f"Error loading file {target_file}: {e}")
            return pd.DataFrame(), None, None

    # --- If no target_file, check the directory ---
    if not os.path.exists(data_dir):
        logger.error(f"The directory {data_dir} does not exist.")
        return pd.DataFrame(), None, None

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

    # Searching for candidate files
    formats = (".csv", ".arff", ".json", ".xls", ".xlsx", ".parquet", ".txt")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(formats)]
    if not files:
        logger.warning("No files found in the data directory.")
        return pd.DataFrame(), None, None

    for file in files:
        file_path = Path(data_dir) / file
        mtime = os.path.getmtime(file_path)

        # Check if the file is new or modified
        if file not in records or mtime > records.get(file, 0):
            try:
                df = load_file(file_path)
                logger.info(f"File {file} loaded successfully.")
                clear_logs(logs_dir, logger)
                return df, file, mtime
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")

    logger.info("No new files found to process.")
    return pd.DataFrame(), None, None
