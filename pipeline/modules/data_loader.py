import os
import pandas as pd
from pathlib import Path
from scipy.io import arff
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')

def load_file(path, delimiter=None):
    """
    Multi-format file loader that handles various data file types.
    
    This function implements a strategy pattern for loading different file formats
    into pandas DataFrames. It supports common data science and machine learning
    file formats with appropriate parsing methods for each type.
    
    Supported formats:
    - CSV: Standard comma-separated values with optional custom delimiter
    - ARFF: Weka's Attribute-Relation File Format for machine learning datasets
    - JSON: JavaScript Object Notation files
    - Excel: Both .xls (legacy) and .xlsx (modern) Excel formats
    - Parquet: Columnar storage format optimized for analytics
    - TXT: Plain text files treated as delimited data
    
    Args:
        path (Path): File path object pointing to the data file
        delimiter (str, optional): Custom delimiter for CSV/TXT files. Defaults to None.
        
    Returns:
        pandas.DataFrame: Loaded data as a DataFrame
        
    Raises:
        ValueError: If file format is unsupported or loading fails
    """
    try:
        # CSV format: Handle both standard comma-separated and custom delimiter cases
        if path.suffix == ".csv":
            return pd.read_csv(path, delimiter=delimiter) if delimiter else pd.read_csv(path)
        
        # ARFF format: Convert Weka format to DataFrame, discarding metadata
        elif path.suffix == ".arff":
            data, meta = arff.loadarff(path)
            return pd.DataFrame(data)
        
        # JSON format: Direct pandas JSON reader
        elif path.suffix == ".json":
            return pd.read_json(path)
        
        # Excel formats: Support both legacy and modern Excel files
        elif path.suffix in (".xls", ".xlsx"):
            return pd.read_excel(path)
        
        # Parquet format: Optimized columnar format
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        
        # Text format: Treat as delimited data with flexible parsing
        elif path.suffix == ".txt":
            return pd.read_csv(path, sep=delimiter if delimiter else None, engine="python")
        
        # Unsupported format handling
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
    except Exception as e:
        # Wrap all exceptions in a consistent error format for upstream handling
        raise ValueError(f"Failed to load file {path}: {e}")

def data_loader(logger, data_dir, control_dir, delimiter=None, target_file=None):
    """
    Intelligent data loader with incremental processing capabilities.
    
    This function implements a dual-mode data loading strategy:
    1. Direct mode: Loads a specific target file when provided
    2. Discovery mode: Scans directory for new/modified files using timestamp tracking
    
    The incremental processing system uses a control file to track file modification
    times (mtime) to avoid reprocessing unchanged data, enabling efficient pipeline
    operations in production environments.
    
    Control File Management:
    - Maintains a CSV-style control file mapping filenames to modification timestamps
    - Format: filename,mtime_timestamp per line
    - Enables incremental processing by tracking which files have been processed
    - Automatically handles missing or corrupted control files
    
    Error Handling Strategy:
    - Graceful degradation: Returns empty DataFrame on failures
    - Comprehensive logging for debugging and monitoring
    - Format-specific error handling through the load_file() function
    - Validation of file existence and directory accessibility
    
    Args:
        logger: Configured logging instance for operation tracking
        data_dir (str): Source directory containing data files
        control_dir (str): Directory for storing processing control files
        delimiter (str, optional): Custom delimiter for delimited file formats
        target_file (str, optional): Specific filename to load (bypasses discovery)
        
    Returns:
        tuple: (DataFrame, filename, mtime) where:
            - DataFrame: Loaded data or empty DataFrame on failure
            - filename: Name of loaded file or None on failure
            - mtime: File modification timestamp or None on failure
    """
    
    # Initialize control file path for tracking processed files
    control_file = Path(control_dir) / "control_file.txt"
    logs_dir = Path(os.getcwd()) / "pipelines" / "my_pipeline_watchdog" / "logs"

    # DIRECT MODE: Load specific target file when provided
    if target_file:
        file_path = Path(data_dir) / target_file
        
        # Validate target file existence
        if not file_path.exists():
            logger.error(f"Target file not found: {target_file} in directory {data_dir}")
            return pd.DataFrame(), None, None
            
        try:
            # Load the specified file and capture its modification time
            df = load_file(file_path, delimiter)
            last_mtime = os.path.getmtime(file_path)
            logger.info(f"Successfully loaded target file: {target_file} ({len(df)} rows)")
            
            return df, target_file, last_mtime
            
        except Exception as e:
            logger.error(f"Failed to load target file {target_file}: {e}")
            return pd.DataFrame(), None, None

    # DISCOVERY MODE: Scan directory for new or modified files
    
    # Validate source directory existence
    if not os.path.exists(data_dir):
        logger.error(f"Source directory does not exist: {data_dir}")
        return pd.DataFrame(), None, None

    # Load existing processing records from control file
    # Records format: {filename: mtime_timestamp}
    records = {}
    if control_file.exists():
        try:
            with open(control_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        records[parts[0]] = float(parts[1])
            logger.debug(f"Loaded {len(records)} file records from control file")
        except Exception as e:
            logger.warning(f"Unable to read control file, proceeding without history: {e}")

    # Discover candidate files by supported format extensions
    supported_formats = (".csv", ".arff", ".json", ".xls", ".xlsx", ".parquet", ".txt")
    candidate_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith(supported_formats)]
    
    if not candidate_files:
        logger.warning(f"No supported data files found in {data_dir}")
        return pd.DataFrame(), None, None

    logger.debug(f"Discovered {len(candidate_files)} candidate files for processing")

    # Process files to find new or modified data
    for filename in candidate_files:
        file_path = Path(data_dir) / filename
        current_mtime = os.path.getmtime(file_path)

        # Incremental processing logic: check if file is new or has been modified
        if filename not in records or int(current_mtime) > int(records.get(filename, 0)):
            try:
                # Attempt to load the new/modified file
                df = load_file(file_path, delimiter)
                logger.info(f"Loaded new/modified file: {filename} ({len(df)} rows, "
                           f"modified: {pd.Timestamp(current_mtime, unit='s')})")
                
                return df, filename, current_mtime
                
            except Exception as e:
                logger.warning(f"Skipping unreadable file {filename}: {e}")
                continue

    # No new files found for processing
    logger.info("No new or modified files detected - all data is up to date")
    return pd.DataFrame(), None, None
