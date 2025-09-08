import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

# Add config path to import defaults
sys.path.append(str(Path(__file__).parent.parent))
from config.defaults import LOGGING


class PipelineLogger:
    def __init__(self, name: str, log_dir: str):
        """
        Initializes the logger for the pipeline.
        Logs will be saved in the specified log_dir.
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)  # Create log directory if it doesn't exist
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """
        Set up the logger with rotating handlers for console, general, error, and warning logs.
        Implements log rotation to prevent disk space issues in production environments.
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # Set base logging level to DEBUG
        logger.handlers.clear()  # Clear any existing handlers

        # Define log format using centralized configuration
        formatter = logging.Formatter(LOGGING["formats"]["pipeline"])

        # Get rotation configuration
        max_bytes = LOGGING["rotation"]["max_bytes"]
        backup_count = LOGGING["rotation"]["backup_count"]

        # Console Handler: For INFO and above level logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Rotating File Handler: For all logs (DEBUG and above)
        file_handler = RotatingFileHandler(
            self.get_general_log_file(),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Rotating Error Handler: For ERROR level logs
        error_handler = RotatingFileHandler(
            self.get_error_log_file(),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # Rotating Warning Handler: For WARNING level logs
        warning_handler = RotatingFileHandler(
            self.get_warning_log_file(),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        warning_handler.setLevel(logging.WARNING)
        warning_handler.setFormatter(formatter)
        logger.addHandler(warning_handler)

        return logger

    def get_general_log_file(self):
        """
        Returns the path for the general log file.
        """
        return self.log_dir / LOGGING["files"]["general"]

    def get_error_log_file(self):
        """
        Returns the path for the error log file.
        """
        return self.log_dir / LOGGING["files"]["errors"]

    def get_warning_log_file(self):
        """
        Returns the path for the warning log file.
        """
        return self.log_dir / LOGGING["files"]["warnings"]

    def get_logger(self):
        """
        Returns the logger instance to be used for logging events.
        """
        return self.logger
