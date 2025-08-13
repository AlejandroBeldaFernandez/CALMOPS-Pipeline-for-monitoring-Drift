import logging
from pathlib import Path


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
        Set up the logger with handlers for console, general, error, and warning logs.
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # Set base logging level to DEBUG
        logger.handlers.clear()  # Clear any existing handlers

        # Define log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )

        # Console Handler: For INFO and above level logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler: For all logs (DEBUG and above)
        file_handler = logging.FileHandler(self.get_general_log_file(), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Error Handler: For ERROR level logs
        error_handler = logging.FileHandler(self.get_error_log_file(), encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # Warning Handler: For WARNING level logs
        warning_handler = logging.FileHandler(self.get_warning_log_file(), encoding='utf-8')
        warning_handler.setLevel(logging.WARNING)
        warning_handler.setFormatter(formatter)
        logger.addHandler(warning_handler)

        return logger

    def get_general_log_file(self):
        """
        Returns the path for the general log file.
        """
        return self.log_dir / 'pipeline.log'

    def get_error_log_file(self):
        """
        Returns the path for the error log file.
        """
        return self.log_dir / 'pipeline_errors.log'

    def get_warning_log_file(self):
        """
        Returns the path for the warning log file.
        """
        return self.log_dir / 'pipeline_warnings.log'

    def get_logger(self):
        """
        Returns the logger instance to be used for logging events.
        """
        return self.logger
