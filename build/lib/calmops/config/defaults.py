# CalmOps Default Configuration Constants
# Centralized configuration for magic numbers, thresholds, and default values

# =========================
# Drift Detection Constants
# =========================

# Statistical Test Configuration
DRIFT_DETECTION = {
    "statistical_tests": {
        "alpha": 0.05,  # Significance level for hypothesis tests (KS, Mann-Whitney, CVM)
        "psi_threshold": 0.10,  # Population Stability Index threshold
        "psi_num_bins": 10,  # Number of bins for PSI calculation
        "hellinger_threshold": 0.10,  # Hellinger distance threshold
        "hellinger_num_bins": 30,  # Number of bins for Hellinger distance
        "emd_adaptive_factor": 0.1,  # Factor for adaptive EMD threshold (0.1 * std_ref)
    },
    "majority_voting": {
        "performance_threshold": 0.5,  # 50% of metrics must fail for drift detection
        "statistical_threshold": 0.5,  # 50% of tests must agree for drift detection
    },
    "comparative_analysis": {
        "degradation_ratio": 0.30,  # 30% performance degradation triggers rollback
    }
}

# =========================
# Model Training Constants
# =========================

TRAINING = {
    "data_split": {
        "test_size": 0.2,  # 20% for evaluation, 80% for training
        "random_state_default": 42,
    },
    "retraining_modes": {
        "replay_fraction_default": 0.4,  # 40% old data in replay mix mode
        "window_size_default": 1000,  # Default window size for windowed retraining
    },
    "cross_validation": {
        "stacking_cv_folds": 5,  # CV folds for stacking ensemble
        "calibration_cv": "prefit",  # Calibration strategy
    },
    "ensemble": {
        "max_restarts": 10,  # Maximum PM2 restarts
        "logistic_regression_max_iter": 200,  # LR max iterations for stacking
        "ridge_alphas": [0.1, 1.0, 10.0],  # Ridge regression alpha values
    }
}

# =========================
# Monitoring System Constants
# =========================

MONITORING = {
    "file_formats": [
        ".arff", ".csv", ".txt", ".xml", ".json", 
        ".parquet", ".xls", ".xlsx"
    ],
    "ports": {
        "streamlit_default": 8501,
        "streamlit_fallback": 8510,
    },
    "scheduling": {
        "misfire_grace_time": 300,  # 5 minutes grace time for missed jobs
        "heartbeat_interval": 30,  # 30 seconds between monitor heartbeats
        "streamlit_shutdown_timeout": 10,  # 10 seconds to gracefully shutdown Streamlit
    },
    "circuit_breaker": {
        "default_failure_threshold": 3,  # Failed attempts before circuit opens
        "default_recovery_timeout": 300,  # 5 minutes before retry
    }
}

# =========================
# Dashboard Configuration
# =========================

DASHBOARD = {
    "performance": {
        "default_max_points": 50000,  # Max points for visualization sampling
        "violin_max_points": 10000,  # Max points for violin plots
        "ecdf_resolution_default": 512,  # ECDF quantile points
        "histogram_bins_default": 40,  # Default histogram bins
        "heatmap_top_k_default": 50,  # Default top K features in heatmap
    },
    "cache": {
        "show_spinner": False,  # Disable spinner for cached operations
    },
    "auto_refresh": {
        "check_interval": 1,  # Check for file changes every second
    }
}

# =========================
# File System Configuration
# =========================

FILE_SYSTEM = {
    "encoding": "utf-8",  # Default file encoding
    "control_file": {
        "name": "control_file.txt",
        "previous_data": "previous_data.csv",
    },
    "directories": {
        "models": "models",
        "control": "control", 
        "logs": "logs",
        "metrics": "metrics",
        "config": "config",
        "candidates": "candidates",
    }
}

# =========================
# Logging Configuration
# =========================

LOGGING = {
    "formats": {
        "default": "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        "pipeline": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    },
    "files": {
        "general": "pipeline.log",
        "errors": "pipeline_errors.log", 
        "warnings": "pipeline_warnings.log",
    },
    "rotation": {
        "max_bytes": 10 * 1024 * 1024,  # 10MB per log file
        "backup_count": 5,  # Keep 5 backup files
    },
    "levels": {
        "production": "INFO",
        "development": "DEBUG",
        "noisy_libraries": "WARNING",  # TensorFlow, urllib3, etc.
    }
}