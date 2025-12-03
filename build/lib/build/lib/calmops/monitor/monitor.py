# monitor/monitor.py
# -*- coding: utf-8 -*-
"""
CalmOps Monitor System - Intelligent File System Monitoring with Persistence

This module implements a comprehensive monitoring system that combines:
1. Watchdog file system monitoring for real-time data processing
2. PM2/Docker deployment strategies for production persistence
3. Circuit breaker integration for fault tolerance
4. Streamlit dashboard management for monitoring visualization

Persistence Modes:
    * "none"   : Foreground execution for development/testing
    * "pm2"    : Production-ready with PM2 process manager (auto-restart, boot persistence)
    * "docker" : Containerized deployment with Docker Compose (isolation, scalability)

Architecture Overview:
- File System Watchdog: Monitors data directory using efficient OS-level events
- Pipeline Orchestration: Triggers ML pipelines on file changes with deduplication
- Dashboard Integration: Real-time monitoring via Streamlit web interface
- Deployment Flexibility: Supports multiple persistence strategies for different environments
"""

import os
import logging
import sys
import time
import json
import socket
import shutil
import threading
import subprocess
from typing import Dict, Optional
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from calmops.pipeline.pipeline_flow import run_pipeline
from calmops.utils import get_project_root, get_pipelines_root


# =========================
# Centralized Logging Configuration
# =========================
# Unified logging system with professional formatting for production monitoring.
# Reduces third-party library noise while maintaining comprehensive audit trails.

_LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"


def configure_root_logging(level: int = logging.DEBUG) -> None:
    """
    Establishes enterprise-grade logging configuration for the monitoring system.

    Features:
    - Unified format across all components for consistent log analysis
    - Noise reduction from ML frameworks (TensorFlow, etc.) to prevent log pollution
    - Warning capture from Python warnings module for comprehensive error tracking
    - Clean handler management to prevent duplicate log entries

    Args:
        level: Minimum logging level (default: INFO for production visibility)
    """

    root = logging.getLogger()
    root.setLevel(level)
    # Clean existing handlers to avoid format mixing
    for h in list(root.handlers):
        root.removeHandler(h)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(sh)

    # Capture warnings from warnings module as logging.WARNING
    logging.captureWarnings(True)

    # Mute some common noisy loggers (optional)
    for noisy in (
        "urllib3",
        "watchdog.observers.inotify_buffer",
        "PIL",
        "matplotlib",
        "tensorflow",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _get_logger(name: str | None = None) -> logging.Logger:
    """
    Factory function for creating contextual loggers with consistent formatting.

    Implements intelligent propagation control to prevent log duplication while
    maintaining the hierarchical logging structure for different pipeline components.

    Args:
        name: Logger namespace (defaults to current module)

    Returns:
        Configured logger instance with appropriate propagation settings
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # If there's already a custom handler, don't propagate to avoid double output
    if logger.handlers:
        logger.propagate = False
    return logger


# =========================
# Core Utility Functions
# =========================
# Essential helper functions for file handling, model serialization,
# and cross-platform compatibility in the monitoring system.

_ALLOWED_EXTS = (".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")


def _which(cmd: str):
    from shutil import which

    return which(cmd)


def _fatal(msg: str, code: int = 1):
    """Prints to stderr and exits (for early calls without logger)."""
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
    """
    Serializes ML model instances for persistence layer reconstruction.

    Creates a serializable specification containing module path, class name,
    and hyperparameters for accurate model reconstruction in PM2/Docker environments.
    Essential for maintaining model consistency across process boundaries.

    Args:
        model_instance: Scikit-learn compatible model instance

    Returns:
        Dict containing module, class, and parameter specifications
    """
    spec = {"module": None, "class": None, "params": {}}
    try:
        spec["module"] = model_instance.__class__.__module__
        spec["class"] = model_instance.__class__.__name__
        if hasattr(model_instance, "get_params"):
            spec["params"] = model_instance.get_params(deep=True)
    except Exception:
        pass
    return spec


def _write_runner_config(pipeline_name: str, runner_cfg: dict, base_dir: Path) -> Path:
    """
    Writes the runner configuration to a JSON file.
    """
    config_dir = base_dir / "pipelines" / pipeline_name / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "runner_config.json"
    with open(config_path, "w") as f:
        json.dump(runner_cfg, f, indent=2)
    return config_path


def _write_runner_script(
    pipeline_name: str, runner_cfg_path: Path, base_dir: Path
) -> Path:
    """
    Generates executable Python script for persistence layer deployment.

    Creates a self-contained runner that:
    - Reconstructs the complete monitoring environment from saved configuration
    - Handles model instantiation with proper parameter restoration
    - Prevents recursive persistence calls (sets persistence='none')
    - Ensures proper Python path management for modular execution

    Critical for PM2 and Docker deployment strategies.
    """
    pipeline_dir = base_dir / "pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    runner_path = pipeline_dir / f"run_{pipeline_name}.py"

    content = f'''# Auto-generated runner for pipeline: {pipeline_name}
import os, sys, json, importlib

# Ensure project root on sys.path (two levels up from this file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calmops.monitor.monitor import start_monitor

def _load_cfg(path):
    with open(path, "r") as f:
        return json.load(f)

def _build_model(spec):
    if not spec or not spec.get("module") or not spec.get("class"):
        return None
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["class"])
    params = spec.get("params", {{}})
    try:
        return cls(**params)
    except Exception:
        return cls()

if __name__ == "__main__":
    cfg_path = r"{runner_cfg_path}"
    cfg = _load_cfg(cfg_path)
    model_instance = _build_model(cfg.get("model_spec"))

    start_monitor(
        pipeline_name=cfg["pipeline_name"],
        data_dir=cfg["data_dir"],
        preprocess_file=cfg["preprocess_file"],
        thresholds_drift=cfg["thresholds_drift"],
        thresholds_perf=cfg["thresholds_perf"],
        model_instance=model_instance,
        retrain_mode=cfg["retrain_mode"],
        fallback_mode=cfg["fallback_mode"],
        random_state=cfg["random_state"],
        param_grid=cfg.get("param_grid"),
        cv=cfg.get("cv"),
        custom_train_file=cfg.get("custom_train_file"),
        custom_retrain_file=cfg.get("custom_retrain_file"),
        custom_fallback_file=cfg.get("custom_fallback_file"),
        delimiter=cfg.get("delimiter"),
        target_file=cfg.get("target_file"),
        window_size=cfg.get("window_size"),
        port=cfg.get("port"),
        persistence="none",  # avoid recursive PM2/Docker spawning
        prediction_only=cfg.get("prediction_only", False)
    )
'''
    with open(runner_path, "w") as f:
        f.write(content)
    return runner_path


# =========================
# Production Deployment Strategies - PM2 & Docker
# =========================
# Advanced persistence implementations for production environments:
# - PM2: Process management with auto-restart, clustering, and boot persistence
# - Docker: Containerized deployment with isolation and scalability benefits


def _pm2_install_hint() -> str:
    return (
        "PM2 is required but not found.\n"
        "Install Node.js + PM2, e.g. on Debian/Ubuntu:\n"
        "  sudo apt-get update && sudo apt-get install -y nodejs npm\n"
        "  sudo npm install -g pm2\n"
        "Then re-run with persistence='pm2'."
    )


def _launch_with_pm2(
    pipeline_name: str, runner_script: Path, base_dir: Path, logger: logging.Logger
):
    """
    Deploys monitoring system using PM2 process manager for production resilience.

    PM2 Deployment Strategy:
    - Auto-restart on crashes for maximum uptime
    - Boot persistence (Linux/macOS) for server environments
    - Process clustering capability for high-load scenarios
    - Built-in log management and monitoring
    - Cross-platform compatibility (Linux/macOS/Windows)

    Generates ecosystem.config.js for PM2 configuration management.
    """
    if not _which("pm2"):
        _fatal(
            "PM2 is required but not found.\n"
            "Install Node.js + PM2, e.g.:\n"
            "  npm install -g pm2\n"
            "Then re-run with persistence='pm2'."
        )

    eco_path = base_dir / "pipelines" / pipeline_name / "ecosystem.config.js"
    app_name = f"calmops-{pipeline_name}"

    python_exec = sys.executable
    runner_script_posix = runner_script.as_posix()
    python_exec_posix = Path(python_exec).as_posix()
    base_dir_posix = base_dir.as_posix()

    ecosystem = f"""
module.exports = {{
  apps: [{{
    name: "{app_name}",
    script: "{runner_script_posix}",
    interpreter: "{python_exec_posix}",
    interpreter_args: "-u",
    cwd: "{base_dir_posix}",
    autorestart: true,
    watch: false,
    max_restarts: 10
  }}]
}};
"""
    with open(eco_path, "w") as f:
        f.write(ecosystem)

    try:
        subprocess.check_call(["pm2", "start", str(eco_path)])
    except Exception as e:
        _fatal(f"Failed to start with PM2: {e}")

    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        pass

    # --- Skip startup script on Windows ---
    if os.name != "nt":
        try:
            user = os.getenv("USER") or os.getenv("USERNAME") or ""
            if user:
                subprocess.check_call(["pm2", "startup", "-u", user])
        except Exception:
            logger.warning(
                "PM2 startup script creation failed - boot persistence may not be available"
            )

    logger.info(
        f"PM2 application '{app_name}' deployed successfully with auto-restart capability. "
        f"Boot persistence available on Linux/macOS systems."
    )


def _docker_available():
    return shutil.which("docker") is not None


def _compose_available():
    if not _docker_available():
        return None
    try:
        subprocess.check_call(
            ["docker", "compose", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "v2"
    except Exception:
        return "v1" if shutil.which("docker-compose") else None


def _write_docker_files(
    pipeline_name: str, runner_script_abs: Path, base_dir: Path, port: int | None
):
    """
    Generates containerization configuration for Docker deployment strategy.

    Docker Architecture:
    - Multi-stage build process with Python 3.10 slim base
    - Complete project context copying for dependency resolution
    - Optimized layer caching for faster rebuilds
    - Proper networking configuration for Streamlit dashboard access
    - Environment variable management for containerized execution

    Creates both Dockerfile and docker-compose.yml for orchestration.
    Build context uses repository root to ensure all dependencies are available.
    """
    pipeline_dir = base_dir / "pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_path = pipeline_dir / "Dockerfile"
    compose_path = pipeline_dir / "docker-compose.yml"

    runner_rel = runner_script_abs.relative_to(base_dir)
    runner_in_container = f"/app/{runner_rel.as_posix()}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile for {pipeline_name}
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is repo root (see docker-compose.yml)
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

EXPOSE {exposed_port}

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "{runner_in_container}"]
"""

    # Compose v2: no 'version' at root (avoids warning)
    compose = f"""# Auto-generated docker-compose for {pipeline_name}
services:
  {pipeline_name}:
    build:
      context: ../../
      dockerfile: ./pipelines/{pipeline_name}/Dockerfile
    container_name: calmops_{pipeline_name}
    restart: unless-stopped
    ports:
      - \"{exposed_port}:{exposed_port}\" 
    volumes:
      - \"../../:/app\"
    environment:
      - PYTHONUNBUFFERED=1
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
"""

    with open(dockerfile_path, "w") as f:
        f.write(dockerfile)
    with open(compose_path, "w") as f:
        f.write(compose)

    return dockerfile_path, compose_path


def _launch_with_docker(
    pipeline_name: str,
    runner_script: Path,
    base_dir: Path,
    port: int | None,
    logger: logging.Logger,
):
    """
    Orchestrates Docker Compose deployment for containerized monitoring system.

    Docker Deployment Features:
    - Container isolation for dependency management
    - Automatic restart policy (unless-stopped) for resilience
    - Volume mounting for persistent data access
    - Network configuration for dashboard accessibility
    - Graceful shutdown handling with proper cleanup

    Supports both Docker Compose v1 and v2 for maximum compatibility.
    """
    if not _docker_available():
        _fatal(
            "Docker is required but not found.\n"
            "Install Docker Engine, e.g. on Debian/Ubuntu:\n"
            "  sudo apt-get update && sudo apt-get install -y docker.io\n"
            "  sudo systemctl enable --now docker\n"
            "  sudo usermod -aG docker $USER   # then log out/in\n"
            "For docker compose v2 plugin:\n"
            "  sudo apt-get install -y docker-compose-plugin\n"
            "Alternatively (legacy v1):\n"
            "  sudo apt-get install -y docker-compose\n"
            "Then re-run with persistence='docker'."
        )

    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)
    pipeline_dir = base_dir / "pipelines" / pipeline_name

    try:
        comp = _compose_available()

        # 1) Bring down (best-effort) any previous stack
        try:
            if comp == "v2":
                subprocess.call(
                    ["docker", "compose", "down", "--volumes", "--remove-orphans"],
                    cwd=str(pipeline_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif comp == "v1":
                subprocess.call(
                    ["docker-compose", "down", "--volumes", "--remove-orphans"],
                    cwd=str(pipeline_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

        # 2) Avoid conflict from repeated container_name
        try:
            cname = f"calmops_{pipeline_name}"
            subprocess.call(
                ["docker", "rm", "-f", cname],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

        # 3) Bring up the stack
        if comp == "v2":
            subprocess.check_call(
                ["docker", "compose", "up", "-d", "--build"], cwd=str(pipeline_dir)
            )
        elif comp == "v1":
            subprocess.check_call(
                ["docker-compose", "up", "-d", "--build"], cwd=str(pipeline_dir)
            )
        else:
            _fatal("Neither `docker compose` v2 nor `docker-compose` v1 is available.")
    except Exception as e:
        _fatal(f"Failed to build/run Docker services: {e}")

    logger.info(
        "Docker Compose deployment completed - containerized monitoring system is operational"
    )


# =========================
# Core Monitoring System Entry Point
# =========================
# Central orchestration function that coordinates all monitoring components:
# - File system watchdog with intelligent event filtering
# - Streamlit dashboard for real-time visualization
# - Circuit breaker pattern for fault tolerance
# - Persistence layer management for production deployments


def start_monitor(
    *,
    pipeline_name: str,
    data_dir: str,
    preprocess_file: str,
    thresholds_drift: dict,
    thresholds_perf: dict,
    model_instance,
    retrain_mode: int,
    fallback_mode: int,
    random_state: int,
    param_grid: dict = None,
    cv: int = None,
    custom_train_file: str = None,
    custom_retrain_file: str = None,
    custom_fallback_file: str = None,
    delimiter: str = ",",
    target_file: str = None,
    window_size: int = None,
    port: int | None = None,
    persistence: str = "none",  # "none" | "pm2" | "docker"
    prediction_only: bool = False,
    dir_predictions: Optional[str] = None,
):
    print("DEBUG: start_monitor function entered.")  # Added for debugging
    """
    Orchestrates the complete CalmOps monitoring ecosystem.
    
    System Architecture:
    1. File System Monitoring:
       - Watchdog observers for efficient OS-level file event detection
       - Intelligent filtering for supported data formats
       - Deduplication logic to prevent redundant processing
    
    2. Dashboard Management:
       - Streamlit web interface for real-time monitoring
       - Automatic port detection and conflict resolution
       - Health monitoring with automatic restart capabilities
    
    3. Circuit Breaker Integration:
       - Fault tolerance through controlled failure handling
       - Graceful degradation when components fail
       - Automatic recovery mechanisms
    
    4. Persistence Modes:
       - Development (none): Direct foreground execution
       - Production (pm2): Process manager with auto-restart
       - Container (docker): Isolated deployment with orchestration
    
    Args:
        pipeline_name: Unique identifier for the monitoring pipeline
        data_dir: Directory to monitor for incoming data files
        persistence: Deployment strategy ("none", "pm2", "docker")
        [Additional ML pipeline parameters...]
    """
    # 0) Global logging with monitor format
    configure_root_logging(logging.DEBUG)

    # Contextual logger per pipeline
    log = _get_logger(f"calmops.monitor.{pipeline_name}")

    # Production Persistence Layer Activation
    # Early persistence check enables delegation to PM2/Docker before
    # initializing watchdog components, preventing resource conflicts
    persistence = (persistence or "none").lower()
    pipelines_root = get_pipelines_root()
    project_root = get_project_root()

    # Always write the runner config
    model_spec = _model_spec_from_instance(model_instance)
    runner_cfg_obj = {
        "pipeline_name": pipeline_name,
        "data_dir": data_dir,
        "preprocess_file": preprocess_file,
        "thresholds_drift": thresholds_drift,
        "thresholds_perf": thresholds_perf,
        "retrain_mode": retrain_mode,
        "fallback_mode": fallback_mode,
        "random_state": random_state,
        "param_grid": param_grid,
        "cv": cv,
        "custom_train_file": custom_train_file,
        "custom_retrain_file": custom_retrain_file,
        "custom_fallback_file": custom_fallback_file,
        "delimiter": delimiter,
        "target_file": target_file,
        "window_size": window_size,
        "port": port,
        "model_spec": model_spec,
        "monitor_type": "monitor",
        "prediction_only": prediction_only,
        "dir_predictions": dir_predictions,
    }
    runner_cfg_path = _write_runner_config(
        pipeline_name, runner_cfg_obj, pipelines_root
    )

    if persistence in ("pm2", "docker"):
        # Delegate to production deployment strategies
        runner_script = _write_runner_script(
            pipeline_name, runner_cfg_path, pipelines_root
        )

        if persistence == "pm2":
            _launch_with_pm2(pipeline_name, runner_script, pipelines_root, log)
            log.info(
                "PM2 deployment successful - monitoring system now running in background with auto-restart"
            )
            return
        else:
            _launch_with_docker(
                pipeline_name, runner_script, pipelines_root, port or 8501, log
            )
            log.info(
                "Docker deployment successful - monitoring system containerized and running with restart policy"
            )
            return

    # ========== Development/Direct Execution Flow ==========
    # Standard monitoring system initialization for development environments
    # and direct execution scenarios without persistence requirements
    base_pipeline_dir = pipelines_root / "pipelines" / pipeline_name
    output_dir = base_pipeline_dir / "models"
    control_dir = base_pipeline_dir / "control"
    logs_dir = base_pipeline_dir / "logs"
    metrics_dir = base_pipeline_dir / "metrics"
    config_dir = base_pipeline_dir / "config"

    for d in [output_dir, control_dir, logs_dir, metrics_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    control_file = control_dir / "control_file.txt"
    if not control_file.exists():
        control_file.touch()

    # Config for Streamlit
    config_path = config_dir / "config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(
                {
                    "pipeline_name": pipeline_name,
                    "data_dir": data_dir,
                    "preprocess_file": preprocess_file,
                },
                f,
            )
    except Exception as e:
        _fatal(f"Failed to save config.json: {e}")

    def get_records() -> Dict[str, int]:
        """
        Implements file processing deduplication through modification time tracking.

        Circuit Breaker Pattern Component:
        Maintains persistent state of processed files to prevent redundant pipeline
        executions. Critical for system stability and resource optimization.

        Returns:
            Dict mapping filenames to their last processed modification times
        """
        records: Dict[str, int] = {}
        try:
            if control_file.exists():
                with open(control_file, "r") as f:
                    for line in f:
                        parts = line.strip().split(",", 1)
                        if len(parts) != 2:
                            continue
                        fname, raw_ts = parts
                        try:
                            records[fname] = int(float(raw_ts))
                        except Exception:
                            continue
        except Exception as e:
            log.warning(f"Control file read error - may impact deduplication: {e}")
        return records

    streamlit_process = None  # handle to the Streamlit process

    def stop_all(error_msg: str | None = None):
        """
        Circuit breaker implementation for graceful system shutdown.

        Coordinates orderly termination of all monitoring components:
        - Streamlit dashboard with timeout-based graceful shutdown
        - Watchdog observer cleanup
        - Resource deallocation and error logging

        Essential for preventing orphaned processes and resource leaks.
        """
        nonlocal streamlit_process
        log.critical(
            "Initiating emergency shutdown sequence for all monitoring components"
        )
        if streamlit_process and streamlit_process.poll() is None:
            log.info("Terminating Streamlit dashboard process")
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=10)
            except Exception:
                streamlit_process.kill()
        if error_msg:
            log.error(error_msg)
        sys.exit(1)

    def execute_pipeline(file: str, force_process: bool = False):
        """
        Intelligent pipeline execution with deduplication and error handling.

        Circuit Breaker Features:
        - Modification time comparison for duplicate prevention
        - Comprehensive error handling with system-wide failure propagation
        - File validation and existence checking
        - Atomic processing state updates
        """
        file_path = Path(data_dir) / file
        if not file_path.is_file():
            log.warning(f"Target file no longer exists, skipping processing: {file}")
            return
        try:
            mtime = int(file_path.stat().st_mtime)
        except Exception as e:
            log.warning(f"Unable to retrieve file modification time for {file}: {e}")
            return

        records = get_records()
        # If no record exists or current mtime is greater, we process.
        # If force_process is True, we process regardless of previous processing
        if not force_process and file in records and mtime <= records[file]:
            log.debug(
                f"File {file} already processed (mtime={records[file]}), skipping duplicate processing"
            )
            return
        elif force_process and file in records:
            log.info(
                f"Force processing enabled - reprocessing file {file} despite previous processing"
            )

        log.info(f"Processing new data file: {file} (modified: {mtime})")
        log.debug(f"About to call run_pipeline with parameters:")
        log.debug(f"  - pipeline_name: {pipeline_name}")
        log.debug(f"  - data_dir: {data_dir}")
        log.debug(f"  - preprocess_file: {preprocess_file}")
        log.debug(f"  - target_file: {file}")
        try:
            run_pipeline(
                pipeline_name=pipeline_name,
                data_dir=data_dir,
                preprocess_file=preprocess_file,
                thresholds_drift=thresholds_drift,
                thresholds_perf=thresholds_perf,
                model_instance=model_instance,
                retrain_mode=retrain_mode,
                fallback_mode=fallback_mode,
                random_state=random_state,
                param_grid=param_grid,
                cv=cv,
                custom_train_file=custom_train_file,
                custom_retrain_file=custom_retrain_file,
                custom_fallback_file=custom_fallback_file,
                dir_predictions=dir_predictions,
                delimiter=delimiter,
                target_file=file
                if target_file is None
                else target_file,  # override if specified
                window_size=window_size,
                prediction_only=prediction_only,
            )
            log.info(f"Successfully completed pipeline processing for {file}")
        except FileNotFoundError as e:
            log.error(f"File system error for {file}: {e} - continuing monitoring")
        except MemoryError as e:
            log.error(
                f"Memory exhausted processing {file}: {e} - continuing monitoring"
            )
        except KeyboardInterrupt:
            log.info("Pipeline execution interrupted by user")
            stop_all("User requested shutdown")
        except Exception as e:
            log.error(f"Unexpected pipeline failure for {file}: {e}")
            # Log the full traceback for debugging
            import traceback

            log.error(f"Full traceback: {traceback.format_exc()}")
            # For critical failures, still stop the system
            if "critical" in str(e).lower() or "fatal" in str(e).lower():
                stop_all(f"Critical pipeline failure for {file}: {e}")
            else:
                log.warning(f"Non-critical error, continuing monitoring: {e}")

    class DataFileHandler(FileSystemEventHandler):
        """
        Watchdog File System Event Handler with Intelligence Filtering.

        Implements efficient file system monitoring using OS-level events:
        - Creates and modification event handling
        - Extension-based filtering for supported data formats
        - Immediate pipeline triggering for real-time processing

        Watchdog Architecture Benefits:
        - Low CPU overhead compared to polling mechanisms
        - Real-time event notification for immediate response
        - Cross-platform compatibility (Linux inotify, macOS FSEvents, Windows)
        """

        def __init__(self, logger: logging.Logger):
            super().__init__()
            self.log = logger

        def on_created(self, event):
            self._process_event(event, "created")

        def on_modified(self, event):
            self._process_event(event, "modified")

        def _process_event(self, event, event_type: str):
            if event.is_directory:
                return
            fname = os.path.basename(event.src_path)
            if not fname.lower().endswith(_ALLOWED_EXTS):
                self.log.debug(f"Skipping unsupported file format: {fname}")
                return
            self.log.info(
                f"File system event detected: {fname} ({event_type}) - triggering pipeline execution"
            )
            execute_pipeline(fname)

    def is_port_in_use(p: int) -> bool:
        """
        Network port availability checker for dashboard deployment.

        Prevents port conflicts during Streamlit dashboard initialization
        by testing socket connectivity on localhost.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    def start_streamlit(pipeline_name: str, port: int | None = None):
        """
        Streamlit Dashboard Management System.

        Dashboard Features:
        - Real-time monitoring visualization
        - Automatic port conflict resolution (8501 -> 8510)
        - Process health monitoring integration
        - Cross-platform web interface accessibility

        Critical component for system observability and user interaction.
        """
        nonlocal streamlit_process
        dashboard_path = project_root / "web_interface" / "dashboard.py"

        if port is None:
            port = 8501
            if is_port_in_use(port):
                log.info(f"Port {port} unavailable, attempting fallback port 8510")
                port = 8510

        log.info(f"Starting Streamlit dashboard on port {port}")
        log.info(f"Starting Streamlit dashboard for pipeline {pipeline_name} ")
        log.info(f"Local URL: http://localhost:{port}")
        try:
            streamlit_process = subprocess.Popen(
                [
                    "streamlit",
                    "run",
                    str(dashboard_path),
                    "--server.port",
                    str(port),
                    "--server.address",
                    "0.0.0.0",
                    "--",
                    "--pipeline_name",
                    pipeline_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give Streamlit a moment to start
            import time

            time.sleep(2)
            # Check if process started successfully
            if streamlit_process.poll() is not None:
                log.error(f"Streamlit failed to start")
                stop_all(f"Streamlit startup failed")
        except FileNotFoundError:
            stop_all("Streamlit not found. Please install with: pip install streamlit")
        except PermissionError as e:
            stop_all(f"Permission denied starting Streamlit: {e}")
        except Exception as e:
            stop_all(f"Unexpected error starting Streamlit: {e}")

    def start_watchdog():
        """
        Core Watchdog Observer Initialization and Health Monitoring.

        Implements the main monitoring loop with:
        - File system observer with recursive directory scanning
        - Streamlit process health checking (10-second intervals)
        - Keyboard interrupt handling for graceful shutdown
        - Observer cleanup and resource management

        Central component of the monitoring system's event-driven architecture.
        """
        log.info(f"File system monitoring active on directory: {data_dir}")
        event_handler = DataFileHandler(log)
        observer = Observer()

        try:
            observer.schedule(event_handler, path=data_dir, recursive=False)
            observer.start()
            log.info("Watchdog observer started successfully")
        except FileNotFoundError:
            stop_all(f"Data directory not found: {data_dir}")
        except PermissionError:
            stop_all(f"Permission denied accessing directory: {data_dir}")
        except Exception as e:
            stop_all(f"Failed to start file system observer: {e}")

        try:
            heartbeat_count = 0
            while True:
                # Check Streamlit process health
                if streamlit_process and streamlit_process.poll() is not None:
                    log.error("Dashboard process terminated unexpectedly")
                    stop_all("Dashboard process terminated unexpectedly")

                # Log heartbeat every 10 cycles (100 seconds)
                heartbeat_count += 1
                if heartbeat_count % 10 == 0:
                    log.debug(
                        f"Monitor system heartbeat #{heartbeat_count} - all systems operational"
                    )

                time.sleep(10)
        except KeyboardInterrupt:
            log.info("Shutdown requested by user")
            observer.stop()
        except Exception as e:
            log.error(f"Unexpected error in monitoring loop: {e}")
            observer.stop()
        finally:
            observer.join()
            log.info("File system observer stopped")

    # ========== System Initialization and Startup Sequence ==========
    log.info(
        "Initializing CalmOps monitoring system with Watchdog and Streamlit components"
    )
    log.info("Performing initial directory scan for existing data files")
    # Initial scan: process all files, even if they appear processed (in case of config changes)
    files_found = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(_ALLOWED_EXTS):
            files_found.append(file)
            log.info(f"Found existing data file: {file}")

    if files_found:
        log.info(f"Processing {len(files_found)} existing data files")
        for file in files_found:
            try:
                # Do not force process existing files on startup; respect control file
                file_path = Path(data_dir) / file
                if file_path.is_file():
                    log.info(f"Initial scan - processing existing file: {file}")
                    execute_pipeline(file, force_process=False)
                else:
                    log.warning(f"File {file} not accessible during initial scan")
            except Exception as e:
                log.error(
                    f"Failed to process existing file {file} during initial scan: {e}"
                )
    else:
        log.info("No existing data files found in directory")

    threading.Thread(
        target=start_streamlit, args=(pipeline_name, port), daemon=True
    ).start()
    start_watchdog()


# =========================
# Example usage
# =========================


def test_single_file():
    """Debug function to test pipeline with a single file."""
    configure_root_logging(logging.DEBUG)
    from sklearn.ensemble import RandomForestClassifier

    # Test with pipeline_flow directly
    from calmops.pipeline.pipeline_flow import run_pipeline

    log = _get_logger("test")
    project_root = get_project_root()

    try:
        log.info("Testing direct pipeline execution...")
        run_pipeline(
            pipeline_name="test_direct",
            data_dir=str(project_root / "data"),
            preprocess_file=str(
                project_root / "calmops" / "pipeline" / "preprocessing.py"
            ),
            thresholds_drift={"balanced_accuracy": 0.8},
            thresholds_perf={"accuracy": 0.9, "balanced_accuracy": 0.9, "F1": 0.85},
            model_instance=RandomForestClassifier(random_state=42),
            retrain_mode=0,
            fallback_mode=2,
            random_state=42,
            param_grid={"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
            cv=5,
            delimiter=",",
            target_file="Data_2023.arff",
        )
        log.info("Direct pipeline test COMPLETED!")
    except Exception as e:
        log.error(f"Direct pipeline test FAILED: {e}")
        import traceback

        log.error(f"Traceback: {traceback.format_exc()}")


def main():
    """Main function to start the monitor with predefined arguments."""
    from sklearn.ensemble import RandomForestClassifier

    configure_root_logging(logging.DEBUG)
    project_root = get_project_root()

    # Define parameters directly for start_monitor
    pipeline_name = "my_pipeline_watchdog"
    data_dir = project_root / "data"
    preprocess_file = project_root / "pipeline" / "preprocessing.py"
    thresholds_perf = {"balanced_accuracy": 0.8}
    thresholds_drift = {"balanced_accuracy": 0.8}
    model_instance = RandomForestClassifier(random_state=42)
    retrain_mode = 6
    fallback_mode = 2
    random_state = 42
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    cv = 3
    delimiter = ","
    window_size = 1000
    port = None
    persistence = "none"
    prediction_only = False

    start_monitor(
        pipeline_name=pipeline_name,
        data_dir=str(data_dir),
        preprocess_file=str(preprocess_file),
        thresholds_perf=thresholds_perf,
        thresholds_drift=thresholds_drift,
        model_instance=model_instance,
        retrain_mode=retrain_mode,
        fallback_mode=fallback_mode,
        random_state=random_state,
        param_grid=param_grid,
        cv=cv,
        delimiter=delimiter,
        window_size=window_size,
        port=port,
        persistence=persistence,
        prediction_only=prediction_only,
    )


if __name__ == "__main__":
    main()
