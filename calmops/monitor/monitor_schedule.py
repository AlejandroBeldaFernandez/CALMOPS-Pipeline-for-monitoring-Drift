# monitor/monitor_schedule.py
# -*- coding: utf-8 -*-
"""
CalmOps Monitor - Scheduled Pipeline Execution System

This module implements a comprehensive monitoring system that periodically checks data directories
and executes ML pipelines using APScheduler. It provides multiple deployment architectures:

Scheduling Engine:
  - APScheduler integration with interval, cron, and date-based triggers
  - Robust job management with coalescing and misfire handling
  - Timezone-aware scheduling with Europe/Madrid as default

Persistence Architectures:
  - "none"   : Foreground execution with direct process management
  - "pm2"    : Process Manager 2 integration for auto-restart and boot persistence
  - "docker" : Containerized deployment with docker-compose orchestration

Pipeline Management:
  - File-based change detection with modification time tracking
  - Streamlit dashboard integration for real-time monitoring
  - Comprehensive error handling and graceful shutdown mechanisms
"""

import os
import sys
import time
import json
import socket
import shutil
import threading
import subprocess
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from calmops.utils import get_project_root, get_pipelines_root

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo

    _TZ_EUROPE_MADRID = ZoneInfo("Europe/Madrid")
except Exception:
    _TZ_EUROPE_MADRID = None  # fallback to system tz

# Keep imports relative to repo root (run as: python -m monitor.monitor_schedule)
from calmops.pipeline.pipeline_stream import run_pipeline


# =========================
# Unified Logging Configuration
# =========================

_LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"


def configure_root_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with unified formatting and reduced verbosity.

    Sets up a single StreamHandler to stdout with monitor-specific formatting,
    cleans any existing handlers, captures warnings, and reduces noise from
    third-party libraries (TensorFlow, APScheduler, PIL, matplotlib).

    Args:
        level: Logging level (default: INFO)
    """

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(sh)

    logging.captureWarnings(True)
    for noisy in ("urllib3", "apscheduler", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _get_logger(name: str | None = None) -> logging.Logger:
    """
    Create a logger that inherits root handler configuration.

    Returns a logger with proper inheritance from root configuration
    without adding duplicate handlers.

    Args:
        name: Logger name (defaults to current module)

    Returns:
        Configured logger instance
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Prevent propagation if handlers already exist to avoid duplicate output
    if logger.handlers:
        logger.propagate = False
    return logger


# Default logger instance (can be reconfigured with pipeline-specific name)
log = _get_logger()
# Reduce APScheduler verbosity to focus on application-level events
logging.getLogger("apscheduler").setLevel(logging.WARNING)


# =========================
# Core Utility Functions
# =========================


def _which(cmd: str):
    """Shorthand for shutil.which."""
    from shutil import which

    return which(cmd)


def _fatal(msg: str, code: int = 1):
    """Log a fatal message and exit with given code."""
    log.critical(msg)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
    """
    Serialize ML model specifications for persistence layer reconstruction.

    Extracts model class information and parameters from sklearn-compatible models
    to enable proper model reconstruction in PM2/Docker environments. Falls back
    to empty constructor if parameter introspection is not available.

    Args:
        model_instance: ML model instance (sklearn-compatible)

    Returns:
        dict: Model specification with module, class, and parameters
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


def _write_runner_config(pipeline_name: str, config_obj: dict, base_dir: Path) -> str:
    """
    Generate JSON configuration for persistence layer runners.

    Creates a JSON configuration file containing all necessary parameters
    for PM2 and Docker runners to reconstruct the complete pipeline execution
    environment with proper model instantiation.

    Args:
        pipeline_name: Unique pipeline identifier
        config_obj: Complete configuration dictionary
        base_dir: Project root directory

    Returns:
        str: Path to generated configuration file
    """
    cfg_dir = base_dir / "pipelines" / pipeline_name / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "runner_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return str(cfg_path)


def _write_runner_script_schedule(
    pipeline_name: str, runner_cfg_path: str, base_dir: Path
) -> str:
    """
    Create a runner script that reconstructs the model & arguments and calls start_monitor_schedule
    with persistence='none' to avoid recursion when PM2/Docker relaunch.
    """
    pipeline_dir = base_dir / "pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    runner_path = pipeline_dir / f"run_{pipeline_name}_schedule.py"

    header = f"# Auto-generated runner (schedule) for pipeline: {pipeline_name}"
    body = f"""
import os, sys, json, importlib
from calmops.monitor.monitor_schedule import start_monitor_schedule

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

    start_monitor_schedule(
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
        schedule=cfg.get("schedule"),
        window_size=cfg.get("window_size"),
        early_start=cfg.get("early_start", False),
        port=cfg.get("port"),
        persistence="none",  # avoid recursive PM2/Docker spawning
        prediction_only=cfg.get("prediction_only", False)
    )
"""
    with open(runner_path, "w") as f:
        f.write(header + body)
    return str(runner_path)


# =========================
# PM2 Process Management Integration
# PM2 provides process persistence, auto-restart, and boot-time startup
# =========================

# PM2 Architecture:
# - Uses ecosystem.config.js for process configuration
# - Provides automatic restart on failure with configurable limits
# - Enables system startup integration via pm2 startup command
# - Supports process monitoring and log aggregation
# - Cross-platform compatibility (Linux/macOS/Windows)


def pm2_list(prefix: str = "calmops-") -> list[dict]:
    """
    Query and display PM2 processes managed by CalmOps.

    Retrieves process information from PM2 daemon and filters by application
    prefix. Displays process status including PID, status, and restart count.

    Args:
        prefix: Process name prefix for filtering (default: "calmops-")

    Returns:
        list[dict]: PM2 process information dictionaries
    """
    if not _which("pm2"):
        log.error("PM2 process manager not found. Install with: npm install -g pm2")
        return []

    try:
        out = subprocess.check_output(["pm2", "jlist"], text=True)
        procs = json.loads(out)
    except Exception as e:
        log.error(f"Could not obtain PM2 list: {e}")
        return []

    filtered = [p for p in procs if p.get("name", "").startswith(prefix)]
    if not filtered:
        log.info(f"No PM2 processes with prefix '{prefix}'.")
        return []

    log.info("PM2 processes:")
    for p in filtered:
        name = p.get("name")
        pid = p.get("pid")
        status = p.get("pm2_env", {}).get("status")
        restart = p.get("pm2_env", {}).get("restart_time")
        log.info(f"- {name} | pid={pid} | status={status} | restarts={restart}")
    return filtered


def pm2_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    """
    Complete pipeline cleanup for PM2-managed processes.

    Performs comprehensive cleanup by stopping the PM2 process, removing it from
    the process registry, saving the configuration, and removing all associated
    files and directories.

    Args:
        pipeline_name: Pipeline identifier to remove
        base_dir: Base directory containing pipeline folders
    """
    app_name = f"calmops-schedule-{pipeline_name}"
    if not _which("pm2"):
        log.error("PM2 process manager not found. Install with: npm install -g pm2")
        return

    try:
        subprocess.call(["pm2", "stop", app_name])
        subprocess.call(["pm2", "delete", app_name])
        subprocess.call(["pm2", "save"])
        log.info(f"PM2 process '{app_name}' successfully terminated and removed")
    except Exception as e:
        log.warning(f"PM2 process cleanup failed: {e}")

    pipeline_path = os.path.join(base_dir, pipeline_name)
    try:
        if os.path.exists(pipeline_path):
            shutil.rmtree(pipeline_path)
            log.info(f"Pipeline directory '{pipeline_path}' successfully removed")
        else:
            log.info(f"Pipeline directory '{pipeline_path}' not found")
    except Exception as e:
        log.error(f"Failed to remove pipeline directory: {e}")


# =========================
# Docker Containerization Integration
# Docker provides isolated execution environment with automatic restart policies
# =========================

# Docker Architecture:
# - Uses multi-stage Dockerfile with Python 3.10 slim base image
# - docker-compose.yml orchestrates container lifecycle and networking
# - Volume mounts enable persistent data access and live code updates
# - Restart policies ensure container resilience and automatic recovery
# - Port mapping exposes Streamlit dashboard to host network
# - Build context optimization minimizes image size and build time


def _docker_install_hint() -> str:
    """
    Provide comprehensive Docker installation guidance.

    Returns detailed installation instructions for Docker Engine and
    docker-compose across different platforms and installation methods.

    Returns:
        str: Multi-line installation instructions
    """
    return (
        "Docker is required but not found.\n"
        "Install Docker Engine, e.g. on Debian/Ubuntu:\n"
        "  sudo apt-get update && sudo apt-get install -y docker.io\n"
        "  sudo systemctl enable --now docker\n"
        "  sudo usermod -aG docker $USER   # then log out/in\n"
        "For docker compose v2 plugin:\n"
        "  sudo apt-get install -y docker-compose-plugin\n"
        "Alternatively (legacy v1):\n"
        "  sudo apt-get install -y docker-compose\n"
    )


def _write_docker_files(
    pipeline_name: str, runner_script_abs: str, base_dir: Path, port: int | None
):
    """
    Generate Docker deployment configuration files.

    Creates optimized Dockerfile and docker-compose.yml with proper build context,
    volume mounts, environment variables, and networking configuration. Uses
    project root as build context to ensure complete repository access.

    Args:
        pipeline_name: Unique pipeline identifier
        runner_script_abs: Absolute path to generated runner script
        base_dir: Project root directory
        port: Streamlit dashboard port (defaults to 8501)

    Returns:
        tuple: Paths to generated Dockerfile and docker-compose.yml
    """
    pipeline_dir = base_dir / "pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_path = pipeline_dir / "Dockerfile"
    compose_path = pipeline_dir / "docker-compose.yml"

    runner_rel = Path(runner_script_abs).relative_to(base_dir)
    runner_in_container = f"/app/{runner_rel.as_posix()}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile (schedule) for {pipeline_name}
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is repo root (see docker-compose.yml)
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    (test -f requirements.txt && pip install --no-cache-dir -r requirements.txt || true)

EXPOSE {exposed_port}

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "{runner_in_container}"]
"""

    compose = f"""# Auto-generated docker-compose (schedule) for {pipeline_name}
version: "3.8"
services:
  {pipeline_name}_schedule:
    build:
      context: ../../
      dockerfile: ./pipelines/{pipeline_name}/Dockerfile
    container_name: calmops_{pipeline_name}_schedule
    restart: unless-stopped
    ports:
      - \"{exposed_port}:{exposed_port}\" 
    volumes:
      - "../../:/app"
    environment:
      - PYTHONUNBUFFERED=1
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
"""

    with open(dockerfile_path, "w") as f:
        f.write(dockerfile)
    with open(compose_path, "w") as f:
        f.write(compose)

    return str(dockerfile_path), str(compose_path)


def _launch_with_docker(
    pipeline_name: str, runner_script: str, base_dir: Path, port: int | None
):
    """
    Deploy pipeline using Docker container orchestration.

    Generates Docker configuration files and launches the containerized pipeline
    in detached mode with automatic restart policies. Supports both docker-compose
    v2 (plugin) and v1 (standalone) installations.

    Args:
        pipeline_name: Unique pipeline identifier
        runner_script: Path to pipeline runner script
        base_dir: Project root directory
        port: Streamlit dashboard port
    """
    if not _which("docker"):
        _fatal(_docker_install_hint())

    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    pipeline_dir = base_dir / "pipelines" / pipeline_name

    try:
        if (
            _which("docker")
            and subprocess.call(
                ["docker", "compose", "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        ):
            subprocess.check_call(
                ["docker", "compose", "up", "-d", "--build"], cwd=str(pipeline_dir)
            )
        elif _which("docker-compose"):
            subprocess.check_call(
                ["docker-compose", "up", "-d", "--build"], cwd=str(pipeline_dir)
            )
        else:
            _fatal(
                "Neither `docker compose` (v2) nor `docker-compose` (v1) is available.\n"
                f"{_docker_install_hint()}"
            )
    except Exception as e:
        _fatal(f"Failed to build/run Docker services: {e}\n{_docker_install_hint()}")


def docker_list(prefix: str = "calmops_") -> list[Tuple[str, str]]:
    """
    Query and display Docker containers managed by CalmOps.

    Retrieves container information and filters by name prefix. Displays
    container names and current status for monitoring purposes.

    Args:
        prefix: Container name prefix for filtering (default: "calmops_")

    Returns:
        list[Tuple[str, str]]: Container (name, status) pairs
    """
    if not _which("docker"):
        log.error("Docker container engine not found")
        return []
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"], text=True
        ).strip()
    except Exception as e:
        log.error(f"Failed to query Docker containers: {e}")
        return []

    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, status = line.split("\t", 1)
        if name.startswith(prefix):
            rows.append((name, status))
    if rows:
        log.info("Active CalmOps containers:")
        for name, status in rows:
            log.info(f"- {name}: {status}")
    else:
        log.info(f"No managed containers found with prefix '{prefix}'")
    return rows


def docker_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    """
    Complete pipeline cleanup for Docker-managed containers.

    Performs comprehensive cleanup by stopping docker-compose services, removing
    volumes and orphaned containers, and cleaning up all associated files.
    Handles both compose plugin and standalone installations.

    Args:
        pipeline_name: Pipeline identifier to remove
        base_dir: Directory containing pipeline configurations
    """
    pipeline_dir = os.path.join(base_dir, pipeline_name)
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")
    if not _which("docker"):
        log.error("Docker container engine not found")
        return

    try:
        if os.path.exists(compose_path):
            if (
                subprocess.call(
                    ["docker", "compose", "version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                == 0
            ):
                subprocess.call(
                    ["docker", "compose", "down", "--volumes", "--remove-orphans"],
                    cwd=pipeline_dir,
                )
            elif _which("docker-compose"):
                subprocess.call(
                    ["docker-compose", "down", "--volumes", "--remove-orphans"],
                    cwd=pipeline_dir,
                )
        else:
            cname = f"calmops_{pipeline_name}_schedule"
            subprocess.call(["docker", "rm", "-f", cname])
        log.info(
            f"Docker containers for pipeline '{pipeline_name}' successfully removed"
        )
    except Exception as e:
        log.warning(f"Docker compose cleanup failed: {e}")

    try:
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)
            log.info(f"Pipeline directory '{pipeline_dir}' successfully removed")
    except Exception as e:
        log.error(f"Failed to remove pipeline directory: {e}")


# =========================
# PM2 Process Deployment and Lifecycle Management
# =========================


def _launch_with_pm2(
    pipeline_name: str,
    runner_script: str,
    base_dir: Path,
    logger: logging.Logger | None = None,
):
    """
    Deploy pipeline using PM2 process manager with comprehensive lifecycle management.

    Creates ecosystem configuration file and launches the pipeline as a managed PM2
    process with automatic restart, monitoring, and optional system startup integration.
    Handles cross-platform compatibility and provides robust error recovery.

    Args:
        pipeline_name: Unique pipeline identifier
        runner_script: Path to pipeline runner script
        base_dir: Project root directory
        logger: Optional logger instance (defaults to calmops.monitor)
    """
    if logger is None:
        import logging

        logger = logging.getLogger("calmops.monitor")

    if not _which("pm2"):
        _fatal(
            "PM2 is required but not found.\n"
            "Install Node.js + PM2, e.g.:\n"
            "  npm install -g pm2\n"
            "Then re-run with persistence='pm2'."
        )

    eco_path = base_dir / "pipelines" / pipeline_name / "ecosystem.schedule.config.js"
    app_name = f"calmops-schedule-{pipeline_name}"
    python_exec = sys.executable

    runner_script_posix = Path(runner_script).as_posix()
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
        _fatal(f"Failed to start with PM2: {e}\n(Install with: npm install -g pm2)")

    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        pass

    # --- Only attempt startup on Linux/macOS ---
    if os.name != "nt":
        try:
            user = os.getenv("USER") or os.getenv("USERNAME") or ""
            if user:
                subprocess.check_call(["pm2", "startup", "-u", user])
            else:
                subprocess.check_call(["pm2", "startup"])
        except Exception:
            logger.warning(
                "PM2 startup configuration failed - manual setup may be required for boot persistence"
            )
    else:
        logger.info(
            "Windows platform detected - boot startup requires manual PM2 configuration"
        )

    logger.info(
        f"PM2 process '{app_name}' successfully deployed with automatic restart enabled"
    )


# =========================
# Core Scheduled Monitoring System
# APScheduler Integration with Robust Job Management
# =========================

# APScheduler Architecture:
# - BackgroundScheduler runs in separate thread for non-blocking operation
# - Job coalescing prevents multiple instances of same job running simultaneously
# - Misfire grace time handles delayed executions due to system load
# - Multiple trigger types: IntervalTrigger, CronTrigger, DateTrigger
# - Timezone-aware scheduling ensures consistent execution across environments
# - Job persistence and recovery mechanisms for system restart scenarios

_ALLOWED_EXTS = (".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")


def start_monitor_schedule(
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
    schedule: dict = None,  # {"type": "interval"|"cron"|"date", "params": {...}}
    window_size=None,
    early_start: bool = False,
    port: int | None = None,
    persistence: str = "none",  # "none" | "pm2" | "docker"
    prediction_only: bool = False,
    dir_predictions: Optional[str] = None,
):
    """
    Launch comprehensive scheduled pipeline monitoring system.

    Core Functionality:
    - File-based change detection with modification time tracking
    - APScheduler integration with configurable trigger types (interval/cron/date)
    - Streamlit dashboard deployment for real-time monitoring
    - Multi-architecture persistence (foreground/PM2/Docker)

    Scheduling Modes:
    - interval: Periodic execution with jitter support for load distribution
    - cron: Unix cron-style scheduling with full expression support
    - date: One-time execution at specific datetime

    Persistence Architectures:
    - none: Direct foreground execution with manual process management
    - pm2: Process Manager 2 with auto-restart and boot persistence
    - docker: Containerized deployment with compose orchestration

    Pipeline Management:
    - Tracks processed files via control file with modification times
    - Supports multiple data formats (CSV, ARFF, JSON, Parquet, Excel)
    - Comprehensive error handling with graceful shutdown procedures
    - Timezone-aware scheduling with robust misfire handling

    Args:
        pipeline_name: Unique identifier for pipeline instance
        data_dir: Directory path for monitoring data files
        preprocess_file: Path to preprocessing script
        thresholds_drift: Performance drift detection thresholds
        thresholds_perf: Model performance evaluation thresholds
        model_instance: ML model instance (sklearn-compatible)
        retrain_mode: Retraining behavior configuration
        fallback_mode: Fallback strategy for failed training
        random_state: Seed for reproducible results
        param_grid: Hyperparameter grid for model tuning
        cv: Cross-validation folds
        custom_train_file: Custom training script path
        custom_retrain_file: Custom retraining script path
        custom_fallback_file: Custom fallback script path
        delimiter: Data file delimiter character
        schedule: Scheduling configuration with type and parameters
        window_size: Data window size for processing
        early_start: Execute initial check before scheduled runs
        port: Streamlit dashboard port number
        persistence: Deployment architecture (none/pm2/docker)
    """
    # Configure unified logging format for monitor operations
    configure_root_logging(logging.INFO)

    # Create pipeline-specific logger for targeted monitoring
    global log
    log = _get_logger(f"calmops.monitor.schedule.{pipeline_name}")

    if schedule is None or "type" not in schedule or "params" not in schedule:
        _fatal(
            "Schedule configuration requires 'type' and 'params' keys with valid scheduling parameters"
        )

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
        "schedule": schedule,
        "window_size": window_size,
        "early_start": early_start,
        "port": port,
        "model_spec": model_spec,
        "monitor_type": "monitor_schedule",
        "prediction_only": prediction_only,
        "dir_predictions": dir_predictions,
    }
    runner_cfg_path = _write_runner_config(
        pipeline_name, runner_cfg_obj, pipelines_root
    )

    if persistence in ("pm2", "docker"):
        # Delegate to production deployment strategies
        runner_script = _write_runner_script_schedule(
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
                pipeline_name, runner_script, pipelines_root, port or 8501
            )
            log.info(
                "Docker deployment successful - monitoring system containerized and running with restart policy"
            )
            return

    # ========== Development/Direct Execution Flow ==========
    # Standard monitoring system initialization for development environments
    # and direct execution scenarios without persistence requirements
    base_pipeline_dir = pipelines_root / "pipelines" / pipeline_name
    output_dir = base_pipeline_dir / "modelos"
    control_dir = base_pipeline_dir / "control"
    logs_dir = base_pipeline_dir / "logs"
    metrics_dir = base_pipeline_dir / "resultados"
    config_dir = base_pipeline_dir / "config"

    for d in [output_dir, control_dir, logs_dir, metrics_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Control file synchronization point for dashboard integration
    control_file = control_dir / "control_file.txt"
    if not control_file.exists():
        control_file.touch()

    # Generate dashboard configuration for pipeline integration
    config_path = config_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "pipeline_name": pipeline_name,
                "data_dir": data_dir,
                "preprocess_file": preprocess_file,
            },
            f,
        )

    streamlit_process = None

    def stop_all(error_msg=None):
        """
        Graceful shutdown procedure for all monitor components.

        Terminates Streamlit dashboard process and performs clean exit
        with optional error message logging.

        Args:
            error_msg: Optional error message for logging
        """
        nonlocal streamlit_process
        log.critical("Initiating graceful shutdown of monitoring system")
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

    def get_records() -> Dict[str, int]:
        """
        Load file processing history from control file.

        Reads the control file containing processed file records with modification
        times to determine which files require processing. Each line contains a
        filename and its last processed modification time.

        Control file format:
            filename.ext,modification_time_epoch

        Returns:
            Dict[str, int]: Mapping of filenames to last processed modification times
        """
        records: Dict[str, int] = {}
        if control_file.exists():
            with open(control_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        try:
                            key = parts[0]
                            records[key] = int(float(parts[1]))
                        except Exception:
                            continue
        return records

    def run_pipeline_for_file(file):
        """
        Execute pipeline processing for a specific data file.

        Launches the complete ML pipeline for the given file with all configured
        parameters, error handling, and processing completion tracking.

        Args:
            file: Filename (basename) to process
        """
        file_path = Path(data_dir) / file
        mtime = int(file_path.stat().st_mtime)
        log.info(f"Executing pipeline for file '{file}' (modified: {mtime})")

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
                delimiter=delimiter,
                target_file=file,  # basename (control_file only uses the name)
                window_size=window_size,
                prediction_only=prediction_only,
                dir_predictions=dir_predictions,
            )
            log.info(f"Pipeline processing completed successfully for '{file}'")
        except Exception as e:
            stop_all(f"Critical pipeline failure for '{file}': {e}")

    def check_files():
        """
        Scan data directory for new or modified files requiring processing.

        Compares current file modification times against control file records
        to identify files that need pipeline execution. Supports multiple data
        formats and maintains processing state persistence.
        """
        try:
            log.info("Scanning data directory for file changes")
            records = get_records()
            for file in os.listdir(data_dir):
                file_path = Path(data_dir) / file
                if not file_path.is_file():
                    continue
                if not file.lower().endswith(_ALLOWED_EXTS):
                    continue

                mtime = int(file_path.stat().st_mtime)
                if file not in records or mtime > records[file]:
                    log.info(f"Detected new or modified file: '{file}'")
                    run_pipeline_for_file(file)
                else:
                    log.debug(
                        f"File '{file}' already processed (mtime: {records[file]}) - skipping"
                    )
        except Exception as e:
            stop_all(f"Critical error during file system scanning: {e}")

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    def start_streamlit(pipeline_name, port=None):
        """
        Launch Streamlit dashboard for real-time pipeline monitoring.

        Starts the web-based dashboard interface with automatic port allocation
        and proper configuration for external access.

        Args:
            pipeline_name: Pipeline identifier for dashboard context
            port: Preferred port number (defaults to 8501 with fallback to 8510)
        """
        nonlocal streamlit_process
        dashboard_path = project_root / "web_interface" / "dashboard.py"

        if port is None:
            port = 8501
            if is_port_in_use(port):
                log.info(f"Port {port} unavailable, attempting fallback port 8510")
                port = 8510

        log.info(f"Launching Streamlit dashboard on port {port}")
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
        except Exception as e:
            stop_all(f"Critical failure launching Streamlit dashboard: {e}")

    log.info("Initializing APScheduler-based monitoring system")

    if early_start:
        check_files()

    # Configure APScheduler with robust job management and misfire handling
    scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
        timezone=_TZ_EUROPE_MADRID if _TZ_EUROPE_MADRID else None,
    )

    stype = schedule["type"]
    sparams = dict(schedule["params"])
    if stype == "interval":
        jitter = sparams.pop("jitter", 10)
        trigger = IntervalTrigger(jitter=jitter, **sparams)
        scheduler.add_job(
            check_files,
            trigger=trigger,
            id=f"check_files:{pipeline_name}",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )
    elif stype == "cron":
        trigger = CronTrigger(**sparams)
        scheduler.add_job(
            check_files,
            trigger=trigger,
            id=f"check_files:{pipeline_name}",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )
    elif stype == "date":
        trigger = DateTrigger(run_date=sparams["run_date"])
        scheduler.add_job(
            check_files,
            trigger=trigger,
            id=f"check_files:{pipeline_name}",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )
    else:
        stop_all(
            f"Unsupported schedule type '{stype}' - supported types: interval, cron, date"
        )

    scheduler.start()

    threading.Thread(
        target=start_streamlit, args=(pipeline_name, port), daemon=True
    ).start()

    try:
        while True:
            log.debug("Monitor system heartbeat - all components operational")
            time.sleep(30)
    except KeyboardInterrupt:
        stop_all("Monitor terminated by user interrupt signal")
    except Exception as e:
        stop_all(f"Unexpected monitor system failure: {e}")


# =========================
# Pipeline Management Utilities
# =========================


def list_pipelines(base_dir="pipelines"):
    """
    Display all available pipeline configurations.

    Scans the pipelines directory and lists all configured pipeline instances
    for management and monitoring purposes.

    Args:
        base_dir: Directory containing pipeline configurations
    """
    try:
        pipelines = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        if not pipelines:
            log.info("No pipeline configurations found")
        else:
            log.info("Available pipeline configurations:")
            for pipeline in pipelines:
                log.info(f"- {pipeline}")
    except Exception as e:
        log.error(f"Failed to scan pipeline directory: {e}")


def delete_pipeline(pipeline_name, base_dir="pipelines"):
    """
    Remove pipeline configuration and all associated files.

    Performs complete pipeline cleanup including configuration files,
    model artifacts, logs, and metrics. Requires user confirmation
    for irreversible deletion.

    Args:
        pipeline_name: Pipeline identifier to remove
        base_dir: Directory containing pipeline configurations
    """
    pipeline_path = os.path.join(base_dir, pipeline_name)
    if not os.path.exists(pipeline_path):
        log.error(f"Pipeline '{pipeline_name}' not found")
        return
    try:
        confirmation = input(
            f"Are you sure you want to delete the pipeline '{pipeline_name}'? This action is irreversible. (y/n): "
        )
        if confirmation.lower() != "y":
            log.info("Pipeline deletion cancelled by user")
            return
        shutil.rmtree(pipeline_path)
        log.info(f"Pipeline '{pipeline_name}' successfully deleted")
    except Exception as e:
        log.error(f"Pipeline deletion failed for '{pipeline_name}': {e}")
