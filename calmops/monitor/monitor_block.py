# monitor/monitor_block.py
# -*- coding: utf-8 -*-

"""
CalmOps Monitor (Watchdog, BLOQUES)
- Watches a directory (watchdog) and runs the pipeline in BLOCKS mode when new/modified files are detected.
- Launches the Streamlit blocks dashboard.
- Persistence modes:
    * "none"   : foreground execution
    * "pm2"    : generates a runner and starts with PM2 (auto-restart + on-boot)
    * "docker" : generates Dockerfile + docker-compose and runs in background
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
from typing import Optional, Dict, Tuple, List
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import the pipeline for BLOCKS (leave this import as you have it in your project)
from calmops.pipeline_block.pipeline_block import run_pipeline as run_pipeline_blocks
from calmops.utils import get_project_root, get_pipelines_root

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")

# =========================
# Logging (unified format)
# =========================

_LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"


def configure_root_logging(level: int = logging.INFO) -> None:
    """
    Configures the root logger with a single StreamHandler to stdout and
    the monitor's format. Clears previous handlers, captures warnings,
    and reduces verbosity of noisy libraries.
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
    for noisy in ("urllib3", "watchdog.observers.inotify_buffer", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _get_logger(name: str | None = None) -> logging.Logger:
    """
    Returns a logger that inherits the handler/format from the root.
    Does not add handlers (avoids duplicates).
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


# Default module logger; it is renamed by pipeline inside start_monitor_block()
log = _get_logger()


# =========================
# Helpers
# =========================

_ALLOWED_EXTS = (".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")


def _which(cmd: str):
    from shutil import which

    return which(cmd)


def _fatal(msg: str, code: int = 1):
    log.critical(msg)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
    """
    Serializes the sklearn-like model class + parameters to recreate it in the runner.
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


def _write_runner_config(pipeline_name: str, config_obj: dict, base_dir: Path) -> Path:
    """Write JSON with runner config."""
    cfg_dir = base_dir / "pipelines" / pipeline_name / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "runner_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return cfg_path


def _write_runner_script(
    pipeline_name: str, runner_cfg_path: Path, base_dir: Path
) -> Path:
    """
    Create a runner script that rebuilds the model and calls start_monitor_block
    with persistence='none' (avoids recursion in PM2/Docker).
    """
    pipeline_dir = base_dir / "pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    runner_path = pipeline_dir / f"run_{pipeline_name}_blocks_watchdog.py"

    content = f'''# Auto-generated runner (watchdog, blocks) for pipeline: {pipeline_name}
import os, json, importlib, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calmops.monitor.monitor_block import start_monitor_block

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

    start_monitor_block(
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
        window_size=cfg.get("window_size"),
        port=cfg.get("port"),
        persistence="none",
        block_col=cfg.get("block_col"),
        blocks_eval=cfg.get("blocks_eval"),
        split_within_blocks=cfg.get("split_within_blocks"),
                train_percentage=cfg.get("train_percentage"),
                fallback_strategy=cfg.get("fallback_strategy"),
                dir_predictions=cfg.get("dir_predictions"),
                target_files=cfg.get("target_files"),
                rest_preprocess_file=cfg.get("rest_preprocess_file"),
                skip_initial_preprocessing=cfg.get("skip_initial_preprocessing", False),
                skip_rest_preprocessing=cfg.get("skip_rest_preprocessing", False),
                target_col=cfg.get("target_col"),
            )
        '''
    with open(runner_path, "w") as f:
        f.write(content)
    return runner_path


# =========================
# PM2 helpers (list / delete)
# =========================


def pm2_list(prefix: str = "calmops-") -> List[dict]:
    if not _which("pm2"):
        log.error(
            "PM2 is not installed. Please install it using 'sudo npm install -g pm2'."
        )
        return []
    try:
        out = subprocess.check_output(["pm2", "jlist"], text=True)
        procs = json.loads(out)
    except Exception as e:
        log.error(f"Could not get PM2 list: {e}")
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
    app_name = f"calmops-{pipeline_name}-blocks"
    if not _which("pm2"):
        log.error("PM2 is not installed.")
        return
    try:
        subprocess.call(["pm2", "stop", app_name])
        subprocess.call(["pm2", "delete", app_name])
        subprocess.call(["pm2", "save"])
        log.info(f"PM2 process '{app_name}' stopped and deleted.")
    except Exception as e:
        log.warning(f"Could not delete in PM2: {e}")

    pipeline_path = os.path.join(base_dir, pipeline_name)
    try:
        if os.path.exists(pipeline_path):
            shutil.rmtree(pipeline_path)
            log.info(f"Pipeline folder '{pipeline_path}' deleted.")
        else:
            log.info(f"Folder '{pipeline_path}' does not exist.")
    except Exception as e:
        log.error(f"Could not delete the pipeline folder: {e}")


# =========================
# Docker helpers
# =========================


def _docker_install_hint() -> str:
    return (
        "Docker is required but not found.\n"
        "Install Docker Engine, e.g. on Debian/Ubuntu:\n"
        "  sudo apt-get update && sudo apt-get install -y docker.io\n"
        "  sudo systemctl enable --now docker\n"
        "  sudo usermod -aG docker $USER\n"
        "For docker compose v2 plugin:\n"
        "  sudo apt-get install -y docker-compose-plugin\n"
        "Alternatively (legacy v1):\n"
        "  sudo apt-get install -y docker-compose\n"
    )


def _write_docker_files(
    pipeline_name: str, runner_script_abs: str, base_dir: str, port: int | None
):
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)

    dockerfile_path = os.path.join(pipeline_dir, "Dockerfile")
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")

    runner_rel = runner_script_abs.replace(base_dir, "").lstrip(os.sep)
    runner_rel_posix = runner_rel.replace(os.sep, "/")
    runner_in_container = f"/app/{runner_rel_posix}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile (watchdog, blocks) for {pipeline_name}
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    (test -f requirements.txt && pip install --no-cache-dir -r requirements.txt || true)

EXPOSE {exposed_port}

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "{runner_in_container}"]
"""

    compose = f"""# Auto-generated docker-compose (watchdog, blocks) for {pipeline_name}
services:
  {pipeline_name}_blocks_watchdog:
    build:
      context: ../../
      dockerfile: ./pipelines/{pipeline_name}/Dockerfile
    container_name: calmops_{pipeline_name}_blocks_watchdog
    restart: unless-stopped
    ports:
      - \"{exposed_port}:{exposed_port}\" 
    volumes:
      - \"../../:/app"
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
    pipeline_name: str, runner_script: str, base_dir: str, port: int | None
):
    if not _which("docker"):
        _fatal(_docker_install_hint())

    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    try:
        # Compose v2
        if (
            _which("docker")
            and subprocess.call(
                ["docker", "compose", "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        ):
            # best-effort down first
            subprocess.call(
                ["docker", "compose", "down", "--volumes", "--remove-orphans"],
                cwd=pipeline_dir,
            )
            subprocess.check_call(
                ["docker", "compose", "up", "-d", "--build"], cwd=pipeline_dir
            )
        elif _which("docker-compose"):
            subprocess.call(
                ["docker-compose", "down", "--volumes", "--remove-orphans"],
                cwd=pipeline_dir,
            )
            subprocess.check_call(
                ["docker-compose", "up", "-d", "--build"], cwd=pipeline_dir
            )
        else:
            _fatal(
                "Neither `docker compose` (v2) nor `docker-compose` (v1) is available.\n"
                f"{_docker_install_hint()}"
            )
    except Exception as e:
        _fatal(f"Failed to build/run Docker services: {e}\n{_docker_install_hint()}")


def docker_list(prefix: str = "calmops_") -> List[Tuple[str, str]]:
    if not _which("docker"):
        log.error("Docker not installed.")
        return []
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"], text=True
        ).strip()
    except Exception as e:
        log.error(f"docker ps failed: {e}")
        return []
    rows: List[Tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, status = line.split("\t", 1)
        if name.startswith(prefix):
            rows.append((name, status))
    if rows:
        log.info("Docker containers:")
        for name, status in rows:
            log.info(f"- {name}: {status}")
    else:
        log.info(f"No containers with prefix '{prefix}'.")
    return rows


def docker_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    pipeline_dir = os.path.join(base_dir, pipeline_name)
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")
    if not _which("docker"):
        log.error("Docker not installed.")
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
            cname = f"calmops_{pipeline_name}_blocks_watchdog"
            subprocess.call(["docker", "rm", "-f", cname])
        log.info(f"Docker pipeline '{pipeline_name}' (blocks watchdog) deleted.")
    except Exception as e:
        log.warning(f"Could not bring down docker compose: {e}")

    try:
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)
            log.info(f"Folder '{pipeline_dir}' deleted.")
    except Exception as e:
        log.error(f"Could not delete the folder: {e}")


# =========================
# Main monitor (Watchdog, BLOQUES)
# =========================


def start_monitor_block(
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
    window_size: Optional[int] = None,
    port: Optional[int] = None,
    dir_predictions: Optional[str] = None,
    persistence: str = "none",  # "none" | "pm2" | "docker"
    # ---
    block_col: Optional[str] = None,
    blocks_eval: Optional[
        List[str]
    ] = None,  # list of blocks to evaluate; passed as eval_blocks to the pipeline
    split_within_blocks: Optional[
        bool
    ] = False,  # New parameter for splitting within blocks
    train_percentage: Optional[float] = 0.8,  # New parameter for train percentage
    fallback_strategy: str = "global",
    # Flexible Preprocessing
    target_files: Optional[list] = None,
    rest_preprocess_file: Optional[str] = None,
    skip_initial_preprocessing: bool = False,
    skip_rest_preprocessing: bool = False,
    target_col: Optional[str] = None,
):
    """
    BLOCK monitoring system (Watchdog):
      - Watches `data_dir` and executes run_pipeline_blocks() on new/modified files.
      - Launches the blocks dashboard.
      - Can start persistently with PM2 or Docker.
    """
    # Global logging with monitor format
    configure_root_logging(logging.INFO)

    # Logger specific to the pipeline
    global log
    log = _get_logger(f"calmops.monitor.blocks.{pipeline_name}")

    persistence = (persistence or "none").lower()
    pipelines_root = get_pipelines_root()
    project_root = get_project_root()

    # Always write the runner config, regardless of persistence mode.
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
        "window_size": window_size,
        "port": port,
        "model_spec": model_spec,
        "block_col": block_col,
        "blocks_eval": blocks_eval,
        "monitor_type": "monitor_block",
        "split_within_blocks": split_within_blocks,  # Add to config
        "train_percentage": train_percentage,  # Add to config
        "fallback_strategy": fallback_strategy,
        "dir_predictions": dir_predictions,
        "target_files": target_files,
        "rest_preprocess_file": rest_preprocess_file,
        "skip_initial_preprocessing": skip_initial_preprocessing,
        "skip_rest_preprocessing": skip_rest_preprocessing,
        "target_col": target_col,
    }
    runner_cfg_path = _write_runner_config(
        pipeline_name, runner_cfg_obj, pipelines_root
    )

    # --- persistence bootstrap (early exit if enabled) ---
    if persistence in ("pm2", "docker"):
        runner_script = _write_runner_script(
            pipeline_name, runner_cfg_path, pipelines_root
        )

        if persistence == "pm2":
            # PM2
            if not _which("pm2"):
                _fatal(
                    "PM2 is required but not found.\n"
                    "sudo npm install -g pm2\n"
                    "Then re-run with persistence='pm2'."
                )
            eco_path = (
                pipelines_root
                / "pipelines"
                / pipeline_name
                / "ecosystem.blocks.watchdog.config.js"
            )
            app_name = f"calmops-{pipeline_name}-blocks"
            python_exec = sys.executable

            runner_script_posix = runner_script.as_posix()
            python_exec_posix = Path(python_exec).as_posix()
            base_dir_posix = pipelines_root.as_posix()

            ecosystem_lines = [
                "module.exports = {",
                "  apps: [{",
                f'    name: "{app_name}",',
                f'    script: "{runner_script_posix}",',
                f'    interpreter: "{python_exec_posix}",',
                '    interpreter_args: "-u",',
                f'    cwd: "{base_dir_posix}",',
                "    autorestart: true,",
                "    watch: false,",
                "    max_restarts: 10",
                "  }]",
                "};",
            ]

            with open(eco_path, "w") as f:
                f.write("\n".join(ecosystem_lines))

            try:
                subprocess.check_call(["pm2", "start", str(eco_path)])
                subprocess.call(["pm2", "save"])
                user = os.getenv("USER", "") or ""
                if user:
                    subprocess.call(["pm2", "startup", "-u", user])
                else:
                    subprocess.call(["pm2", "startup"])
            except Exception as e:
                _fatal(f"Failed to start with PM2: {e}")
            log.info("PM2 persistence enabled. Exiting foreground process.")
            return
        else:
            _launch_with_docker(
                pipeline_name, str(runner_script), str(pipelines_root), port or 8501
            )
            log.info(
                "Docker persistence enabled via docker-compose. Exiting foreground process."
            )
            return

    # ---------- regular (non-persistent) flow ----------
    base_pipeline_dir = pipelines_root / "pipelines" / pipeline_name
    output_dir = base_pipeline_dir / "outputs"
    control_dir = base_pipeline_dir / "control"
    logs_dir = base_pipeline_dir / "logs"
    metrics_dir = base_pipeline_dir / "metrics"
    config_dir = base_pipeline_dir / "config"

    for d in [output_dir, control_dir, logs_dir, metrics_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    control_file = control_dir / "control_file.txt"
    if not control_file.exists():
        control_file.touch()

    # Config for dashboard (blocks)
    config_path = config_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "pipeline_name": pipeline_name,
                "data_dir": data_dir,
                "preprocess_file": preprocess_file,
                "block_col": block_col,
                "mode": "blocks",
            },
            f,
        )

    streamlit_process = None

    def stop_all(error_msg=None):
        nonlocal streamlit_process
        log.critical("Stopping all processes...")
        if streamlit_process and streamlit_process.poll() is None:
            log.info("Terminating Streamlit...")
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=10)
            except Exception:
                streamlit_process.kill()
        if error_msg:
            log.error(error_msg)
        sys.exit(1)

    def get_records() -> Dict[str, int]:
        """Read control_file with processed files + mtimes. <filename>,<mtime>"""
        records: Dict[str, int] = {}
        if control_file.exists():
            with open(control_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        try:
                            records[parts[0]] = int(float(parts[1]))
                        except Exception:
                            continue
        return records

    def run_pipeline_for_file(file: str):
        file_path = Path(data_dir) / file
        if not file_path.is_file():
            log.warning(f"File not found (skipping): {file}")
            return
        mtime = int(file_path.stat().st_mtime)
        log.info(f"Running blocks pipeline for {file} (mtime={mtime})...")
        try:
            # If splitting within blocks, the evaluator should consider all blocks present in the eval set.
            # If not, it should only consider the specific `blocks_eval` list.
            current_eval_blocks = None if split_within_blocks else blocks_eval

            run_pipeline_blocks(
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
                target_file=file,  # basename (aligned with control_file)
                window_size=window_size,
                block_col=block_col,
                eval_blocks=current_eval_blocks,  # passes block evaluation to the pipeline
                split_within_blocks=split_within_blocks,  # Pass new parameter
                train_percentage=train_percentage,  # Pass new parameter
                dir_predictions=dir_predictions,
                target_files=target_files,
                rest_preprocess_file=rest_preprocess_file,
                skip_initial_preprocessing=skip_initial_preprocessing,
                skip_rest_preprocessing=skip_rest_preprocessing,
                target_col=target_col,
            )
            log.info(f"Blocks pipeline completed for {file}")
        except Exception as e:
            stop_all(f"The block pipeline failed for {file}: {e}")

    def execute_if_needed(file: str):
        """Trigger processing only if the file is new or modified with respect to the control."""
        file_path = Path(data_dir) / file
        if not file_path.is_file():
            return
        if not file.lower().endswith(_ALLOWED_EXTS):
            log.info(f"Ignored file: {file}")
            return
        try:
            mtime = int(file_path.stat().st_mtime)
        except Exception as e:
            log.warning(f"Could not stat mtime for {file}: {e}")
            return
        records = get_records()
        if file not in records or mtime > records[file]:
            log.info(f"New/updated file found: {file}")
            run_pipeline_for_file(file)
        else:
            log.info(
                f"{file} already processed (stored mtime={records[file]}), skipping."
            )

    class DataFileHandler(FileSystemEventHandler):
        """Watchdog handler for creates/updates of supported files."""

        def __init__(self, logger: logging.Logger):
            super().__init__()
            self.log = logger

        def on_created(self, event):
            if event.is_directory:
                return
            fname = os.path.basename(event.src_path)
            self.log.info(f"File created: {fname} -> check & run")
            execute_if_needed(fname)

        def on_modified(self, event):
            if event.is_directory:
                return
            fname = os.path.basename(event.src_path)
            self.log.info(f"File modified: {fname} -> check & run")
            execute_if_needed(fname)

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    def start_streamlit(pipeline_name: str, port: Optional[int] = None):
        nonlocal streamlit_process
        dashboard_path = project_root / "web_interface" / "dashboard_block.py"

        if port is None:
            port = 8501
            if is_port_in_use(port):
                log.info(f"Port {port} is occupied. Trying 8510...")
                port = 8510

        log.info(f"Launching blocks dashboard on port {port}...")
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
            stop_all(f"Could not launch Streamlit (blocks): {e}")

    def start_watchdog():
        """Start Watchdog and monitor the health of the dashboard."""
        log.info(f"Watching directory: {data_dir} ...")
        event_handler = DataFileHandler(log)
        observer = Observer()
        observer.schedule(event_handler, path=data_dir, recursive=False)
        observer.start()
        try:
            while True:
                if streamlit_process and streamlit_process.poll() is not None:
                    stop_all("Streamlit dashboard unexpectedly closed.")
                time.sleep(10)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    log.info("Launching blocks monitor with Watchdog + Streamlit...")
    log.info("Verifying files in the directory at startup...")

    # Initial scan: process only if new/modified vs control_file.txt
    for file in os.listdir(data_dir):
        execute_if_needed(file)

    threading.Thread(
        target=start_streamlit, args=(pipeline_name, port), daemon=True
    ).start()
    start_watchdog()


# =========================
# Utils (list / delete)
# =========================


def list_pipelines(base_dir="pipelines"):
    try:
        pipelines = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        if not pipelines:
            log.info("No pipelines found.")
        else:
            log.info("Available Pipelines:")
            for pipeline in pipelines:
                log.info(f"- {pipeline}")
    except Exception as e:
        log.error(f"Could not list pipelines: {e}")


def delete_pipeline(pipeline_name, base_dir="pipelines"):
    pipeline_path = os.path.join(base_dir, pipeline_name)
    if not os.path.exists(pipeline_path):
        log.error(f"Pipeline {pipeline_name} does not exist.")
        return
    try:
        confirmation = input(
            f"Are you sure you want to delete the pipeline '{pipeline_name}'? This action is irreversible. (y/n): "
        )
        if confirmation.lower() != "y":
            log.info("Deletion cancelled.")
            return
        shutil.rmtree(pipeline_path)
        log.info(f"Pipeline '{pipeline_name}' has been deleted.")
    except Exception as e:
        log.error(f"Failed to delete pipeline {pipeline_name}: {e}")
