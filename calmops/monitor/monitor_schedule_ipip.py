# monitor/monitor_schedule_ipip.py  (CON schedule, basado en APScheduler)
# -*- coding: utf-8 -*-
"""
CalmOps Monitor (IPIP) — with schedule
- Schedules periodic executions of the IPIP pipeline using APScheduler.
- Launches the Streamlit dashboard.
- Optional persistence modes (pm2/docker) with auto-generated runner.

`schedule` parameter:
  - {"type": "interval", "params": {"minutes": 2}}
  - {"type": "cron", "params": {"hour": 3, "minute": 0}}  # every day at 03:00
"""

from __future__ import annotations

import os
import sys
import time
import json
import socket
import shutil
import threading
import subprocess

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from calmops.utils import get_project_root, get_pipelines_root

# IPIP pipeline
from calmops.IPIP.pipeline_ipip import run_pipeline

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel("ERROR")


# =========================
# Logging
# =========================
_LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"


def configure_root_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(sh)
    logging.captureWarnings(True)
    for noisy in ("watchdog", "urllib3", "apscheduler", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


log = _get_logger()


# =========================
# Helpers
# =========================
def _which(cmd: str):
    from shutil import which

    return which(cmd)


def _fatal(msg: str, code: int = 1):
    log.critical(msg)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
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
    cfg_dir = base_dir / "pipelines" / pipeline_name / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "runner_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return str(cfg_path)


def _write_runner_script(
    pipeline_name: str, runner_cfg_path: str, base_dir: Path
) -> str:
    pipeline_dir = base_dir / "pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    runner_path = pipeline_dir / f"run_{pipeline_name}_ipip_schedule.py"

    content = f'''# Auto-generated scheduled runner (IPIP) for pipeline: {pipeline_name}
import os, sys, json, importlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calmops.monitor.monitor_schedule_ipip import start_monitor_schedule_ipip

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

    start_monitor_schedule_ipip(
        pipeline_name=cfg["pipeline_name"],
        data_dir=cfg["data_dir"],
        preprocess_file=cfg["preprocess_file"],
        model_instance=model_instance,
        retrain_mode=cfg["retrain_mode"],
        random_state=cfg["random_state"],
        custom_train_file=cfg.get("custom_train_file"),
        custom_retrain_file=cfg.get("custom_retrain_file"),
        delimiter=cfg.get("delimiter"),
        target_file=cfg.get("target_file"),
        window_size=cfg.get("window_size"),
        schedule=cfg.get("schedule"),
        early_start=cfg.get("early_start", True),
        port=cfg.get("port"),
        persistence="none",
        block_col=cfg.get("block_col"),
        ipip_config=cfg.get("ipip_config"),
        ipip_config=cfg.get("ipip_config"),
        prediction_only=cfg.get("prediction_only", False),
        target_files=cfg.get("target_files"),
        rest_preprocess_file=cfg.get("rest_preprocess_file"),
        skip_initial_preprocessing=cfg.get("skip_initial_preprocessing", False),
        skip_rest_preprocessing=cfg.get("skip_rest_preprocessing", False),
        target_col=cfg.get("target_col"),
    )
'''
    with open(runner_path, "w") as f:
        f.write(content)
    return str(runner_path)


def _pm2_install_hint() -> str:
    return (
        "PM2 is required but not found.\n"
        "Install Node.js + PM2:\n"
        "  sudo apt-get update && sudo apt-get install -y nodejs npm\n"
        "  sudo npm install -g pm2\n"
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
    pipeline_name: str, runner_script_abs: str, base_dir: Path, port: int | None
):
    pipeline_dir = base_dir / "pipelines" / pipeline_name
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_path = pipeline_dir / "Dockerfile"
    compose_path = pipeline_dir / "docker-compose.yml"

    runner_rel = Path(runner_script_abs).relative_to(base_dir)
    runner_in_container = f"/app/{runner_rel.as_posix()}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile (IPIP scheduled) for {pipeline_name}
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
    compose = f"""# Auto-generated docker-compose (IPIP scheduled) for {pipeline_name}
version: "3.8"
services:
  {pipeline_name}_ipip_schedule:
    build:
      context: ../../
      dockerfile: ./pipelines/{pipeline_name}/Dockerfile
    container_name: calmops_{pipeline_name}_ipip_schedule
    restart: unless-stopped
    ports:
      - "{exposed_port}:{exposed_port}"
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


def _launch_with_pm2(pipeline_name: str, runner_script: str, base_dir: Path):
    if not _which("pm2"):
        _fatal(_pm2_install_hint())

    eco_path = base_dir / "pipelines" / pipeline_name / "ecosystem.ipip.schedule.js"
    app_name = f"calmops-{pipeline_name}-ipip-schedule"

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
        _fatal(f"Failed to start with PM2: {e}\n{_pm2_install_hint()}")

    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        pass
    try:
        subprocess.check_call(["pm2", "startup", "-u", os.getenv("USER", "") or ""])
    except Exception:
        pass

    log.info(f"[PM2] App '{app_name}' started and saved.")


def _launch_with_docker(
    pipeline_name: str, runner_script: str, base_dir: Path, port: int | None
):
    if not _which("docker"):
        _fatal(
            "Docker is required but not found.\n"
            "Install Docker Engine and the docker compose plugin."
        )
    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    pipeline_dir = base_dir / "pipelines" / pipeline_name
    try:
        avail = _compose_available()
        if avail == "v2":
            subprocess.check_call(
                ["docker", "compose", "up", "-d", "--build"], cwd=str(pipeline_dir)
            )
        elif avail == "v1":
            subprocess.check_call(
                ["docker-compose", "up", "-d", "--build"], cwd=str(pipeline_dir)
            )
        else:
            _fatal(
                "Neither `docker compose` (v2) nor `docker-compose` (v1) is available."
            )
    except Exception as e:
        _fatal(f"Failed to launch Docker services: {e}")


# =========================
# Monitor with schedule
# =========================
def start_monitor_schedule_ipip(
    *,
    pipeline_name: str,
    data_dir: str,
    preprocess_file: str,
    model_instance,
    retrain_mode: int,
    random_state: int,
    custom_train_file: str = None,
    custom_retrain_file: str = None,
    delimiter: str = ",",
    target_file: str = None,
    window_size: int = None,
    schedule: Dict[str, Any] = None,  # {"type": "interval"|"cron", "params": {...}}
    early_start: bool = True,
    port: int | None = None,
    persistence: str = "none",  # "none" | "pm2" | "docker"
    block_col: str | None = None,
    ipip_config: dict | None = None,
    prediction_only: bool = False,
    dir_predictions: Optional[str] = None,
    # Flexible Preprocessing
    target_files: Optional[list] = None,
    rest_preprocess_file: Optional[str] = None,
    skip_initial_preprocessing: bool = False,
    skip_rest_preprocessing: bool = False,
    target_col: Optional[str] = None,
):
    """
    Programa ejecuciones periódicas del pipeline. Si `target_file` es None,
    `data_loader` decidirá si hay nuevos/actualizados (usando control_file).
    """
    configure_root_logging(logging.INFO)
    global log
    log = _get_logger(f"calmops.monitor.ipip.schedule.{pipeline_name}")

    persistence = (persistence or "none").lower()
    pipelines_root = get_pipelines_root()
    project_root = get_project_root()

    # Always write the runner config, regardless of persistence mode.
    model_spec = _model_spec_from_instance(model_instance)
    runner_cfg_obj = {
        "pipeline_name": pipeline_name,
        "data_dir": data_dir,
        "preprocess_file": preprocess_file,
        "retrain_mode": retrain_mode,
        "random_state": random_state,
        "custom_train_file": custom_train_file,
        "custom_retrain_file": custom_retrain_file,
        "delimiter": delimiter,
        "target_file": target_file,
        "window_size": window_size,
        "schedule": schedule or {"type": "interval", "params": {"minutes": 5}},
        "early_start": bool(early_start),
        "port": port,
        "block_col": block_col,
        "ipip_config": ipip_config,
        "model_spec": model_spec,
        "monitor_type": "monitor_schedule_ipip",
        "prediction_only": prediction_only,
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

    # --- persistence (early exit) ---
    if persistence in ("pm2", "docker"):
        runner_script = _write_runner_script(
            pipeline_name, runner_cfg_path, pipelines_root
        )

        if persistence == "pm2":
            _launch_with_pm2(pipeline_name, runner_script, pipelines_root)
            log.info("[PM2] Persistence enabled. Exiting foreground process.")
            return
        else:
            _launch_with_docker(
                pipeline_name, runner_script, pipelines_root, port or 8501
            )
            log.info(
                "[DOCKER] Persistence enabled via docker-compose. Exiting foreground process."
            )
            return

    # ---------- setup dirs & dashboard config ----------
    base_pipeline_dir = pipelines_root / "pipelines" / pipeline_name
    output_dir = base_pipeline_dir / "modelos"
    control_dir = base_pipeline_dir / "control"
    logs_dir = base_pipeline_dir / "logs"
    metrics_dir = base_pipeline_dir / "metrics"
    config_dir = base_pipeline_dir / "config"

    for d in [output_dir, control_dir, logs_dir, metrics_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    control_file = control_dir / "control_file.txt"
    if not control_file.exists():
        control_file.touch()

    # config para dashboard
    config_path = config_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "pipeline_name": pipeline_name,
                "data_dir": data_dir,
                "preprocess_file": preprocess_file,
                "block_col": block_col,
                "mode": "ipip",
            },
            f,
        )

    # ---------- scheduler ----------
    if not schedule or "type" not in schedule or "params" not in schedule:
        schedule = {"type": "interval", "params": {"minutes": 5}}
    sched_type = str(schedule["type"]).lower()
    sched_params = schedule["params"] or {}

    scheduler = BackgroundScheduler(timezone="UTC")

    def run_once():
        try:
            log.info("[SCHEDULE] Running IPIP pipeline job...")
            run_pipeline(
                pipeline_name=pipeline_name,
                data_dir=data_dir,
                preprocess_file=preprocess_file,
                model_instance=model_instance,
                retrain_mode=retrain_mode,
                random_state=random_state,
                custom_train_file=custom_train_file,
                custom_retrain_file=custom_retrain_file,
                delimiter=delimiter,
                target_file=target_file,  # if None, data_loader decides if there is anything new
                window_size=window_size,
                block_col=block_col,
                ipip_config=ipip_config,
                prediction_only=prediction_only,
                dir_predictions=dir_predictions,
                target_files=target_files,
                rest_preprocess_file=rest_preprocess_file,
                skip_initial_preprocessing=skip_initial_preprocessing,
                skip_rest_preprocessing=skip_rest_preprocessing,
                target_col=target_col,
            )
            log.info("[SCHEDULE] Pipeline finished.")
        except Exception as e:
            log.exception(f"[SCHEDULE] Pipeline error: {e}")

    if sched_type == "interval":
        trigger = IntervalTrigger(**sched_params)
    elif sched_type == "cron":
        trigger = CronTrigger(**sched_params)
    else:
        log.warning(
            f"Unknown schedule type '{{sched_type}}', defaulting to interval(5 min)."
        )
        trigger = IntervalTrigger(minutes=5)

    scheduler.add_job(
        run_once, trigger=trigger, id=f"ipip_{pipeline_name}_job", replace_existing=True
    )
    scheduler.start()
    log.info(
        f"[SCHEDULE] Scheduler started with type='{sched_type}', params={sched_params}"
    )

    # early start
    if early_start:
        log.info("[SCHEDULE] Early start enabled → running first job now.")
        run_once()

    # ---------- streamlit ----------
    streamlit_process = None

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    def start_streamlit(pipeline_name: str, port: int | None = None):
        nonlocal streamlit_process
        dashboard_path = project_root / "web_interface" / "dashboard_ipip.py"

        if port is None:
            port = 8501
            if is_port_in_use(port):
                log.info(f"[STREAMLIT] Port {port} occupied. Trying port 8510...")
                port = 8510

        log.info(f"[STREAMLIT] Launching IPIP dashboard on port {port}...")
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
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.error(f"Failed to start Streamlit: {e}")

    threading.Thread(
        target=start_streamlit, args=(pipeline_name, port), daemon=True
    ).start()

    # ---------- keep alive ----------
    try:
        while True:
            if streamlit_process and streamlit_process.poll() is not None:
                log.error("Streamlit dashboard unexpectedly closed. Relaunching...")
                threading.Thread(
                    target=start_streamlit, args=(pipeline_name, port), daemon=True
                ).start()
            time.sleep(10)
    except KeyboardInterrupt:
        log.info("Stopping scheduler...")
        scheduler.shutdown(wait=False)
        if streamlit_process and streamlit_process.poll() is None:
            try:
                streamlit_process.terminate()
            except Exception:
                pass


# Direct use example
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    def main():
        """Main function to start the monitor with predefined arguments."""
        configure_root_logging(logging.INFO)

        # Define parameters directly for start_monitor
        pipeline_name = "my_pipeline_ipip_schedule"
        data_dir = "/path/to/data"
        preprocess_file = "/path/to/preprocessing_ipip.py"
        model_instance = RandomForestClassifier(random_state=42)
        retrain_mode = 6
        random_state = 42
        delimiter = ","
        target_file = None
        window_size = None
        schedule = {"type": "interval", "params": {"minutes": 2}}
        early_start = True
        port = 8600
        persistence = "none"
        block_col = "chunk"
        ipip_config = {
            "p": 20,
            "b": 5,
            "prop_majoritaria": 0.55,
            "val_size": 0.20,
            "prop_minor_frac": 0.75,
        }
        prediction_only = False

        start_monitor_schedule_ipip(
            pipeline_name=pipeline_name,
            data_dir=data_dir,
            preprocess_file=preprocess_file,
            model_instance=model_instance,
            retrain_mode=retrain_mode,
            random_state=random_state,
            delimiter=delimiter,
            target_file=target_file,
            window_size=window_size,
            schedule=schedule,
            early_start=early_start,
            port=port,
            persistence=persistence,
            block_col=block_col,
            ipip_config=ipip_config,
            prediction_only=prediction_only,
        )

    main()
