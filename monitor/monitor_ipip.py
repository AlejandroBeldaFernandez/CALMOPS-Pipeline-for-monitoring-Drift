# monitor/monitor_ipip.py  (SIN schedule, basado en Watchdog)
# -*- coding: utf-8 -*-
"""
CalmOps Monitor (IPIP) — without schedule
- Watches a data directory using Watchdog and runs the IPIP pipeline upon file creation/modification.
- Launches the Streamlit dashboard.
- Optional persistence modes:
    * "none"   : run in the foreground (default)
    * "pm2"    : generate a runner and start with PM2 (auto-restart + on boot)
    * "docker" : generate Dockerfile + docker-compose and run in the background
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
import importlib
import logging
from typing import Dict, Tuple

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# IPIP pipeline
from IPIP.pipeline_ipip import run_pipeline


# =========================
# Logging 
# =========================
_LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"

def configure_root_logging(level: int = logging.INFO) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
_ALLOWED_EXTS = (".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

def _write_runner_config(pipeline_name: str, config_obj: dict, base_dir: str) -> str:
    cfg_dir = os.path.join(base_dir, "pipelines", pipeline_name, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "runner_ipip_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return cfg_path

def _write_runner_script(pipeline_name: str, runner_cfg_path: str, base_dir: str) -> str:
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)
    runner_path = os.path.join(pipeline_dir, f"run_{pipeline_name}_ipip.py")

    content = f'''# Auto-generated runner (IPIP) for pipeline: {pipeline_name}
import os, sys, json, importlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from monitor.monitor_ipip import start_monitor_ipip

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

    start_monitor_ipip(
        pipeline_name=cfg["pipeline_name"],
        data_dir=cfg["data_dir"],
        preprocess_file=cfg["preprocess_file"],
        thresholds_drift=cfg["thresholds_drift"],
        thresholds_perf=cfg["thresholds_perf"],
        model_instance=model_instance,
        retrain_mode=cfg["retrain_mode"],
        random_state=cfg["random_state"],
        custom_train_file=cfg.get("custom_train_file"),
        custom_retrain_file=cfg.get("custom_retrain_file"),
        delimiter=cfg.get("delimiter"),
        target_file=cfg.get("target_file"),
        window_size=cfg.get("window_size"),
        block_col=cfg.get("block_col"),
        ipip_config=cfg.get("ipip_config"),
        port=cfg.get("port"),
        persistence="none"
    )
'''
    with open(runner_path, "w") as f:
        f.write(content)
    return runner_path

# =========================
# PM2 / Docker helpers
# =========================
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
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return "v2"
    except Exception:
        return "v1" if shutil.which("docker-compose") else None

def _write_docker_files(pipeline_name: str, runner_script_abs: str, base_dir: str, port: int | None):
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)

    dockerfile_path = os.path.join(pipeline_dir, "Dockerfile")
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")

    runner_rel = runner_script_abs.replace(base_dir, "").lstrip(os.sep)
    runner_rel_posix = runner_rel.replace(os.sep, "/")
    runner_in_container = f"/app/{runner_rel_posix}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile (IPIP) for {pipeline_name}
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \\
    (test -f requirements.txt && pip install --no-cache-dir -r requirements.txt || true)

EXPOSE {exposed_port}

ENV PYTHONUNBUFFERED=1 \\
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \\
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "{runner_in_container}"]
"""

    compose = f"""# Auto-generated docker-compose (IPIP) for {pipeline_name}
version: "3.8"
services:
  {pipeline_name}_ipip:
    build:
      context: ../../
      dockerfile: ./pipelines/{pipeline_name}/Dockerfile
    container_name: calmops_{pipeline_name}_ipip
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

    return dockerfile_path, compose_path

def _launch_with_pm2(
    pipeline_name: str,
    runner_script: str,
    base_dir: str,
    logger: logging.Logger | None = None
):
    """
    Starts a pipeline with PM2.
    Works on Linux/macOS. On Windows, PM2 runs but cannot auto-start on boot.
    """
    import os, sys, subprocess, logging

    if logger is None:
        logger = logging.getLogger("calmops.monitor")

    # Verificar que PM2 esté instalado
    if not _which("pm2"):
        _fatal(_pm2_install_hint())

    # Rutas y nombres
    eco_path = os.path.join(base_dir, "pipelines", pipeline_name, "ecosystem.ipip.config.js")
    app_name = f"calmops-{pipeline_name}-ipip"

    python_exec = sys.executable
    runner_script_posix = runner_script.replace("\\", "/")
    python_exec_posix = python_exec.replace("\\", "/")
    base_dir_posix = base_dir.replace("\\", "/")

    # Contenido del ecosystem config
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
    # Guardar el archivo de configuración
    with open(eco_path, "w") as f:
        f.write(ecosystem)

    # Iniciar la app con PM2
    try:
        subprocess.check_call(["pm2", "start", eco_path])
    except Exception as e:
        _fatal(f"[PM2] Failed to start: {e}\n{_pm2_install_hint()}")

    # Guardar el estado de PM2
    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        logger.warning("[PM2] Could not save PM2 process list.")

    # Intentar configurar auto-start en Linux/macOS
    if os.name != "nt":
        try:
            user = os.getenv("USER") or os.getenv("USERNAME") or ""
            if user:
                subprocess.check_call(["pm2", "startup", "-u", user])
            else:
                subprocess.check_call(["pm2", "startup"])
        except Exception:
            logger.warning("[PM2] Could not create startup script. Configure manually if needed.")
    else:
        logger.info("[PM2] Windows detected: automatic startup on boot is not configured.")

    logger.info(f"[PM2] App '{app_name}' started and saved. Autorestart enabled.")


def _launch_with_docker(pipeline_name: str, runner_script: str, base_dir: str, port: int | None):
    if not _which("docker"):
        _fatal(
            "Docker is required but not found.\n"
            "Install Docker Engine and the docker compose plugin."
        )
    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    try:
        avail = _compose_available()
        if avail == "v2":
            subprocess.check_call(["docker", "compose", "up", "-d", "--build"], cwd=pipeline_dir)
        elif avail == "v1":
            subprocess.check_call(["docker-compose", "up", "-d", "--build"], cwd=pipeline_dir)
        else:
            _fatal("Neither `docker compose` (v2) nor `docker-compose` (v1) is available.")
    except Exception as e:
        _fatal(f"Failed to launch Docker services: {e}")

# =========================
# Main monitor entrypoint
# =========================
def start_monitor_ipip(
    *,
    pipeline_name: str,
    data_dir: str,
    preprocess_file: str,
    thresholds_drift: dict,
    thresholds_perf: dict,
    model_instance,
    retrain_mode: int,
    random_state: int,
    custom_train_file: str = None,
    custom_retrain_file: str = None,
    delimiter: str = ",",
    target_file: str = None,
    window_size: int = None,
    port: int | None = None,
    persistence: str = "none",   # "none" | "pm2" | "docker"
    block_col: str | None = None,
    ipip_config: dict | None = None,
):
    configure_root_logging(logging.INFO)
    global log
    log = _get_logger(f"calmops.monitor.ipip.{pipeline_name}")

    # --- persistence (early exit) ---
    persistence = (persistence or "none").lower()
    base_dir = _project_root()

    if persistence in ("pm2", "docker"):
        model_spec = _model_spec_from_instance(model_instance)
        runner_cfg_obj = {
            "pipeline_name": pipeline_name,
            "data_dir": data_dir,
            "preprocess_file": preprocess_file,
            "thresholds_drift": thresholds_drift,
            "thresholds_perf": thresholds_perf,
            "retrain_mode": retrain_mode,
            "random_state": random_state,
            "custom_train_file": custom_train_file,
            "custom_retrain_file": custom_retrain_file,
            "delimiter": delimiter,
            "target_file": target_file,
            "window_size": window_size,
            "port": port,
            "block_col": block_col,
            "ipip_config": ipip_config,
            "model_spec": model_spec,
        }
        runner_cfg_path = _write_runner_config(pipeline_name, runner_cfg_obj, base_dir)
        runner_script = _write_runner_script(pipeline_name, runner_cfg_path, base_dir)

        if persistence == "pm2":
            _launch_with_pm2(pipeline_name, runner_script, base_dir)
            log.info("[PM2] Persistence enabled. Exiting foreground process.")
            return
        else:
            _launch_with_docker(pipeline_name, runner_script, base_dir, port or 8501)
            log.info("[DOCKER] Persistence enabled via docker-compose. Exiting foreground process.")
            return

    # ---------- regular flow ----------
    BASE_PIPELINE_DIR = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    OUTPUT_DIR  = os.path.join(BASE_PIPELINE_DIR, "modelos")
    CONTROL_DIR = os.path.join(BASE_PIPELINE_DIR, "control")
    LOGS_DIR    = os.path.join(BASE_PIPELINE_DIR, "logs")
    METRICS_DIR = os.path.join(BASE_PIPELINE_DIR, "metrics")
    CONFIG_DIR  = os.path.join(BASE_PIPELINE_DIR, "config")

    for d in [OUTPUT_DIR, CONTROL_DIR, LOGS_DIR, METRICS_DIR, CONFIG_DIR]:
        os.makedirs(d, exist_ok=True)

    control_file = os.path.join(CONTROL_DIR, "control_file.txt")
    if not os.path.exists(control_file):
        open(control_file, "w").close()

    # Dashboard config
    config_path = os.path.join(CONFIG_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "pipeline_name": pipeline_name,
            "data_dir": data_dir,
            "preprocess_file": preprocess_file,
            "block_col": block_col,
            "mode": "ipip"
        }, f)

    streamlit_process = None

    def stop_all(error_msg=None):
        nonlocal streamlit_process
        log.critical("Stopping all processes...")
        if streamlit_process and streamlit_process.poll() is None:
            log.info("[STREAMLIT] Terminating Streamlit...")
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=10)
            except Exception:
                streamlit_process.kill()
        if error_msg:
            log.error(error_msg)
        sys.exit(1)

    def already_processed(file: str, mtime: float) -> bool:
        if not os.path.exists(control_file):
            return False
        with open(control_file, "r") as f:
            for line in f:
                try:
                    fname, ts = line.strip().split(",")
                except ValueError:
                    continue
                if fname == file and int(float(ts)) == int(float(mtime)):
                    return True
        return False

    def execute_pipeline(file: str):
        file_path = os.path.join(data_dir, file)
        try:
            mtime = os.path.getmtime(file_path)
        except FileNotFoundError:
            log.warning(f"File vanished before processing: {file}")
            return

        if already_processed(file, mtime):
            log.info(f"[CONTROL] File {file} already processed, skipping.")
            return

        log.info(f"[PIPELINE] Executing IPIP pipeline for file {file}...")
        try:
            run_pipeline(
                pipeline_name=pipeline_name,
                data_dir=data_dir,
                preprocess_file=preprocess_file,
                thresholds_drift=thresholds_drift,
                thresholds_perf=thresholds_perf,
                model_instance=model_instance,
                retrain_mode=retrain_mode,
                random_state=random_state,
                custom_train_file=custom_train_file,
                custom_retrain_file=custom_retrain_file,
                delimiter=delimiter,
                target_file=file if target_file is None else target_file,
                window_size=window_size,
                block_col=block_col,
                ipip_config=ipip_config
            )
            log.info(f"[PIPELINE] Pipeline finished for {file}")
        except Exception as e:
            stop_all(f"Pipeline failed for {file}: {e}")

    class DataFileHandler(FileSystemEventHandler):
        def on_created(self, event):
            self._process_event(event, "created")

        def on_modified(self, event):
            self._process_event(event, "modified")

        def _process_event(self, event, event_type: str):
            if event.is_directory:
                return
            fname = os.path.basename(event.src_path)
            if not fname.lower().endswith(_ALLOWED_EXTS):
                log.info(f"Ignored file: {fname}")
                return
            log.info(f"[WATCHDOG] File {event_type}: {fname} → executing pipeline")
            execute_pipeline(fname)

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    def start_streamlit(pipeline_name: str, port: int | None = None):
        nonlocal streamlit_process
        dashboard_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "web_interface", "dashboard_ipip.py")
        )

        if port is None:
            port = 8501
            if is_port_in_use(port):
                log.info(f"[STREAMLIT] Port {port} occupied. Trying port 8510...")
                port = 8510

        log.info(f"[STREAMLIT] Launching IPIP dashboard on port {port}...")
        try:
            streamlit_process = subprocess.Popen([
                "streamlit", "run", dashboard_path,
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--", "--pipeline_name", pipeline_name
            ])
        except Exception as e:
            stop_all(f"Failed to start Streamlit: {e}")

    def start_watchdog():
        log.info(f"[MONITOR] Watching directory: {data_dir}...")
        event_handler = DataFileHandler()
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

    # --- Initial launch ---
    log.info("[MAIN] Launching IPIP monitor with Watchdog + Streamlit...")
    log.info("[CHECK] Scanning the directory at startup...]")
    for file in os.listdir(data_dir):
        if file.lower().endswith(_ALLOWED_EXTS):
            execute_pipeline(file)

    threading.Thread(target=start_streamlit, args=(pipeline_name, port), daemon=True).start()
    start_watchdog()


if __name__ == "__main__":
    configure_root_logging(logging.INFO)
    from sklearn.ensemble import RandomForestClassifier

    start_monitor_ipip(
        pipeline_name="my_pipeline_ipip",
        data_dir="/home/alex/datos",
        preprocess_file="/home/alex/calmops/IPIP/preprocessing.py",
        thresholds_perf={"balanced_accuracy": 0.6},   
        thresholds_drift={"balanced_accuracy": 0.6},                         
        model_instance=RandomForestClassifier(random_state=42),
        retrain_mode=6,
        random_state=42,
        delimiter=",",
        block_col="chunk",
        ipip_config={
            "p": 20,
            "b": 5,
            "prop_majoritaria": 0.55,   # proportion
            "val_size": 0.20,
            "prop_minor_frac": 0.75
        },
        port=8501,
        persistence="none"
    )
