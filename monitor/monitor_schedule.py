# monitor/monitor_schedule.py
# -*- coding: utf-8 -*-
"""
CalmOps Monitor (Scheduled)
- Programa comprobaciones periódicas con APScheduler (interval/cron/date).
- Lanza el dashboard de Streamlit para monitorizar.
- Modos de persistencia opcionales:
    * "none"   : ejecución en primer plano (comportamiento actual)
    * "pm2"    : genera un runner y arranca con PM2 (auto-restart + on-boot)
    * "docker" : genera Dockerfile + docker-compose y arranca en background con restart policy
"""

import os
import sys
import time
import json
import socket
import shutil
import threading
import subprocess

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from calmops.pipeline.pipeline_stream import run_pipeline


# =========================
# Helpers
# =========================

def _project_root() -> str:
    """Return project root as absolute path (assumes this file is monitor/monitor_schedule.py)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _which(cmd: str):
    """Shorthand for shutil.which."""
    from shutil import which
    return which(cmd)


def _fatal(msg: str, code: int = 1):
    """Print a fatal message and exit with given code."""
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
    """
    Serialize a sklearn-like model class + params so the runner can re-instantiate it.
    Falls back to empty-kwargs constructor if get_params is not available.
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


def _write_runner_config(pipeline_name: str, config_obj: dict, base_dir: str) -> str:
    """Write a JSON config for the runner to reconstruct args & model."""
    cfg_dir = os.path.join(base_dir, "pipelines", pipeline_name, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "runner_schedule_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return cfg_path


def _write_runner_script_schedule(pipeline_name: str, runner_cfg_path: str, base_dir: str) -> str:
    """
    Create a runner script that reconstructs the model & arguments and calls start_monitor_schedule
    with persistence='none' to avoid recursion when PM2/Docker relaunch.
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)
    runner_path = os.path.join(pipeline_dir, f"run_{pipeline_name}_schedule.py")

    content = f'''# Auto-generated runner (schedule) for pipeline: {pipeline_name}
import json, importlib
from monitor.monitor_schedule import start_monitor_schedule

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
        persistence="none"  # avoid recursive PM2/Docker spawning
    )
'''
    with open(runner_path, "w") as f:
        f.write(content)
    return runner_path


def _pm2_install_hint() -> str:
    return (
        "PM2 is required but not found.\n"
        "Install Node.js + PM2, e.g. on Debian/Ubuntu:\n"
        "  sudo apt-get update && sudo apt-get install -y nodejs npm\n"
        "  sudo npm install -g pm2\n"
        "Then re-run with persistence='pm2'."
    )


def _docker_install_hint() -> str:
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
        "Then re-run with persistence='docker'."
    )


def _write_docker_files(pipeline_name: str, runner_script_abs: str, base_dir: str, port: int | None):
    """Generate Dockerfile and docker-compose.yml in project root."""
    dockerfile_path = os.path.join(base_dir, "Dockerfile")
    compose_path = os.path.join(base_dir, "docker-compose.yml")
    runner_in_container = "/app" + runner_script_abs.replace(base_dir, "").replace("\\", "/")

    dockerfile = f"""# Auto-generated Dockerfile (schedule) for {pipeline_name}
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \\
    (test -f requirements.txt && pip install --no-cache-dir -r requirements.txt || true)

EXPOSE {port or 8501}

ENV PYTHONUNBUFFERED=1 \\
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \\
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "{runner_in_container}"]
"""

    compose = f"""# Auto-generated docker-compose (schedule) for {pipeline_name}
version: "3.8"
services:
  {pipeline_name}_schedule:
    build: .
    container_name: calmops_{pipeline_name}_schedule
    restart: unless-stopped
    ports:
      - "{(port or 8501)}:{(port or 8501)}"
    volumes:
      - "{base_dir}:/app"
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


def _launch_with_pm2(pipeline_name: str, runner_script: str, base_dir: str):
    """Start the schedule runner with PM2."""
    if not _which("pm2"):
        _fatal(_pm2_install_hint())

    eco_path = os.path.join(base_dir, "pipelines", pipeline_name, "ecosystem.schedule.config.js")
    app_name = f"calmops-schedule-{pipeline_name}"
    python_exec = sys.executable

    ecosystem = f"""
module.exports = {{
  apps: [{{
    name: "{app_name}",
    script: "{runner_script.replace("\\\\","/")}",
    interpreter: "{python_exec.replace("\\\\","/")}",
    interpreter_args: "-u",
    cwd: "{base_dir.replace("\\\\","/")}",
    autorestart: true,
    watch: false,
    max_restarts: 10
  }}]
}};
"""
    with open(eco_path, "w") as f:
        f.write(ecosystem)

    try:
        subprocess.check_call(["pm2", "start", eco_path])
    except Exception as e:
        _fatal(f"Failed to start with PM2: {e}\n{_pm2_install_hint()}")

    # Persist process list (best-effort)
    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        pass
    try:
        subprocess.check_call(["pm2", "startup", "-u", os.getenv("USER", "") or ""])
    except Exception:
        pass

    print(f"[PM2] App '{app_name}' started and saved. It will restart on boot (if pm2 startup is active).")


def _launch_with_docker(pipeline_name: str, runner_script: str, base_dir: str, port: int | None):
    """Build and run docker-compose in detached mode with restart policy."""
    if not _which("docker"):
        _fatal(_docker_install_hint())

    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    # Prefer 'docker compose' (v2), fallback to 'docker-compose' (v1)
    try:
        if _which("docker") and subprocess.call(["docker", "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            subprocess.check_call(["docker", "compose", "up", "-d", "--build"], cwd=_project_root())
        elif _which("docker-compose"):
            subprocess.check_call(["docker-compose", "up", "-d", "--build"], cwd=_project_root())
        else:
            _fatal(
                "Neither `docker compose` (v2) nor `docker-compose` (v1) is available.\n"
                f"{_docker_install_hint()}"
            )
    except Exception as e:
        _fatal(f"Failed to build/run Docker services: {e}\n{_docker_install_hint()}")


# =========================
# Main scheduled monitor
# =========================

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
    persistence: str = "none",   # "none" | "pm2" | "docker"
):
    """
    Starts the scheduled monitoring system:
      - Periodically checks the data directory and triggers run_pipeline on new/modified files
      - Launches Streamlit dashboard
      - Optionally bootstraps persistence via PM2 or Docker

    On PM2/Docker selection without the tools installed, a friendly error is printed and the process exits.
    """

    if schedule is None or "type" not in schedule or "params" not in schedule:
        _fatal("schedule must be a dict with keys: 'type' and 'params'.")

    # --- persistence bootstrap (early) ---
    persistence = (persistence or "none").lower()
    base_dir = _project_root()

    if persistence in ("pm2", "docker"):
        # Prepare runner config + script that reconstructs the model & args
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
        }
        runner_cfg_path = _write_runner_config(pipeline_name, runner_cfg_obj, base_dir)
        runner_script = _write_runner_script_schedule(pipeline_name, runner_cfg_path, base_dir)

        if persistence == "pm2":
            _launch_with_pm2(pipeline_name, runner_script, base_dir)
            print("[PM2] Persistence enabled. Exiting foreground process.")
            return
        else:
            _launch_with_docker(pipeline_name, runner_script, base_dir, port or 8501)
            print("[DOCKER] Persistence enabled via docker-compose. Exiting foreground process.")
            return

    # ---------- regular (non-persistent) flow below ----------
    BASE_PIPELINE_DIR = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    OUTPUT_DIR = os.path.join(BASE_PIPELINE_DIR, "modelos")
    CONTROL_DIR = os.path.join(BASE_PIPELINE_DIR, "control")
    LOGS_DIR = os.path.join(BASE_PIPELINE_DIR, "logs")
    METRICS_DIR = os.path.join(BASE_PIPELINE_DIR, "resultados")
    CONFIG_DIR = os.path.join(BASE_PIPELINE_DIR, "config")

    for d in [OUTPUT_DIR, CONTROL_DIR, LOGS_DIR, METRICS_DIR, CONFIG_DIR]:
        os.makedirs(d, exist_ok=True)

    control_file = os.path.join(CONTROL_DIR, "archivo_control.txt")
    if not os.path.exists(control_file):
        open(control_file, "w").close()

    # Save config for dashboard
    config_path = os.path.join(CONFIG_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "pipeline_name": pipeline_name,
            "data_dir": data_dir,
            "preprocess_file": preprocess_file
        }, f)

    streamlit_process = None

    def stop_all(error_msg=None):
        nonlocal streamlit_process
        print("[CRITICAL] Stopping all processes...")
        if streamlit_process and streamlit_process.poll() is None:
            print("[STREAMLIT] Terminating Streamlit...")
            streamlit_process.terminate()
            streamlit_process.wait()
        if error_msg:
            print(f"[ERROR] {error_msg}")
        sys.exit(1)

    def get_records():
        records = {}
        if os.path.exists(control_file):
            with open(control_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        try:
                            records[parts[0]] = int(float(parts[1]))
                        except Exception:
                            continue
        return records

    def run_pipeline_for_file(file):
        file_path = os.path.join(data_dir, file)
        mtime = int(os.path.getmtime(file_path))

        print(f"[PIPELINE] Running pipeline for file {file}...")
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
                target_file=file,
                window_size=window_size
            )
            print(f"[PIPELINE] Pipeline completed for {file}")
        except Exception as e:
            stop_all(f"The pipeline failed for {file}: {e}")

    def check_files():
        try:
            print("[CHECK] Checking files in directory...")
            records = get_records()
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if not os.path.isfile(file_path):
                    continue
                if not file.lower().endswith((".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")):
                    continue

                mtime = int(os.path.getmtime(file_path))
                if file not in records or mtime > records[file]:
                    print(f"[CHECK] New or modified file: {file}")
                    run_pipeline_for_file(file)
                else:
                    print(f"[CONTROL] {file} already processed, skipping.")
        except Exception as e:
            stop_all(f"Error in file check: {e}")

    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', p)) == 0

    def start_streamlit(pipeline_name, port=None):
        nonlocal streamlit_process
        DASHBOARD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web_interface", "dashboard.py"))

        if port is None:
            port = 8501
            if is_port_in_use(port):
                print(f"[STREAMLIT] Port {port} is occupied. Trying 8510...")
                port = 8510

        print(f"[STREAMLIT] Launching dashboard on port {port}...")
        try:
            # Bind 0.0.0.0 to work inside Docker as well
            streamlit_process = subprocess.Popen([
                "streamlit", "run", DASHBOARD_PATH,
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--", "--pipeline_name", pipeline_name
            ])
        except Exception as e:
            stop_all(f"Could not launch Streamlit: {e}")

    print(f"[MAIN] Launching monitor with APScheduler...")

    if early_start:
        check_files()

    scheduler = BackgroundScheduler()

    # Schedule type and parameters are combined in the 'schedule' argument
    stype = schedule["type"]
    sparams = schedule["params"]
    if stype == "interval":
        scheduler.add_job(check_files, IntervalTrigger(**sparams))
    elif stype == "cron":
        scheduler.add_job(check_files, CronTrigger(**sparams))
    elif stype == "date":
        scheduler.add_job(check_files, DateTrigger(run_date=sparams["run_date"]))
    else:
        stop_all(f"[ERROR] Invalid schedule type: {stype}")

    scheduler.start()

    threading.Thread(target=start_streamlit, args=(pipeline_name, port), daemon=True).start()

    try:
        while True:
            print("[DEBUG] Monitor running...")
            time.sleep(30)
    except KeyboardInterrupt:
        stop_all("[INTERRUPT] Terminated by keyboard.")
    except Exception as e:
        stop_all(f"[FATAL] Unexpected error in monitor: {e}")


# =========================
# Utility functions
# =========================

def list_pipelines(base_dir="pipelines"):
    """List all existing pipelines in the specified directory."""
    try:
        pipelines = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not pipelines:
            print("No pipelines found.")
        else:
            print("Available Pipelines:")
            for pipeline in pipelines:
                print(f"- {pipeline}")
    except Exception as e:
        print(f"[ERROR] Could not list pipelines: {e}")


def delete_pipeline(pipeline_name, base_dir="pipelines"):
    """Delete a specified pipeline and all its files."""
    pipeline_path = os.path.join(base_dir, pipeline_name)
    if not os.path.exists(pipeline_path):
        print(f"[ERROR] Pipeline {pipeline_name} does not exist.")
        return
    try:
        confirmation = input(f"Are you sure you want to delete the pipeline '{pipeline_name}'? This action is irreversible. (y/n): ")
        if confirmation.lower() != 'y':
            print("Deletion cancelled.")
            return
        shutil.rmtree(pipeline_path)
        print(f"[INFO] Pipeline '{pipeline_name}' has been deleted.")
    except Exception as e:
        print(f"[ERROR] Failed to delete pipeline {pipeline_name}: {e}")


# =========================
# Example of usage
# =========================

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    start_monitor_schedule(
        pipeline_name="my_pipeline_schedule",
        data_dir="/home/alex/demo/data",
        preprocess_file="/home/alex/demo/pipeline/preprocessing.py",
        thresholds_drift={"balanced_accuracy": 0.8},
        thresholds_perf={"accuracy": 0.9, "balanced_accuracy": 0.9, "f1": 0.85},
        model_instance=RandomForestClassifier(random_state=42),
        retrain_mode=0,
        fallback_mode=2,
        random_state=42,
        param_grid={"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        cv=5,
        delimiter=",",
        schedule={
            "type": "interval",  # Options: 'interval', 'cron', 'date'
            "params": {"minutes": 2}  # For 'interval' type
        },
        early_start=True,
        port=None,                # Or specify a fixed port like 8600
        persistence="none"        # "none" | "pm2" | "docker"
    )
