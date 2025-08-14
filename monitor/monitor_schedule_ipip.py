# monitor/monitor_schedule_ipip.py
# -*- coding: utf-8 -*-
"""
CalmOps Monitor (Scheduled, IPIP)
- Revisa periódicamente un directorio de datos con APScheduler (interval/cron/date).
- Lanza un dashboard Streamlit para monitorización.
- Modos de persistencia opcionales:
    * "none"   : ejecución en primer plano (comportamiento por defecto)
    * "pm2"    : genera runner y arranca con PM2 (auto-restart + on-boot)
    * "docker" : genera Dockerfile + docker-compose y ejecuta en background con restart policy
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

# Importa el runner IPIP por bloques
from pipeline.modules.pipeline_block_ipip import run_pipeline


# =========================
# Helpers
# =========================

def _project_root() -> str:
    """Ruta absoluta a la raíz del proyecto (asumiendo este archivo en monitor/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _which(cmd: str):
    from shutil import which
    return which(cmd)


def _fatal(msg: str, code: int = 1):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
    """Serializa clase + params del modelo para reconstruirlo en el runner."""
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
    """Guarda el JSON de configuración del runner."""
    cfg_dir = os.path.join(base_dir, "pipelines", pipeline_name, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "runner_ipip_schedule_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return cfg_path


def _write_runner_script_schedule(pipeline_name: str, runner_cfg_path: str, base_dir: str) -> str:
    """
    Genera un runner que llama a start_monitor_schedule_ipip con persistence='none'
    para evitar bucles cuando PM2/Docker relanzan el servicio.
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)
    runner_path = os.path.join(pipeline_dir, f"run_{pipeline_name}_ipip_schedule.py")

    header = f"# Auto-generated runner (schedule, IPIP) for pipeline: {pipeline_name}"
    body = f"""
import json, importlib
from monitor.monitor_schedule_ipip import start_monitor_schedule_ipip

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
        block_col=cfg.get("block_col"),
        ipip_config=cfg.get("ipip_config"),
        persistence="none"
    )
"""
    with open(runner_path, "w") as f:
        f.write(header + body)
    return runner_path


# =========================
# PM2 / Docker helpers
# =========================

def pm2_list(prefix: str = "calmops-") -> list[dict]:
    """Lista procesos PM2 cuyo nombre empieza por prefix."""
    if not _which("pm2"):
        print("[ERROR] PM2 no está instalado. sudo npm install -g pm2")
        return []
    try:
        out = subprocess.check_output(["pm2", "jlist"], text=True)
        procs = json.loads(out)
    except Exception as e:
        print(f"[ERROR] Could not obtain PM2 list: {e}")
        return []
    filtered = [p for p in procs if p.get("name", "").startswith(prefix)]
    if not filtered:
        print(f"No PM2 processes with prefix '{prefix}'.")
        return []
    print("PM2 processes:")
    for p in filtered:
        name = p.get("name")
        pid = p.get("pid")
        status = p.get("pm2_env", {}).get("status")
        restart = p.get("pm2_env", {}).get("restart_time")
        print(f"- {name} | pid={pid} | status={status} | restarts={restart}")
    return filtered


def pm2_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    """Detiene y elimina proceso PM2 y borra la carpeta de la pipeline."""
    app_name = f"calmops-schedule-{pipeline_name}-ipip"
    if not _which("pm2"):
        print("[ERROR] PM2 no está instalado. sudo npm install -g pm2")
        return
    try:
        subprocess.call(["pm2", "stop", app_name])
        subprocess.call(["pm2", "delete", app_name])
        subprocess.call(["pm2", "save"])
        print(f"[PM2] Proceso '{app_name}' detenido y eliminado.")
    except Exception as e:
        print(f"[WARN] No se pudo eliminar en PM2: {e}")

    pipeline_path = os.path.join(base_dir, pipeline_name)
    try:
        if os.path.exists(pipeline_path):
            shutil.rmtree(pipeline_path)
            print(f"[FS] Carpeta '{pipeline_path}' eliminada.")
        else:
            print(f"[FS] Carpeta '{pipeline_path}' no existe.")
    except Exception as e:
        print(f"[ERROR] No se pudo eliminar la carpeta: {e}")


def _docker_install_hint() -> str:
    return (
        "Docker requerido pero no encontrado.\n"
        "Instala Docker Engine y docker compose plugin.\n"
    )


def _write_docker_files(pipeline_name: str, runner_script_abs: str, base_dir: str, port: int | None):
    """Genera Dockerfile y docker-compose.yml para la pipeline programada IPIP."""
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)

    dockerfile_path = os.path.join(pipeline_dir, "Dockerfile")
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")

    runner_rel = runner_script_abs.replace(base_dir, "").lstrip(os.sep)
    runner_rel_posix = runner_rel.replace(os.sep, "/")
    runner_in_container = f"/app/{runner_rel_posix}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile (schedule, IPIP) for {pipeline_name}
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

    compose = f"""# Auto-generated docker-compose (schedule, IPIP) for {pipeline_name}
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

    return dockerfile_path, compose_path


def _compose_available():
    if not _which("docker"):
        return None
    try:
        subprocess.check_call(
            ["docker", "compose", "version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return "v2"
    except Exception:
        return "v1" if _which("docker-compose") else None


def _launch_with_docker(pipeline_name: str, runner_script: str, base_dir: str, port: int | None):
    """Construye y arranca docker-compose en modo detach."""
    if not _which("docker"):
        _fatal(_docker_install_hint())

    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)

    try:
        if _compose_available() == "v2":
            subprocess.check_call(["docker", "compose", "up", "-d", "--build"], cwd=pipeline_dir)
        elif _compose_available() == "v1":
            subprocess.check_call(["docker-compose", "up", "-d", "--build"], cwd=pipeline_dir)
        else:
            _fatal("Ni `docker compose` (v2) ni `docker-compose` (v1) disponibles.")
    except Exception as e:
        _fatal(f"Failed to build/run Docker services: {e}\n{_docker_install_hint()}")


def docker_list(prefix: str = "calmops_") -> list[tuple[str, str]]:
    """Lista contenedores Docker cuyo nombre empieza por prefix."""
    if not _which("docker"):
        print("[ERROR] Docker not installed.")
        return []
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"], text=True
        ).strip()
    except Exception as e:
        print(f"[ERROR] docker ps failed: {e}")
        return []

    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, status = line.split("\t", 1)
        if name.startswith(prefix):
            rows.append((name, status))
    if rows:
        print("Docker containers:")
        for name, status in rows:
            print(f"- {name}: {status}")
    else:
        print(f"No docker containers with prefix '{prefix}'.")
    return rows


def docker_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    """Baja el compose de la pipeline y elimina su carpeta."""
    pipeline_dir = os.path.join(base_dir, pipeline_name)
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")
    if not _which("docker"):
        print("[ERROR] Docker not installed.")
        return

    try:
        if os.path.exists(compose_path):
            if _compose_available() == "v2":
                subprocess.call(["docker", "compose", "down", "--volumes", "--remove-orphans"], cwd=pipeline_dir)
            elif _compose_available() == "v1":
                subprocess.call(["docker-compose", "down", "--volumes", "--remove-orphans"], cwd=pipeline_dir)
        else:
            cname = f"calmops_{pipeline_name}_ipip_schedule"
            subprocess.call(["docker", "rm", "-f", cname])
        print(f"[DOCKER] Pipeline '{pipeline_name}' containers removed.")
    except Exception as e:
        print(f"[WARN] Could not bring down docker compose: {e}")

    try:
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)
            print(f"[FS] Folder '{pipeline_dir}' removed.")
    except Exception as e:
        print(f"[ERROR] Could not remove pipeline folder: {e}")


def _launch_with_pm2(pipeline_name: str, runner_script: str, base_dir: str):
    """Arranca el runner programado con PM2."""
    if not _which("pm2"):
        _fatal(
            "PM2 requerido pero no encontrado.\n"
            "Instala Node.js + PM2: sudo npm install -g pm2"
        )

    eco_path = os.path.join(base_dir, "pipelines", pipeline_name, "ecosystem.ipip.schedule.config.js")
    app_name = f"calmops-schedule-{pipeline_name}-ipip"
    python_exec = sys.executable

    runner_script_posix = runner_script.replace("\\", "/")
    python_exec_posix = python_exec.replace("\\", "/")
    base_dir_posix = base_dir.replace("\\", "/")

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
        subprocess.check_call(["pm2", "start", eco_path])
    except Exception as e:
        _fatal(f"Failed to start with PM2: {e}\n(Install with: sudo npm install -g pm2)")

    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        pass
    try:
        user = os.getenv("USER", "") or ""
        if user:
            subprocess.check_call(["pm2", "startup", "-u", user])
        else:
            subprocess.check_call(["pm2", "startup"])
    except Exception:
        pass

    print(f"[PM2] App '{app_name}' started and saved.")


# =========================
# Main scheduled monitor
# =========================

def start_monitor_schedule_ipip(
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
    # IPIP / bloques:
    block_col: str | None = None,
    ipip_config: dict | None = None,
):
    """
    - Programa chequeos periódicos sobre el directorio de datos y corre run_pipeline (IPIP).
    - Lanza Streamlit dashboard.
    - Opcionalmente crea runner para PM2/Docker.
    """
    if schedule is None or "type" not in schedule or "params" not in schedule:
        _fatal("schedule debe ser un dict con claves: 'type' y 'params'.")

    # --- persistencia (early) ---
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
            "block_col": block_col,
            "ipip_config": ipip_config,
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

    # ---------- ejecución normal (no persistente) ----------
    BASE_PIPELINE_DIR = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    OUTPUT_DIR = os.path.join(BASE_PIPELINE_DIR, "modelos")
    CONTROL_DIR = os.path.join(BASE_PIPELINE_DIR, "control")
    LOGS_DIR = os.path.join(BASE_PIPELINE_DIR, "logs")
    METRICS_DIR = os.path.join(BASE_PIPELINE_DIR, "metrics")
    CONFIG_DIR = os.path.join(BASE_PIPELINE_DIR, "config")

    for d in [OUTPUT_DIR, CONTROL_DIR, LOGS_DIR, METRICS_DIR, CONFIG_DIR]:
        os.makedirs(d, exist_ok=True)

    control_file = os.path.join(CONTROL_DIR, "control_file.txt")
    if not os.path.exists(control_file):
        open(control_file, "w").close()

    # Guarda config para dashboard
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
        print("[CRITICAL] Stopping all processes...")
        if streamlit_process and streamlit_process.poll() is None:
            print("[STREAMLIT] Terminating Streamlit...")
            streamlit_process.terminate()
            streamlit_process.wait()
        if error_msg:
            print(f"[ERROR] {error_msg}")
        sys.exit(1)

    def get_records():
        """Lee el control_file (fichero, mtime) para evitar reprocesos."""
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

        print(f"[PIPELINE] Running pipeline (IPIP) for file {file}...")
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
                window_size=window_size,
                block_col=block_col,
                ipip_config=ipip_config
            )
            print(f"[PIPELINE] Pipeline completed for {file}")
        except Exception as e:
            stop_all(f"The pipeline failed for {file}: {e}")

    def check_files():
        """Escanea el directorio buscando ficheros nuevos/modificados y dispara la pipeline."""
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
        DASHBOARD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web_interface", "dashboard_ipip.py"))

        if port is None:
            port = 8501
            if is_port_in_use(port):
                print(f"[STREAMLIT] Port {port} is occupied. Trying 8510...")
                port = 8510

        print(f"[STREAMLIT] Launching dashboard on port {port}...")
        try:
            streamlit_process = subprocess.Popen([
                "streamlit", "run", DASHBOARD_PATH,
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--", "--pipeline_name", pipeline_name
            ])
        except Exception as e:
            stop_all(f"Could not launch Streamlit: {e}")

    print(f"[MAIN] Launching scheduled IPIP monitor with APScheduler...")

    if early_start:
        check_files()

    scheduler = BackgroundScheduler()

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
            print("[DEBUG] IPIP scheduled monitor running...")
            time.sleep(30)
    except KeyboardInterrupt:
        stop_all("[INTERRUPT] Terminated by keyboard.")
    except Exception as e:
        stop_all(f"[FATAL] Unexpected error in monitor: {e}")


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    start_monitor_schedule_ipip(
        pipeline_name="my_pipeline_ipip_schedule",
        data_dir="/path/to/datos",
        preprocess_file="/path/to/preprocessing_ipip.py",
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
            "type": "interval",          # 'interval' | 'cron' | 'date'
            "params": {"minutes": 5}     # Para 'interval'
        },
        early_start=True,
        port=None,                      # o un puerto fijo (p.ej., 8600)
        block_col="mes",
        ipip_config={"p": 20, "b": 5},  # parámetros propios de IPIP (opcional)
        persistence="none"              # "none" | "pm2" | "docker"
    )
