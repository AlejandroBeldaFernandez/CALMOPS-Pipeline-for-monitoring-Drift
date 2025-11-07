# monitor/monitor_schedule_block.py
# -*- coding: utf-8 -*-
"""
CalmOps Monitor (Scheduled, BLOQUES)
- Revisa periódicamente un directorio con APScheduler (interval/cron/date).
- Lanza la pipeline por BLOQUES y el dashboard Streamlit de bloques.
- Modos de persistencia:
    * "none"   : primer plano
    * "pm2"    : genera runner y lanza con PM2 (auto-restart + on-boot)
    * "docker" : genera Dockerfile + docker-compose y corre en background
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
from typing import Optional, Dict, Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _TZ_EUROPE_MADRID = ZoneInfo("Europe/Madrid")
except Exception:
    _TZ_EUROPE_MADRID = None

# Importa la pipeline por BLOQUES
from pipeline_block.pipeline_block import run_pipeline as run_pipeline_blocks


# =========================
# Logging (formato unificado)
# =========================

_LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"

def configure_root_logging(level: int = logging.INFO) -> None:
    """
    Configura el root logger con un único StreamHandler a stdout y
    el formato del monitor. Limpia handlers previos, captura warnings
    y reduce verbosidad de librerías ruidosas.
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
    Devuelve un logger que hereda el handler/format del root.
    No añade handlers (evita duplicados).
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


# Logger por defecto del módulo; dentro de start_monitor_schedule_block() se renombra por pipeline
log = _get_logger()
logging.getLogger("apscheduler").setLevel(logging.WARNING)


# =========================
# Helpers
# =========================

def _project_root() -> str:
    """Absolute path al root del proyecto (asumiendo este archivo en monitor/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _which(cmd: str):
    from shutil import which
    return which(cmd)


def _fatal(msg: str, code: int = 1):
    log.critical(msg)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
    """
    Serializa clase + parámetros del modelo sklearn-like para recrearlo en el runner.
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
    """Escribe JSON con config del runner."""
    cfg_dir = os.path.join(base_dir, "pipelines", pipeline_name, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "runner_schedule_blocks_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return cfg_path


def _write_runner_script(pipeline_name: str, runner_cfg_path: str, base_dir: str) -> str:
    """
    Crea script runner que reconstruye el modelo y llama a start_monitor_schedule_block
    con persistence='none' (evita recursión en PM2/Docker).
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)
    runner_path = os.path.join(pipeline_dir, f"run_{pipeline_name}_blocks_schedule.py")

    content = f'''# Auto-generated runner (schedule, blocks) for pipeline: {pipeline_name}
import json, importlib
from monitor.monitor_schedule_block import start_monitor_schedule_block

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

    start_monitor_schedule_block(
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
        persistence="none",
        block_col=cfg.get("block_col")
    )
'''
    with open(runner_path, "w") as f:
        f.write(content)
    return runner_path


# =========================
# PM2 helpers (list / delete)
# =========================

def pm2_list(prefix: str = "calmops-") -> list[dict]:
    if not _which("pm2"):
        log.error("PM2 no está instalado. sudo npm install -g pm2")
        return []
    try:
        out = subprocess.check_output(["pm2", "jlist"], text=True)
        procs = json.loads(out)
    except Exception as e:
        log.error(f"No se pudo obtener la lista de PM2: {e}")
        return []
    filtered = [p for p in procs if p.get("name", "").startswith(prefix)]
    if not filtered:
        log.info(f"No hay procesos PM2 con prefijo '{prefix}'.")
        return []
    log.info("Procesos PM2:")
    for p in filtered:
        name = p.get("name")
        pid = p.get("pid")
        status = p.get("pm2_env", {}).get("status")
        restart = p.get("pm2_env", {}).get("restart_time")
        log.info(f"- {name} | pid={pid} | status={status} | restarts={restart}")
    return filtered


def pm2_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    app_name = f"calmops-schedule-{pipeline_name}-blocks"
    if not _which("pm2"):
        log.error("[ERROR] PM2 no está instalado.")
        return
    try:
        subprocess.call(["pm2", "stop", app_name])
        subprocess.call(["pm2", "delete", app_name])
        subprocess.call(["pm2", "save"])
        log.info(f"[PM2] Proceso '{app_name}' detenido y eliminado.")
    except Exception as e:
        log.warning(f"No se pudo eliminar en PM2: {e}")

    pipeline_path = os.path.join(base_dir, pipeline_name)
    try:
        if os.path.exists(pipeline_path):
            shutil.rmtree(pipeline_path)
            log.info(f"[FS] Carpeta '{pipeline_path}' eliminada.")
        else:
            log.info(f"[FS] Carpeta '{pipeline_path}' no existe.")
    except Exception as e:
        log.error(f"No se pudo eliminar la carpeta de la pipeline: {e}")


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


def _write_docker_files(pipeline_name: str, runner_script_abs: str, base_dir: str, port: int | None):
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)

    dockerfile_path = os.path.join(pipeline_dir, "Dockerfile")
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")

    runner_rel = runner_script_abs.replace(base_dir, "").lstrip(os.sep)
    runner_rel_posix = runner_rel.replace(os.sep, "/")
    runner_in_container = f"/app/{runner_rel_posix}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile (schedule, blocks) for {pipeline_name}
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

    compose = f"""# Auto-generated docker-compose (schedule, blocks) for {pipeline_name}
version: "3.8"
services:
  {pipeline_name}_blocks_schedule:
    build:
      context: ../../
      dockerfile: ./pipelines/{pipeline_name}/Dockerfile
    container_name: calmops_{pipeline_name}_blocks_schedule
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


def _launch_with_docker(pipeline_name: str, runner_script: str, base_dir: str, port: int | None):
    if not _which("docker"):
        _fatal(_docker_install_hint())

    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    try:
        if _which("docker") and subprocess.call(
            ["docker", "compose", "version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) == 0:
            subprocess.check_call(["docker", "compose", "up", "-d", "--build"], cwd=pipeline_dir)
        elif _which("docker-compose"):
            subprocess.check_call(["docker-compose", "up", "-d", "--build"], cwd=pipeline_dir)
        else:
            _fatal(
                "Neither `docker compose` (v2) nor `docker-compose` (v1) is available.\n"
                f"{_docker_install_hint()}"
            )
    except Exception as e:
        _fatal(f"Failed to build/run Docker services: {e}\n{_docker_install_hint()}")


def docker_list(prefix: str = "calmops_") -> list[Tuple[str, str]]:
    if not _which("docker"):
        log.error("Docker no instalado.")
        return []
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"], text=True
        ).strip()
    except Exception as e:
        log.error(f"docker ps falló: {e}")
        return []
    rows: list[Tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, status = line.split("\t", 1)
        if name.startswith(prefix):
            rows.append((name, status))
    if rows:
        log.info("Contenedores Docker:")
        for name, status in rows:
            log.info(f"- {name}: {status}")
    else:
        log.info(f"No hay contenedores con prefijo '{prefix}'.")
    return rows


def docker_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    pipeline_dir = os.path.join(base_dir, pipeline_name)
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")
    if not _which("docker"):
        log.error("Docker no instalado.")
        return
    try:
        if os.path.exists(compose_path):
            if subprocess.call(["docker", "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                subprocess.call(["docker", "compose", "down", "--volumes", "--remove-orphans"], cwd=pipeline_dir)
            elif _which("docker-compose"):
                subprocess.call(["docker-compose", "down", "--volumes", "--remove-orphans"], cwd=pipeline_dir)
        else:
            cname = f"calmops_{pipeline_name}_blocks_schedule"
            subprocess.call(["docker", "rm", "-f", cname])
        log.info(f"[DOCKER] Pipeline '{pipeline_name}' (blocks schedule) eliminada.")
    except Exception as e:
        log.warning(f"No se pudo bajar docker compose: {e}")

    try:
        if os.path.exists(pipeline_dir):
            shutil.rmtree(pipeline_dir)
            log.info(f"[FS] Carpeta '{pipeline_dir}' eliminada.")
    except Exception as e:
        log.error(f"No se pudo eliminar la carpeta: {e}")


# =========================
# PM2 launcher (schedule blocks)
# =========================

def _launch_with_pm2(pipeline_name: str, runner_script: str, base_dir: str, logger: logging.Logger | None = None):
    """
    Launch a pipeline with PM2.
    Works on Linux/macOS. On Windows, PM2 can run but automatic startup on boot won't work.
    """
    import os, sys, subprocess, logging

    if logger is None:
        logger = logging.getLogger("calmops.monitor")

    if not _which("pm2"):
        _fatal(
            "PM2 is required but not found.\n"
            "Install it with: sudo npm install -g pm2\n"
            "Then re-run with persistence='pm2'."
        )

    eco_path = os.path.join(base_dir, "pipelines", pipeline_name, "ecosystem.blocks.schedule.config.js")
    app_name = f"calmops-schedule-{pipeline_name}-blocks"

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

    # Start the PM2 app
    try:
        subprocess.check_call(["pm2", "start", eco_path])
    except Exception as e:
        _fatal(f"Failed to start with PM2: {e}\n(Install with: sudo npm install -g pm2)")

    # Save the PM2 process list
    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        logger.warning("[PM2] Could not save PM2 process list.")

    # Attempt auto-start on Linux/macOS
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
        logger.info("[PM2] Windows detected: automatic startup on boot not configured.")

    logger.info(f"[PM2] App '{app_name}' started and saved. Autorestart enabled.")



# =========================
# Main scheduled monitor (BLOQUES)
# =========================

_ALLOWED_EXTS = (".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")

def start_monitor_schedule_block(
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
    schedule: dict = None,   # {"type": "interval"|"cron"|"date", "params": {...}}
    window_size: Optional[int] = None,
    early_start: bool = False,
    port: Optional[int] = None,
    persistence: str = "none",   # "none" | "pm2" | "docker"
    # --- BLOQUES ---
    block_col: Optional[str] = None,
):
    """
    Sistema programado por BLOQUES:
      - Comprueba periódicamente el directorio y ejecuta run_pipeline_blocks() en nuevos/modificados.
      - Lanza el dashboard de bloques.
      - Puede arrancar persistente con PM2 o Docker.
    """
    # Logging global con formato del monitor
    configure_root_logging(logging.INFO)

    # Logger específico por pipeline
    global log
    log = _get_logger(f"calmops.monitor.schedule_blocks.{pipeline_name}")

    if schedule is None or "type" not in schedule or "params" not in schedule:
        _fatal("schedule must be a dict with keys: 'type' and 'params'.")

    # --- persistence bootstrap (early) ---
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
            "model_spec": model_spec,
            "block_col": block_col,
            "monitor_type": "monitor_schedule_block",
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

    # ---------- regular (non-persistent) flow ----------
    BASE_PIPELINE_DIR = os.path.join(os.getcwd(), "pipelines", pipeline_name)
    OUTPUT_DIR  = os.path.join(BASE_PIPELINE_DIR, "modelos")
    CONTROL_DIR = os.path.join(BASE_PIPELINE_DIR, "control")
    LOGS_DIR    = os.path.join(BASE_PIPELINE_DIR, "logs")
    METRICS_DIR = os.path.join(BASE_PIPELINE_DIR, "resultados")
    CONFIG_DIR  = os.path.join(BASE_PIPELINE_DIR, "config")

    for d in [OUTPUT_DIR, CONTROL_DIR, LOGS_DIR, METRICS_DIR, CONFIG_DIR]:
        os.makedirs(d, exist_ok=True)

    control_file = os.path.join(CONTROL_DIR, "control_file.txt")
    if not os.path.exists(control_file):
        open(control_file, "w").close()

    # Config para dashboard (bloques)
    config_path = os.path.join(CONFIG_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "pipeline_name": pipeline_name,
            "data_dir": data_dir,
            "preprocess_file": preprocess_file,
            "block_col": block_col,
            "mode": "blocks"
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

    def get_records() -> Dict[str, int]:
        """Lee control_file con ficheros + mtimes procesados. <filename>,<mtime>"""
        records: Dict[str, int] = {}
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

    def run_pipeline_for_file(file: str):
        file_path = os.path.join(data_dir, file)
        mtime = int(os.path.getmtime(file_path))
        log.info(f"[PIPELINE] (blocks) Running pipeline for {file} (mtime={mtime})...")
        try:
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
                target_file=file,          # basename (alineado con control_file)
                window_size=window_size,
                block_col=block_col,
            )
            log.info(f"[PIPELINE] (blocks) Completed for {file}")
        except Exception as e:
            stop_all(f"The block pipeline failed for {file}: {e}")

    def check_files():
        """Busca nuevos/modificados y dispara procesamiento."""
        try:
            log.info("[CHECK] Scanning directory for new/modified files...")
            records = get_records()
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if not os.path.isfile(file_path):
                    continue
                if not file.lower().endswith(_ALLOWED_EXTS):
                    continue
                mtime = int(os.path.getmtime(file_path))
                if file not in records or mtime > records[file]:
                    log.info(f"[CHECK] New/updated: {file}")
                    run_pipeline_for_file(file)
                else:
                    log.info(f"[CONTROL] {file} already processed (stored mtime={records[file]}), skipping.")
        except Exception as e:
            stop_all(f"Error during directory scan: {e}")

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', p)) == 0

    def start_streamlit(pipeline_name: str, port: Optional[int] = None):
        nonlocal streamlit_process
        dashboard_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "web_interface", "dashboard_block.py")
        )

        if port is None:
            port = 8501
            if is_port_in_use(port):
                log.info(f"[STREAMLIT] Port {port} is occupied. Trying 8510...")
                port = 8510

        log.info(f"[STREAMLIT] Launching blocks dashboard on port {port}...")
        log.info(f"Starting Streamlit dashboard for pipeline {pipeline_name} ")
        log.info(f"Local URL: http://localhost:{port}")
        try:
            streamlit_process = subprocess.Popen([
                "streamlit", "run", dashboard_path,
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--", "--pipeline_name", pipeline_name
            ])
        except Exception as e:
            stop_all(f"Could not launch Streamlit (blocks): {e}")

    log.info("[MAIN] Launching scheduled blocks monitor with APScheduler...")

    # Primera ejecución (opcional)
    if early_start:
        check_files()

    # Scheduler robusto
    scheduler = BackgroundScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 300
        },
        timezone=_TZ_EUROPE_MADRID if _TZ_EUROPE_MADRID else None
    )

    if schedule is None:
        _fatal("You must provide a schedule dict with 'type' and 'params'.")

    stype = schedule["type"]
    sparams = dict(schedule["params"])

    if stype == "interval":
        jitter = sparams.pop("jitter", 10)
        trigger = IntervalTrigger(jitter=jitter, **sparams)
        scheduler.add_job(
            check_files, trigger=trigger,
            id=f"check_files:{pipeline_name}:blocks",
            replace_existing=True, coalesce=True,
            max_instances=1, misfire_grace_time=300
        )
    elif stype == "cron":
        trigger = CronTrigger(**sparams)
        scheduler.add_job(
            check_files, trigger=trigger,
            id=f"check_files:{pipeline_name}:blocks",
            replace_existing=True, coalesce=True,
            max_instances=1, misfire_grace_time=300
        )
    elif stype == "date":
        trigger = DateTrigger(run_date=sparams["run_date"])
        scheduler.add_job(
            check_files, trigger=trigger,
            id=f"check_files:{pipeline_name}:blocks",
            replace_existing=True, coalesce=True,
            max_instances=1, misfire_grace_time=300
        )
    else:
        stop_all(f"[ERROR] Invalid schedule type: {stype}")

    scheduler.start()

    threading.Thread(target=start_streamlit, args=(pipeline_name, port), daemon=True).start()

    try:
        while True:
            log.info("Monitor heartbeat (BLOCKS)...")
            time.sleep(30)
    except KeyboardInterrupt:
        stop_all("[INTERRUPT] Terminated by keyboard.")
    except Exception as e:
        stop_all(f"[FATAL] Unexpected error in monitor: {e}")


# =========================
# Utils (listar / borrar)
# =========================

def list_pipelines(base_dir="pipelines"):
    try:
        pipelines = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
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
        if confirmation.lower() != 'y':
            log.info("Deletion cancelled.")
            return
        shutil.rmtree(pipeline_path)
        log.info(f"[INFO] Pipeline '{pipeline_name}' has been deleted.")
    except Exception as e:
        log.error(f"[ERROR] Failed to delete pipeline {pipeline_name}: {e}")


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Asegura el formato unificado al ejecutar directamente
    configure_root_logging(logging.INFO)

    from sklearn.ensemble import RandomForestClassifier

    start_monitor_schedule_block(
        pipeline_name="my_pipeline_blocks_schedule",
        data_dir="/home/alex/datos",
        preprocess_file="/home/alex/calmops/pipeline_block/preprocessing.py",
        thresholds_drift={"balanced_accuracy": 0.8},
        thresholds_perf={"balanced_accuracy": 0.8},
        model_instance=RandomForestClassifier(random_state=42),
        retrain_mode=6,
        fallback_mode=2,
        random_state=42,
        param_grid={
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },
        cv=3,
        delimiter=",",
        schedule={"type": "interval", "params": {"minutes": 2}},
        early_start=True,
        port=None,              # p.ej., 8600 si necesitas puerto fijo
        persistence="none",     # "none" | "pm2" | "docker"
        block_col=None          # si None, la pipeline intentará detectar "block"/"block_id"
    )
