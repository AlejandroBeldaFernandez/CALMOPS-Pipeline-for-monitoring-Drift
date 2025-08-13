# monitor/monitor.py
# -*- coding: utf-8 -*-
"""
CalmOps Monitor
- Watches a data directory for new/modified files and triggers the ML pipeline.
- Launches a Streamlit dashboard for monitoring.
- Optional persistence modes:
    * "none"   : run in foreground (current behavior)
    * "pm2"    : generate a runner and start with PM2 (auto restart + on-boot)
    * "docker" : generate Dockerfile + docker-compose and run in background with restart policy
"""

import os
import sys
import time
import json
import socket
import shutil
import threading
import subprocess
import importlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.ensemble import RandomForestClassifier
from pipeline.pipeline_stream import run_pipeline


# =========================
# Helpers
# =========================

def _project_root() -> str:
    """Return project root as absolute path (assumes this file is monitor/monitor.py)."""
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
    cfg_path = os.path.join(cfg_dir, "runner_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return cfg_path


def _write_runner_script(pipeline_name: str, runner_cfg_path: str, base_dir: str) -> str:
    """
    Create a runner script that reconstructs the model & arguments and calls start_monitor
    with persistence='none' to avoid recursion when PM2/Docker relaunch.
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)
    runner_path = os.path.join(pipeline_dir, f"run_{pipeline_name}.py")

    content = f'''# Auto-generated runner for pipeline: {pipeline_name}
import os, sys, json, importlib

# Ensure project root on sys.path (two levels up from this file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from monitor.monitor import start_monitor

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
        persistence="none"  # avoid recursive PM2/Docker spawning
    )
'''
    with open(runner_path, "w") as f:
        f.write(content)
    return runner_path


# =========================
# Persistence Launchers
# =========================
def pm2_list(prefix: str = "calmops-") -> list[dict]:
    """
    Return a list of PM2 apps (as dicts) whose name starts with `prefix`.
    Also imprime un resumen legible.
    Requiere pm2 instalado.
    """
    if not _which("pm2"):
        print("[ERROR] PM2 no está instalado. Instálalo con: sudo npm install -g pm2")
        return []

    try:
        # jlist devuelve JSON con la descripción de procesos
        out = subprocess.check_output(["pm2", "jlist"], text=True)
        procs = json.loads(out)
    except Exception as e:
        print(f"[ERROR] No se pudo obtener la lista de PM2: {e}")
        return []

    filtered = [p for p in procs if p.get("name", "").startswith(prefix)]
    if not filtered:
        print(f"No hay procesos PM2 con prefijo '{prefix}'.")
        return []

    print("Procesos PM2:")
    for p in filtered:
        name = p.get("name")
        pid = p.get("pid")
        status = p.get("pm2_env", {}).get("status")
        restart = p.get("pm2_env", {}).get("restart_time")
        print(f"- {name} | pid={pid} | status={status} | restarts={restart}")
    return filtered


def pm2_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    """
    Detiene y elimina el proceso PM2 de la pipeline y borra su carpeta:
      - pm2 delete calmops-<pipeline_name>
      - rm -rf pipelines/<pipeline_name>

    Usa con cuidado: elimina todos los artefactos de esa pipeline.
    """
    app_name = f"calmops-{pipeline_name}"
    if not _which("pm2"):
        print("[ERROR] PM2 no está instalado. Instálalo con: sudo npm install -g pm2")
        return

    # 1) detener y borrar en PM2 (ignorar errores si no existe)
    try:
        subprocess.call(["pm2", "stop", app_name])
        subprocess.call(["pm2", "delete", app_name])
        # guardar lista para restauración tras reinicio
        subprocess.call(["pm2", "save"])
        print(f"[PM2] Proceso '{app_name}' detenido y eliminado.")
    except Exception as e:
        print(f"[WARN] No se pudo eliminar en PM2: {e}")

    # 2) borrar carpeta de la pipeline
    pipeline_path = os.path.join(base_dir, pipeline_name)
    try:
        if os.path.exists(pipeline_path):
            shutil.rmtree(pipeline_path)
            print(f"[FS] Carpeta '{pipeline_path}' eliminada.")
        else:
            print(f"[FS] Carpeta '{pipeline_path}' no existe.")
    except Exception as e:
        print(f"[ERROR] No se pudo eliminar la carpeta de la pipeline: {e}")


def _pm2_install_hint() -> str:
    return (
        "PM2 is required but not found.\n"
        "Install Node.js + PM2, e.g. on Debian/Ubuntu:\n"
        "  sudo apt-get update && sudo apt-get install -y nodejs npm\n"
        "  sudo npm install -g pm2\n"
        "Then re-run with persistence='pm2'."
    )

def _docker_available():
    return shutil.which("docker") is not None

def _compose_available():
    # docker compose v2 preferred, fallback to docker-compose v1
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

def docker_list(all_containers: bool = True):
    """
    Print list of docker containers (like `docker ps`).
    """
    if not _docker_available():
        print("[ERROR] Docker no está instalado. Instálalo y vuelve a intentarlo.")
        return
    cmd = ["docker", "ps", "-a"] if all_containers else ["docker", "ps"]
    subprocess.call(cmd)

def docker_logs(container_name: str, tail: int = 200, follow: bool = True):
    """
    Stream logs for a given container.
    """
    if not _docker_available():
        print("[ERROR] Docker no está instalado.")
        return
    cmd = ["docker", "logs", f"--tail={tail}"]
    if follow:
        cmd.append("-f")
    cmd.append(container_name)
    subprocess.call(cmd)

def docker_stop(container_name: str, remove: bool = False):
    """
    Stop a container (and optionally remove it).
    """
    if not _docker_available():
        print("[ERROR] Docker no está instalado.")
        return
    subprocess.call(["docker", "stop", container_name])
    if remove:
        subprocess.call(["docker", "rm", "-f", container_name])

def docker_compose_up(project_root: str = None, service: str | None = None, build: bool = True, detach: bool = True):
    """
    `docker compose up` (o docker-compose up) desde la raíz del proyecto.
    """
    if _compose_available() is None:
        print("[ERROR] Ni `docker compose` (v2) ni `docker-compose` (v1) están disponibles.")
        return
    cwd = project_root or _project_root()
    is_v2 = _compose_available() == "v2"
    base = ["docker", "compose"] if is_v2 else ["docker-compose"]
    cmd = base + ["up"]
    if detach:
        cmd.append("-d")
    if build:
        cmd.append("--build")
    if service:
        cmd.append(service)
    subprocess.call(cmd, cwd=cwd)

def docker_compose_down(project_root: str = None, remove_images: bool = False, remove_volumes: bool = False):
    """
    `docker compose down` (o docker-compose down) para parar y limpiar.
    """
    if _compose_available() is None:
        print("[ERROR] Ni `docker compose` (v2) ni `docker-compose` (v1) están disponibles.")
        return
    cwd = project_root or _project_root()
    is_v2 = _compose_available() == "v2"
    base = ["docker", "compose"] if is_v2 else ["docker-compose"]
    cmd = base + ["down"]
    if remove_images:
        cmd += ["--rmi", "all"]
    if remove_volumes:
        cmd += ["--volumes"]
    subprocess.call(cmd, cwd=cwd)

def docker_purge_pipeline(pipeline_name: str, also_delete_files: bool = False):
    """
    Detiene y elimina el contenedor de la pipeline y (opcional) borra su carpeta.
    Contenedor creado por nuestro compose: `calmops_{pipeline_name}`
    """
    container = f"calmops_{pipeline_name}"
    if _docker_available():
        subprocess.call(["docker", "rm", "-f", container])
    else:
        print("[WARN] Docker no está instalado; no puedo eliminar el contenedor.")

    if also_delete_files:
        delete_pipeline(pipeline_name)  # reutiliza tu función existente
        

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


def _launch_with_pm2(pipeline_name: str, runner_script: str, base_dir: str):
    """Start the runner with PM2, enable save and startup (best-effort) with friendly errors."""
    if not _which("pm2"):
        _fatal(_pm2_install_hint())

    eco_path = os.path.join(base_dir, "pipelines", pipeline_name, "ecosystem.config.js")
    app_name = f"calmops-{pipeline_name}"

    # Precompute POSIX-ish paths to avoid backslashes inside f-string expressions
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


def _write_docker_files(pipeline_name: str, runner_script_abs: str, base_dir: str, port: int | None):
    """
    Generate Dockerfile and docker-compose.yml inside pipelines/<pipeline_name>/.
    Build context is the project root so the entire repo is copied into /app.
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)

    dockerfile_path = os.path.join(pipeline_dir, "Dockerfile")
    compose_path = os.path.join(pipeline_dir, "docker-compose.yml")

    # Map host runner path -> container /app/...
    runner_rel = runner_script_abs.replace(base_dir, "").lstrip(os.sep)
    runner_rel_posix = runner_rel.replace(os.sep, "/")
    runner_in_container = f"/app/{runner_rel_posix}"

    exposed_port = port or 8501

    dockerfile = f"""# Auto-generated Dockerfile for {pipeline_name}
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The build context is the project root (set in docker-compose.yml)
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \\
    (test -f requirements.txt && pip install --no-cache-dir -r requirements.txt || true)

EXPOSE {exposed_port}

ENV PYTHONUNBUFFERED=1 \\
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \\
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "{runner_in_container}"]
"""

    # docker-compose.yml lives in pipelines/<pipeline_name>/
    # Build context: repo root (../../ from here)
    # Dockerfile path: ./pipelines/<pipeline_name>/Dockerfile
    # Mount the whole repo so hot-reload of code is easy during dev
    compose = f"""# Auto-generated docker-compose for {pipeline_name}
version: "3.8"
services:
  {pipeline_name}:
    build:
      context: ../../
      dockerfile: ./pipelines/{pipeline_name}/Dockerfile
    container_name: calmops_{pipeline_name}
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
    """
    Build and run docker-compose in detached mode from pipelines/<pipeline_name>/,
    with friendly errors if Docker/Compose are missing.
    """
    def _which(cmd: str):
        from shutil import which
        return which(cmd)

    def _fatal(msg: str, code: int = 1):
        print(f"[FATAL] {msg}", file=sys.stderr)
        sys.exit(code)

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
        )

    if not _which("docker"):
        _fatal(_docker_install_hint())

    # Ensure files are written under pipelines/<pipeline_name>/
    _write_docker_files(pipeline_name, runner_script, base_dir, port or 8501)

    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)

    # Prefer 'docker compose' (v2), fallback to 'docker-compose' (v1)
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



# =========================
# Main monitor entrypoint
# =========================

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
    persistence: str = "none",   # "none" | "pm2" | "docker"
):
    """
    Starts the monitoring system:
      - Watches the data directory for new/modified files and triggers run_pipeline
      - Launches Streamlit dashboard
      - Optionally bootstraps persistence via PM2 or Docker

    On PM2/Docker selection without the tools installed, a friendly error is printed and the process exits.
    """

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
            "target_file": target_file,
            "window_size": window_size,
            "port": port,
            "model_spec": model_spec,
        }
        runner_cfg_path = _write_runner_config(pipeline_name, runner_cfg_obj, base_dir)
        runner_script = _write_runner_script(pipeline_name, runner_cfg_path, base_dir)

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
    OUTPUT_DIR = os.path.join(BASE_PIPELINE_DIR, "models")
    CONTROL_DIR = os.path.join(BASE_PIPELINE_DIR, "control")
    LOGS_DIR = os.path.join(BASE_PIPELINE_DIR, "logs")
    METRICS_DIR = os.path.join(BASE_PIPELINE_DIR, "metrics")
    CONFIG_DIR = os.path.join(BASE_PIPELINE_DIR, "config")

    # Ensure directories exist
    for d in [OUTPUT_DIR, CONTROL_DIR, LOGS_DIR, METRICS_DIR, CONFIG_DIR]:
        os.makedirs(d, exist_ok=True)

    control_file = os.path.join(CONTROL_DIR, "control_file.txt")
    if not os.path.exists(control_file):
        open(control_file, "w").close()

    # Save basic config for Streamlit
    config_path = os.path.join(CONFIG_DIR, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump({
                "pipeline_name": pipeline_name,
                "data_dir": data_dir,
                "preprocess_file": preprocess_file
            }, f)
    except Exception as e:
        _fatal(f"Failed to save config.json: {e}")

    streamlit_process = None  # handle to the Streamlit process

    def stop_all(error_msg=None):
        """Terminate Streamlit and exit."""
        nonlocal streamlit_process
        print("[CRITICAL] Stopping all processes...")
        if streamlit_process and streamlit_process.poll() is None:
            print("[STREAMLIT] Terminating Streamlit...")
            streamlit_process.terminate()
            streamlit_process.wait()
        if error_msg:
            print(f"[ERROR] {error_msg}")
        sys.exit(1)

    def already_processed(file: str, mtime: float) -> bool:
        """Check control_file to avoid duplicate processing."""
        if not os.path.exists(control_file):
            return False
        with open(control_file, "r") as f:
            for line in f:
                try:
                    fname, ts = line.strip().split(",")
                except ValueError:
                    continue
                if fname == file and ts == str(mtime):
                    return True
        return False

    def execute_pipeline(file: str):
        """Trigger run_pipeline for the given file if not processed yet."""
        file_path = os.path.join(data_dir, file)
        try:
            mtime = os.path.getmtime(file_path)
        except FileNotFoundError:
            print(f"[WARN] File vanished before processing: {file}")
            return

        if already_processed(file, mtime):
            print(f"[CONTROL] File {file} already processed, skipping.")
            return

        print(f"[PIPELINE] Executing pipeline for file {file}...")
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
                target_file=file if target_file is None else target_file,
                window_size=window_size
            )
            print(f"[PIPELINE] Pipeline finished for {file}")
        except Exception as e:
            stop_all(f"Pipeline failed for {file}: {e}")

    class DataFileHandler(FileSystemEventHandler):
        """Watchdog handler reacting to creates/updates on supported files."""
        def on_created(self, event):
            self._process_event(event, "created")

        def on_modified(self, event):
            self._process_event(event, "modified")

        def _process_event(self, event, event_type: str):
            if event.is_directory:
                return
            fname = os.path.basename(event.src_path)
            if not fname.lower().endswith((
                ".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx"
            )):
                print(f"[INFO] Ignored file: {fname}")
                return
            print(f"[WATCHDOG] File {event_type}: {fname} → executing pipeline")
            execute_pipeline(fname)

    def is_port_in_use(p: int) -> bool:
        """Check if TCP port is already in use on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    def start_streamlit(pipeline_name: str, port: int | None = None):
        """Launch Streamlit dashboard on the chosen port."""
        nonlocal streamlit_process
        dashboard_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "web_interface", "dashboard.py")
        )

        if port is None:
            port = 8501
            if is_port_in_use(port):
                print(f"[STREAMLIT] Port {port} occupied. Trying port 8510...")
                port = 8510

        print(f"[STREAMLIT] Launching Streamlit dashboard on port {port}...")
        try:
            # Bind to 0.0.0.0 so it also works inside Docker
            streamlit_process = subprocess.Popen([
                "streamlit", "run", dashboard_path,
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--", "--pipeline_name", pipeline_name
            ])
        except Exception as e:
            stop_all(f"Failed to start Streamlit: {e}")

    def start_watchdog():
        """Start Watchdog observer loop; also monitor Streamlit process health."""
        print(f"[MONITOR] Watching directory: {data_dir}...")
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
    print("[MAIN] Launching monitor with Watchdog + Streamlit...")
    print("[CHECK] Verifying files in the directory at startup...")
    for file in os.listdir(data_dir):
        if file.lower().endswith((".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")):
            execute_pipeline(file)

    threading.Thread(target=start_streamlit, args=(pipeline_name, port), daemon=True).start()
    start_watchdog()


# =========================
# Utility functions
# =========================

def list_pipelines(base_dir="pipelines"):
    """List available pipelines under base_dir."""
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
    """Delete a pipeline directory and its contents (irreversible)."""
    pipeline_path = os.path.join(base_dir, pipeline_name)
    if not os.path.exists(pipeline_path):
        print(f"[ERROR] Pipeline {pipeline_name} does not exist.")
        return
    try:
        confirmation = input(
            f"Are you sure you want to delete the pipeline '{pipeline_name}'? This action is irreversible. (y/n): "
        )
        if confirmation.lower() != 'y':
            print("Deletion cancelled.")
            return
        shutil.rmtree(pipeline_path)
        print(f"[INFO] Pipeline '{pipeline_name}' has been deleted.")
    except Exception as e:
        print(f"[ERROR] Failed to delete pipeline {pipeline_name}: {e}")


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Minimal example: adapt to your environment


    start_monitor(
        pipeline_name="my_pipeline_watchdog",
        data_dir="/home/alex/datos",
        preprocess_file="/home/alex/calmops/pipeline/preprocessing.py",
        thresholds_perf={"balanced_accuracy": 0.8},
        thresholds_drift={"balanced_accuracy": 0.8},
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
        window_size=1000,
        port=None,          # e.g., 8600 if you need a specific port
        persistence="none"  # "none" | "pm2" | "docker"
    )
