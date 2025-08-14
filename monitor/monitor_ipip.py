# monitor/monitor_ipip.py
# -*- coding: utf-8 -*-
"""
CalmOps Monitor (IPIP)
- Vigila un directorio de datos con Watchdog (event-driven) y dispara la pipeline IPIP por bloques.
- Lanza un dashboard Streamlit para monitorización.
- Modos de persistencia opcionales:
    * "none"   : ejecución en primer plano (comportamiento por defecto)
    * "pm2"    : genera un runner y arranca con PM2 (auto-restart + on-boot)
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
import importlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Importa el runner IPIP por bloques
from pipeline.modules.pipeline_block_ipip import run_pipeline


# =========================
# Helpers
# =========================

def _project_root() -> str:
    """Ruta absoluta a la raíz del proyecto (asumiendo este archivo en monitor/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _which(cmd: str):
    """shutil.which shorthand."""
    from shutil import which
    return which(cmd)


def _fatal(msg: str, code: int = 1):
    """Imprime un mensaje fatal y sale."""
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)


def _model_spec_from_instance(model_instance):
    """
    Serializa clase + params de un modelo sklearn-like para reconstruir en el runner.
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
    """Guarda el JSON de configuración para el runner."""
    cfg_dir = os.path.join(base_dir, "pipelines", pipeline_name, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "runner_ipip_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_obj, f, indent=2)
    return cfg_path


def _write_runner_script(pipeline_name: str, runner_cfg_path: str, base_dir: str) -> str:
    """
    Genera un runner que reconstruye el modelo y llama a start_monitor_ipip
    con persistence='none' para evitar bucles con PM2/Docker.
    """
    pipeline_dir = os.path.join(base_dir, "pipelines", pipeline_name)
    os.makedirs(pipeline_dir, exist_ok=True)
    runner_path = os.path.join(pipeline_dir, f"run_{pipeline_name}_ipip.py")

    content = f'''# Auto-generated runner (IPIP) for pipeline: {pipeline_name}
import os, sys, json, importlib

# Ensure project root on sys.path (two levels up from this file)
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
        block_col=cfg.get("block_col"),
        ipip_config=cfg.get("ipip_config"),
        port=cfg.get("port"),
        persistence="none"  # avoid recursive PM2/Docker spawning
    )
'''
    with open(runner_path, "w") as f:
        f.write(content)
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
        print(f"[ERROR] No se pudo obtener la lista de PM2: {e}")
        return []
    filtered = [p for p in procs if p.get("name", "").startswith(prefix)]
    if not filtered:
        print(f"No hay procesos PM2 con prefijo '{prefix}'.")
        return []
    print("Procesos PM2:")
    for p in filtered:
        name = p.get("name"); pid = p.get("pid")
        status = p.get("pm2_env", {}).get("status")
        restart = p.get("pm2_env", {}).get("restart_time")
        print(f"- {name} | pid={pid} | status={status} | restarts={restart}")
    return filtered


def pm2_delete_pipeline(pipeline_name: str, base_dir: str = "pipelines") -> None:
    """Detiene y elimina proceso PM2 y borra la carpeta de la pipeline."""
    app_name = f"calmops-{pipeline_name}-ipip"
    if not _which("pm2"):
        print("[ERROR] PM2 no está instalado.")
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


def _pm2_install_hint() -> str:
    return (
        "PM2 requerido pero no encontrado.\n"
        "Instala Node.js + PM2:\n"
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


def docker_list(all_containers: bool = True):
    """Equivalente a `docker ps`."""
    if not _docker_available():
        print("[ERROR] Docker no está instalado.")
        return
    cmd = ["docker", "ps", "-a"] if all_containers else ["docker", "ps"]
    subprocess.call(cmd)


def docker_logs(container_name: str, tail: int = 200, follow: bool = True):
    """Muestra logs de un contenedor."""
    if not _docker_available():
        print("[ERROR] Docker no está instalado.")
        return
    cmd = ["docker", "logs", f"--tail={tail}"]
    if follow:
        cmd.append("-f")
    cmd.append(container_name)
    subprocess.call(cmd)


def docker_stop(container_name: str, remove: bool = False):
    """Para un contenedor (y opcionalmente lo elimina)."""
    if not _docker_available():
        print("[ERROR] Docker no está instalado.")
        return
    subprocess.call(["docker", "stop", container_name])
    if remove:
        subprocess.call(["docker", "rm", "-f", container_name])


def _write_docker_files(pipeline_name: str, runner_script_abs: str, base_dir: str, port: int | None):
    """
    Genera Dockerfile y docker-compose.yml en pipelines/<pipeline_name>/.
    El contexto de build es la raíz del repo.
    """
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


def _launch_with_pm2(pipeline_name: str, runner_script: str, base_dir: str):
    """Arranca el runner con PM2 y lo persiste."""
    if not _which("pm2"):
        _fatal(_pm2_install_hint())

    eco_path = os.path.join(base_dir, "pipelines", pipeline_name, "ecosystem.ipip.config.js")
    app_name = f"calmops-{pipeline_name}-ipip"

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

    try:
        subprocess.check_call(["pm2", "save"])
    except Exception:
        pass
    try:
        subprocess.check_call(["pm2", "startup", "-u", os.getenv("USER", "") or ""])
    except Exception:
        pass

    print(f"[PM2] App '{app_name}' started and saved.")


def _launch_with_docker(pipeline_name: str, runner_script: str, base_dir: str, port: int | None):
    """Construye y arranca docker-compose en modo detach."""
    if not _which("docker"):
        _fatal(
            "Docker requerido pero no encontrado.\n"
            "Instala Docker Engine y docker compose plugin."
        )
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
        _fatal(f"Fallo al lanzar Docker services: {e}")


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
    # IPIP / bloques:
    block_col: str | None = None,
    ipip_config: dict | None = None,
):
    """
    - Vigila el directorio de datos y ejecuta run_pipeline (IPIP) cuando hay cambios.
    - Lanza Streamlit dashboard.
    - Opcionalmente crea runner para PM2/Docker.
    """

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

    # Config para dashboard
    config_path = os.path.join(CONFIG_DIR, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump({
                "pipeline_name": pipeline_name,
                "data_dir": data_dir,
                "preprocess_file": preprocess_file,
                "block_col": block_col,
                "mode": "ipip"
            }, f)
    except Exception as e:
        _fatal(f"Failed to save config.json: {e}")

    streamlit_process = None

    def stop_all(error_msg=None):
        """Termina Streamlit y sale."""
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
        """Evita reprocesar el mismo fichero/mtime."""
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
        """Dispara la pipeline para un fichero si no está procesado aún."""
        file_path = os.path.join(data_dir, file)
        try:
            mtime = os.path.getmtime(file_path)
        except FileNotFoundError:
            print(f"[WARN] File vanished before processing: {file}")
            return

        if already_processed(file, mtime):
            print(f"[CONTROL] File {file} already processed, skipping.")
            return

        print(f"[PIPELINE] Executing pipeline (IPIP) for file {file}...")
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
                window_size=window_size,
                block_col=block_col,
                ipip_config=ipip_config
            )
            print(f"[PIPELINE] Pipeline finished for {file}")
        except Exception as e:
            stop_all(f"Pipeline failed for {file}: {e}")

    class DataFileHandler(FileSystemEventHandler):
        """Watchdog handler: crea/modifica -> pipeline."""
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
        """Comprueba si un puerto TCP está en uso."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", p)) == 0

    def start_streamlit(pipeline_name: str, port: int | None = None):
        """Arranca Streamlit dashboard en el puerto escogido."""
        nonlocal streamlit_process
        dashboard_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "web_interface", "dashboard_ipip.py")
        )

        if port is None:
            port = 8501
            if is_port_in_use(port):
                print(f"[STREAMLIT] Port {port} occupied. Trying port 8510...")
                port = 8510

        print(f"[STREAMLIT] Launching Streamlit dashboard on port {port}...")
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
        """Bucle del observer; vigila también la salud del proceso Streamlit."""
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

    # --- Lanzamiento inicial ---
    print("[MAIN] Launching IPIP monitor with Watchdog + Streamlit...")
    print("[CHECK] Verifying files in the directory at startup...")
    for file in os.listdir(data_dir):
        if file.lower().endswith((".arff", ".csv", ".txt", ".xml", ".json", ".parquet", ".xls", ".xlsx")):
            execute_pipeline(file)

    threading.Thread(target=start_streamlit, args=(pipeline_name, port), daemon=True).start()
    start_watchdog()


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Ejemplo mínimo; ajusta a tu entorno real
    from sklearn.ensemble import RandomForestClassifier

    start_monitor_ipip(
        pipeline_name="my_pipeline_ipip",
        data_dir="/path/to/datos",
        preprocess_file="/path/to/preprocessing_ipip.py",
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
        block_col="mes",                 # por ejemplo, si tu bloque es 'mes'
        ipip_config={"p": 20, "b": 5},  # parámetros propios del ensamblado IPIP (opcional)
        port=None,
        persistence="none"               # "none" | "pm2" | "docker"
    )
