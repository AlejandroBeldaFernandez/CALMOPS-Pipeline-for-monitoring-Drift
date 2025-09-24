import os
import sys
import json
import webbrowser
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
from collections import defaultdict

from .SyntheticReporter import SyntheticReporter  # usamos reporter para guardar JSON


# Defaults
DEFAULT_OUTPUT_DIR = "salida_tiempo_real"
DASHBOARD_REL_PATH = os.path.join("Dashboards", "dashboard_synthetic.py")
DEFAULT_PORT = 8052


class SyntheticGenerator:
    def __init__(self, port: Optional[int] = None,
                 enable_report: bool = False,     # por defecto desactivado
                 launch_viewer: bool = False):    # por defecto desactivado
        """
        Generador de datos sintéticos (sin AutoVisualizer).
        Con flags para (opcionalmente) guardar reporte JSON y lanzar un viewer Dash.

        Args:
            port: Puerto opcional para el viewer (si se usa).
            enable_report: si True, escribe report.json.
            launch_viewer: si True, lanza el dashboard de solo lectura.
        """
        self.port = port
        self.enable_report = enable_report
        self.launch_viewer = launch_viewer

    # -------------------
    # Interfaz pública
    # -------------------
    def generate(self,
                 output_path: Optional[str],
                 filename: str,
                 n_samples: int,
                 method: str = "sea",
                 method_params: dict = None,
                 drift_type: str = "none",
                 position_of_drift: int = None,
                 target_col: str = "target",
                 balance: bool = False,
                 random_state: int = None,
                 # fechas opcionales
                 date_start: Optional[str] = None,
                 date_every: int = 1,
                 date_step: Optional[Dict[str, int]] = None,
                 date_col: str = "timestamp"):
        """
        Genera datos sintéticos con River vía GeneratorFactory.

        - drift_type: "none" | "concept" | "data" | "both"
        - position_of_drift: índice donde ocurre el drift (necesario si drift != "none")
        - balance: si True, intenta balancear por clases (streaming básico)
        - date_*: controla inyección de una columna temporal "por tramos"
        """
        from .GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig

        # Resolver output
        out_dir = self._resolve_output_dir(output_path)

        # Mapear método
        method_mapping = {
            "sea": GeneratorType.SEA,
            "agrawal": GeneratorType.AGRAWAL,
            "hyperplane": GeneratorType.HYPERPLANE,
            "sine": GeneratorType.SINE,
            "stagger": GeneratorType.STAGGER,
            "random_tree": GeneratorType.RANDOM_TREE,
            "mixed": GeneratorType.MIXED,
            "friedman": GeneratorType.FRIEDMAN,
            "random_rbf": GeneratorType.RANDOM_RBF
        }
        if method not in method_mapping:
            raise ValueError(f"Invalid method '{method}'. Choose one of {list(method_mapping.keys())}")

        # Config base
        method_params = (method_params or {}).copy()
        if random_state is not None:
            method_params['random_state'] = random_state

        config = GeneratorConfig(**method_params)
        factory = GeneratorFactory()
        generator_instance = factory.create_generator(method_mapping[method], config)

        # Generador para concept drift (si aplica)
        generator_instance_drift = None
        if drift_type != "none" and position_of_drift is not None:
            drift_params = method_params.copy()
            # Cambia la función/clasificador para producir cambio de concepto
            if method == "sea":
                drift_params['function'] = drift_params.get('function', 0) + 1
            elif method == "agrawal":
                drift_params['classification_function'] = drift_params.get('classification_function', 0) + 1
            # nueva semilla para el drift (si procede)
            if random_state is not None:
                drift_params['random_state'] = random_state + 1
            drift_config = GeneratorConfig(**drift_params)
            generator_instance_drift = factory.create_generator(method_mapping[method], drift_config)

        # Ejecutar interno
        return self._generate_internal(
            generator_instance=generator_instance,
            output_path=out_dir,
            filename=filename,
            n_samples=n_samples,
            generator_instance_drift=generator_instance_drift,
            position_of_drift=position_of_drift,
            ratio_before=None,
            ratio_after=None,
            target_col=target_col,
            balance=balance,
            drift_type=drift_type,
            extra_info=None,
            date_start=date_start,
            date_every=date_every,
            date_step=date_step,
            date_col=date_col
        )

    # -------------------
    # Núcleo de generación
    # -------------------
    def _generate_internal(self,
                           generator_instance,
                           output_path: str,
                           filename: str,
                           n_samples: int,
                           generator_instance_drift=None,
                           position_of_drift: int = None,
                           ratio_before: dict = None,
                           ratio_after: dict = None,
                           target_col: str = "target",
                           balance: bool = False,
                           drift_type: str = "none",
                           extra_info: dict = None,
                           # fechas
                           date_start: Optional[str] = None,
                           date_every: int = 1,
                           date_step: Optional[Dict[str, int]] = None,
                           date_col: str = "timestamp"):

        # Validaciones previas (incluye drift params)
        self.validate_params(
            generator_instance=generator_instance,
            output_path=output_path,
            filename=filename,
            n_samples=n_samples,
            generator_instance_drift=generator_instance_drift,
            position_of_drift=position_of_drift,
            ratio_before=ratio_before,
            ratio_after=ratio_after,
            target_col=target_col,
            balance=balance,
            drift_type=drift_type
        )

        # Generar datos según tipo de drift
        if drift_type == "none":
            data = (self._generate_balanced(generator_instance, n_samples)
                    if balance else self._generate_data(generator_instance, n_samples))
        elif drift_type == "concept":
            data = self._generate_concept_drift(generator_instance, generator_instance_drift,
                                                n_samples, position_of_drift)
        elif drift_type == "data":
            data = self._generate_data_drift(generator_instance, n_samples, position_of_drift,
                                             ratio_before, ratio_after)
        elif drift_type == "both":
            data = self._generate_both_drift(generator_instance, generator_instance_drift,
                                             n_samples, position_of_drift, ratio_before, ratio_after)
        else:
            raise ValueError(f"Unsupported drift_type '{drift_type}'")

        # Columnas (x... + target)
        first_sample = next(iter(generator_instance.take(1)))
        columns = list(first_sample[0].keys()) + [target_col]
        df = pd.DataFrame(data, columns=columns)

        # Inyección de fechas (opcional)
        df = self._inject_dates(df, date_col=date_col, date_start=date_start,
                                date_every=date_every, date_step=date_step)

        # Guardar CSV
        full_path = os.path.join(output_path, filename)
        df.to_csv(full_path, index=False)
        print(f"[SyntheticGenerator] Data generated and saved at: {full_path}")

        # Reporte JSON (opcional)
        if self.enable_report:
            is_block_dataset = any(col in df.columns for col in ['block', 'chunk', 'Block', 'Chunk'])
            self._save_report_json(
                df=df,
                target_col=target_col,
                drift_type=drift_type,
                position_of_drift=position_of_drift,
                is_block_dataset=is_block_dataset,
                extra_info=extra_info or {},
                output_dir=output_path
            )

        # Lanzar viewer (opcional)
        if self.launch_viewer:
            self._launch_dashboard_viewer(port=self.port)

        return full_path

    # -------------------
    # Helpers de generación
    # -------------------
    def _generate_balanced(self, generator_instance, n_samples: int):
        """
        Toma más muestras de las necesarias (factor 10) y recorta por clase
        para aproximar balance (útil cuando el generador no es nativamente balanceado).
        """
        class_samples = defaultdict(list)
        for x, y in generator_instance.take(n_samples * 10):
            row = list(x.values()) + [y]
            class_samples[y].append(row)
            # condición de parada temprana
            if len(class_samples) >= 1:
                try:
                    min_class_samples = min(len(samples) for samples in class_samples.values())
                    if min_class_samples >= n_samples // max(1, len(class_samples)):
                        break
                except ValueError:
                    pass

        data = []
        n_classes = max(1, len(class_samples))
        per_class = n_samples // n_classes
        for _, samples in class_samples.items():
            data.extend(samples[:per_class])
        # si faltan por división entera, completa con el remanente disponible
        if len(data) < n_samples:
            for _, samples in class_samples.items():
                extra = min(len(samples) - per_class, n_samples - len(data))
                if extra > 0:
                    data.extend(samples[per_class:per_class + extra])
                if len(data) >= n_samples:
                    break
        return data[:n_samples]

    def _generate_data(self, generator_instance, n_samples: int):
        return [list(x.values()) + [y] for x, y in generator_instance.take(n_samples)]

    def _generate_concept_drift(self, generator_instance, generator_instance_drift,
                                n_samples: int, position_of_drift: int):
        if generator_instance_drift is None:
            raise ValueError("generator_instance_drift must be provided for concept drift")
        data = [list(x.values()) + [y] for x, y in generator_instance.take(position_of_drift)]
        data += [list(x.values()) + [y] for x, y in generator_instance_drift.take(n_samples - position_of_drift)]
        return data

    def _generate_data_drift(self, generator_instance, n_samples: int,
                             position_of_drift: int, ratio_before: dict, ratio_after: dict):
        if ratio_before is None or ratio_after is None:
            raise ValueError("ratio_before/ratio_after must be provided for data drift")
        data = []
        data += self._generate_with_ratios(generator_instance, position_of_drift, ratio_before)
        data += self._generate_with_ratios(generator_instance, n_samples - position_of_drift, ratio_after)
        return data

    def _generate_both_drift(self, generator_instance, generator_instance_drift,
                             n_samples: int, position_of_drift: int,
                             ratio_before: dict, ratio_after: dict):
        if generator_instance_drift is None:
            raise ValueError("generator_instance_drift must be provided for 'both' drift")
        if ratio_before is None or ratio_after is None:
            raise ValueError("ratio_before/ratio_after must be provided for 'both' drift")
        data = []
        data += self._generate_with_ratios(generator_instance, position_of_drift, ratio_before)
        data += self._generate_with_ratios(generator_instance_drift, n_samples - position_of_drift, ratio_after)
        return data

    def _generate_with_ratios(self, generator_instance, n_samples: int, target_ratios: Dict):
        """
        Extrae n_samples intentando respetar proporciones target_ratios (e.g. {0:0.3, 1:0.7}).
        """
        target_counts = {cls: int(n_samples * ratio) for cls, ratio in target_ratios.items()}
        total_assigned = sum(target_counts.values())
        if total_assigned != n_samples:
            # ajusta el mayor para cuadrar
            max_class = max(target_counts, key=lambda x: target_counts[x])
            target_counts[max_class] += (n_samples - total_assigned)

        class_samples = defaultdict(list)
        for x, y in generator_instance.take(n_samples * 10):
            if y in target_counts and len(class_samples[y]) < target_counts[y]:
                class_samples[y].append(list(x.values()) + [y])
            if all(len(class_samples[c]) >= target_counts[c] for c in target_counts.keys()):
                break

        final_data = []
        for cls, target_count in target_counts.items():
            final_data.extend(class_samples[cls][:target_count])
        # por seguridad recorta
        return final_data[:n_samples]

    # -------------------
    # Inyección de fechas
    # -------------------
    def _inject_dates(self, df: pd.DataFrame,
                      date_col: str,
                      date_start: Optional[str],
                      date_every: int,
                      date_step: Optional[Dict[str, int]]) -> pd.DataFrame:
        """
        Agrega una columna fecha/fecha-hora que aumenta cada `date_every` filas
        con un offset calendar-safe (pandas.DateOffset) según `date_step`.
        """
        if date_start is None:
            return df

        if not isinstance(date_every, int) or date_every <= 0:
            raise ValueError(f"date_every must be a positive integer, got {date_every}")

        step = date_step or {'days': 1}
        valid_keys = {'years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds',
                      'microseconds', 'nanoseconds'}
        invalid = set(step.keys()) - valid_keys
        if invalid:
            raise ValueError(f"Invalid date_step keys: {invalid}. Allowed: {sorted(valid_keys)}")

        try:
            start_ts = pd.to_datetime(date_start)
        except Exception as e:
            raise ValueError(f"Invalid date_start '{date_start}': {e}")

        total = len(df)
        periods = (total + date_every - 1) // date_every
        anchors = []
        current = start_ts
        offset = pd.DateOffset(**step)
        for _ in range(periods):
            anchors.append(current)
            current = current + offset

        series = pd.Series(anchors).repeat(date_every).iloc[:total].reset_index(drop=True)

        try:
            df.insert(0, date_col, series)
        except ValueError:
            df[date_col] = series

        print(f"[SyntheticGenerator] Injected date column '{date_col}' starting at {start_ts} "
              f"every {date_every} rows with step {step}")
        return df

    # -------------------
    # Report JSON helper
    # -------------------
    def _save_report_json(self, df: pd.DataFrame, target_col: str, drift_type: str,
                          position_of_drift, is_block_dataset: bool,
                          extra_info: dict, output_dir: str) -> Path:
        """
        Genera report.json usando SyntheticReporter si está disponible; si no, genera uno mínimo.
        """
        os.makedirs(output_dir, exist_ok=True)
        report_path = Path(output_dir) / "report.json"

        try:
            reporter = SyntheticReporter()
            result = reporter.generate_report(
                df=df,
                target_col=target_col,
                drift_type=drift_type,
                position_of_drift=position_of_drift,
                is_block_dataset=is_block_dataset,
                extra_info=extra_info or {}
            )
            if isinstance(result, dict):
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                return report_path
        except Exception as e:
            print(f"[SyntheticGenerator] SyntheticReporter fallback (reason: {e})")

        # Fallback mínimo
        try:
            n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
            missing = int(df.isna().sum().sum())
            total_cells = max(1, n_rows * n_cols)
            data_completeness = max(0.0, 100.0 - (missing / total_cells * 100.0))

            class_balance = None
            class_dist = None
            if target_col in df.columns:
                vc = df[target_col].value_counts(normalize=True)
                class_dist = {str(k): float(v) for k, v in vc.items()}
                if len(vc) >= 2:
                    import numpy as np
                    p = vc.to_numpy()
                    u = np.ones_like(p) / len(p)
                    l1 = float(np.abs(p - u).sum())
                    class_balance = max(0.0, 100.0 * (1.0 - l1 / 2.0))
                else:
                    class_balance = 50.0

            report = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "rows": n_rows,
                    "cols": n_cols,
                    "target_col": target_col,
                    "is_block_dataset": bool(is_block_dataset),
                    "drift_type": drift_type,
                    "position_of_drift": position_of_drift
                },
                "quality": {
                    "data_completeness": data_completeness,
                    "class_balance": class_balance,
                    "class_distribution": class_dist
                },
                "extra_info": extra_info or {}
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SyntheticGenerator] Could not write report.json: {e}")

        return report_path

    # -------------------
    # Validaciones
    # -------------------
    def validate_params(self,
                        generator_instance,
                        output_path: str,
                        filename: str,
                        n_samples: int,
                        generator_instance_drift=None,
                        position_of_drift: int = None,
                        ratio_before: dict = None,
                        ratio_after: dict = None,
                        target_col: str = "target",
                        balance: bool = False,
                        drift_type: str = "none"):
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")

        valid_drift_types = ["none", "concept", "data", "both"]
        if drift_type not in valid_drift_types:
            raise ValueError(f"Invalid drift_type '{drift_type}'. Must be one of {valid_drift_types}")

        if drift_type in ["concept", "data", "both"]:
            if position_of_drift is None or not (0 < position_of_drift < n_samples):
                raise ValueError(f"position_of_drift must be between 0 and n_samples ({n_samples}) when drift is applied")

        if not isinstance(balance, bool):
            raise ValueError(f"balance must be a boolean, got {type(balance)}")

        self._validate_drift_params(drift_type, generator_instance_drift, ratio_before, ratio_after)

        if generator_instance is None:
            raise ValueError("generator_instance must be provided")

        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError("output_path must resolve to a non-empty string")
        os.makedirs(output_path, exist_ok=True)

        return True

    def _validate_drift_params(self, drift_type, generator_instance_drift, ratio_before, ratio_after):
        if drift_type in ["concept", "both"] and generator_instance_drift is None:
            raise ValueError("For concept and both drift types, generator_instance_drift must be provided.")
        if drift_type == "data" and (ratio_before is None or ratio_after is None):
            raise ValueError("For data drift, both ratio_before and ratio_after must be provided.")
        if drift_type in ["data", "both"]:
            if abs(sum(ratio_before.values()) - 1.0) > 1e-6:
                raise ValueError(f"ratio_before must sum to 1.0, got {sum(ratio_before.values())}")
            if abs(sum(ratio_after.values()) - 1.0) > 1e-6:
                raise ValueError(f"ratio_after must sum to 1.0, got {sum(ratio_after.values())}")

    # -------------------
    # Lanzador del dashboard (opcional)
    # -------------------
    def _launch_dashboard_viewer(self, port: Optional[int] = None):
        """
        Lanza Dashboards/dashboard_synthetic.py sin pasar rutas.
        El dashboard lee por su cuenta (p. ej., 'salida_tiempo_real/').
        """
        here = os.path.dirname(os.path.abspath(__file__))   # .../Synthetic
        project_root = os.path.dirname(here)                # .../data-generators
        script_path = os.path.join(project_root, "Dashboards", "dashboard_synthetic.py")
        script_path = os.path.abspath(script_path)

        if not os.path.exists(script_path):
            print(f"[SyntheticGenerator] Dashboard script not found at '{script_path}'. Skipping viewer launch.")
            return

        if port is None:
            port = 8061

        cmd = [sys.executable, script_path, "--port", str(int(port))]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", ""))

        print(f"[SyntheticGenerator] Launching dashboard viewer:\n  {' '.join(cmd)}\n  cwd={project_root}")
        try:
            subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env
            )
            print(f"[SyntheticGenerator] Viewer started. URL: http://localhost:{port}")
        except Exception as e:
            print(f"[SyntheticGenerator] Could not launch viewer: {e}")

    # -------------------
    # Resolución de paths
    # -------------------
    def _resolve_output_dir(self, output_path: Optional[str]) -> str:
        """
        Si output_path es None o vacío, usa DEFAULT_OUTPUT_DIR.
        Asegura que exista y devuelve ruta absoluta.
        """
        out = (output_path or "").strip() or DEFAULT_OUTPUT_DIR
        out_abs = os.path.abspath(out)
        os.makedirs(out_abs, exist_ok=True)
        return out_abs
