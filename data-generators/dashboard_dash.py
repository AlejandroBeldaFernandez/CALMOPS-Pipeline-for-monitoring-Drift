#!/usr/bin/env python3
"""
CALMOPS - Dashboard Principal con Dash integrado con Synthetic
=============================================================

Dashboard principal que usa la infraestructura real de Synthetic/GeneratorFactory
para generar datos con River, manteniendo las visualizaciones del dashboard.

Author: CalmOps Team  
Version: 6.1 - Synthetic Integration + Auto Mode
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash.exceptions
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from dash import no_update
from typing import Optional, Dict
import time
# Import Synthetic generators
from Synthetic.SyntheticGenerator import SyntheticGenerator
from Synthetic.SyntheticBlockGenerator import SyntheticBlockGenerator
from Synthetic.GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig
import json  # ya lo tienes, pero por si acaso
import os

# Dónde buscar el estado escrito por el generador (o por cualquier servicio)
STREAM_STATUS_DIR = "salida_tiempo_real"
STREAM_STATUS_FILE = os.path.join(STREAM_STATUS_DIR, "stream_status.json")

def read_stream_status(fallback_running: bool) -> dict:
    """
    Lee stream_status.json si existe. Si no, usa fallback_running (p. ej. auto_mode_active).
    """
    status = {
        "status": "running" if fallback_running else "stopped",
        "updated_at": None,
        "source": "fallback"
    }
    try:
        if os.path.exists(STREAM_STATUS_FILE):
            with open(STREAM_STATUS_FILE, "r", encoding="utf-8") as f:
                j = json.load(f)
            if isinstance(j, dict) and "status" in j:
                status.update(j)
                status["source"] = "file"
    except Exception:
        pass
    return status

def render_stream_status_badge(running: bool, updated_at: str | None = None) -> html.Div:
    color_bg = "#0fbf61" if running else "#cc0000"
    color_glow = "rgba(15,191,97,0.45)" if running else "rgba(204,0,0,0.45)"
    label = "EN VIVO" if running else "DETENIDO"
    sub = f"• actualizado {updated_at}" if updated_at else ""
    return html.Div([
        html.Span("●", style={"marginRight": "8px"}),
        html.Span(label, style={"fontWeight": 700, "letterSpacing": "1px"}),
        html.Span(f"  {sub}", style={"marginLeft": "8px", "opacity": 0.8})
    ], style={
        "display": "inline-flex", "alignItems": "center",
        "padding": "10px 14px", "borderRadius": "999px",
        "background": color_bg, "color": "#fff",
        "boxShadow": f"0 0 22px {color_glow}", "fontSize": "12px"
    })


# Configuración de tema oscuro para Plotly
DARK_THEME = {
    'plot_bgcolor': '#1e1e1e',
    'paper_bgcolor': '#1e1e1e', 
    'font_color': '#ffffff',
    'grid_color': '#404040',
    'title_color': '#00d4ff'
}

# Paleta de colores profesional
COLOR_PALETTE = {
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'info': '#74c0fc',
    'accent': '#9775fa',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

class DashDataGenerator:
    """Fixed configuration generator: Agrawal, balanced, no noise (using print)."""

    def __init__(self):
        # Hard limits / defaults
        self.max_samples = 100000
        self.current_seed = 42

        # Fixed generator config
        self.last_generator_type = 'agrawal'  # always agrawal
        self.last_n_features = 9              # agrawal has 9 features
        self.last_noise = 0.0                 # no noise
        self.balance_classes = True           # always balanced

        self.current_data = pd.DataFrame()
        self.dataset_info = {}
        self.temp_dir = './plots'
        os.makedirs(self.temp_dir, exist_ok=True)

        # Auto mode (streaming)
        self.auto_mode_active = False
        self.auto_generator = None
        self.auto_batch_size = 100

        # Single available type (kept for info display)
        self.generator_types = {'agrawal': 'Agrawal (9 features)'}
    
        # --- Concept drift (Agrawal): cambiar cf cada 400 instancias ---
        self.cf_cycle = list(range(10))   # 10 funciones clásicas de Agrawal: 0..9
        self.active_cf_idx = 0            # índice dentro de cf_cycle
        self.drift_span = 400             # cada 400 instancias cambiamos de cf
        self.just_restarted = False       # flag para reinicio de dashboard
        
        # --- Target (label) drift ---
        self.target_drift_span = 1000     # cada 1000 instancias
        self.target_drift_level = 0       # 0=sin drift; 1,2,... = niveles sucesivos

    def _prepare_method_params(self, _generator_type, _n_features, _noise, _class_sep):
        """Return params for Agrawal only (balanced, no noise)."""
        cf = self.cf_cycle[self.active_cf_idx] if _generator_type == 'agrawal' else 0
        return {
            'classification_function': cf,
            'balance_classes': True,
            'perturbation': 0.0
        }
        
    def set_seed_and_soft_reset(self, new_seed: int, cf_idx: int = 0):
        """Cambia la semilla y reinicia buffers/plots sin parar el stream."""
        try:
            new_seed = int(new_seed)
        except Exception:
            new_seed = self.current_seed or 42

        self.current_seed = new_seed
        self.target_drift_level = 0

        # Reconfigura el generador respetando tipo/features/noise
        self._setup_real_time_generator(
            self.last_generator_type,
            self.last_n_features,
            self.last_noise,
            new_seed,
            cf_idx=cf_idx  # patrón base
        )

        # Reset visual/buffer
        self.current_data = pd.DataFrame()
        self.dataset_info["samples"] = 0
        self.dataset_info["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Siembra un primer batch para que el dashboard pinte algo al instante
        try:
            x1, y1 = next(self.auto_generator.take(1))
            feature_cols2 = [str(c) for c in x1.keys()]
            cols2 = feature_cols2 + ["target"]

            rows2 = [list(x1.values()) + [y1]]
            for x, y in self.auto_generator.take(self.auto_batch_size - 1):
                rows2.append(list(x.values()) + [y])

            if rows2:
                df_batch2 = pd.DataFrame(rows2, columns=cols2)
                df_batch2["sample_id"] = range(0, len(df_batch2))
                step = pd.Timedelta(days=1); group_size = 25
                sid2 = df_batch2["sample_id"].astype(int)
                base2 = pd.Timestamp.today().floor('D') - ((len(df_batch2) - 1) // group_size) * step
                df_batch2["timestamp"] = base2 + (sid2 // group_size) * step
                self.current_data = df_batch2
                self.dataset_info["samples"] = len(self.current_data)
        except StopIteration:
            pass


    def generate_standard_dataset(self, n_samples=500, generator_type='agrawal',
                                  n_features=9, class_sep=0.8, noise=0.0):
        """Generate fixed Agrawal dataset (balanced, no noise)."""
        random_state=  self.current_seed
        try:
            print(f"[INFO] Generating Agrawal dataset: n={n_samples}, seed={random_state}")
            syn_gen = SyntheticGenerator()
            method_params = self._prepare_method_params('agrawal', 9, 0.0, class_sep)

            filename = f"dataset_agrawal_{random_state}.csv"
            steps = max(0, n_samples // 25)                        # nº de saltos semanales
            end_dt = pd.Timestamp.today().floor('D')                # hoy
            start_dt = end_dt - pd.Timedelta(days=steps)           # arranca hace 'steps' semanas

            syn_gen.generate(
                output_path=self.temp_dir,
                filename=filename,
                n_samples=n_samples,
                method='agrawal',
                method_params=method_params,
                drift_type="none",
                random_state=random_state,
                date_start=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                date_every=25,                # ↩️ cambia cada 100 filas
                date_step={"days": 1},        # ↩️ +1 semana
                date_col="timestamp"
            )


            file_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                self.current_data = self._process_generated_data(df, 'agrawal')

                factory = GeneratorFactory()
                gen_info = factory.get_generator_info(GeneratorType.AGRAWAL)

                self.dataset_info = {
                    'type': 'Estándar',
                    'generator': self.generator_types['agrawal'],
                    'samples': len(self.current_data),
                    'features': len([c for c in self.current_data.columns if c != 'target']),
                    'classes': self.current_data['target'].nunique() if 'target' in self.current_data.columns else 0,
                    'description': gen_info.get('description', 'Agrawal stream'),
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return True
            else:
                print(f"[ERROR] File not found: {file_path}")
                return False
        except Exception as e:
            print(f"[EXCEPTION] Error generating dataset: {e}")
            return False

    def _rebalance_dataframe(self, df: pd.DataFrame, target_col: str = "target",
                             strategy: str = "downsample", random_state: int = 42) -> pd.DataFrame:
        """
        Keep batch balanced if needed. For Agrawal it's usually already balanced, but keep as safety.
        """
        if target_col not in df.columns:
            return df
        vc = df[target_col].value_counts(dropna=False)
        if len(vc) <= 1:
            return df
        min_count = int(vc.min())
        if min_count == 0:
            print("[WARN] One class has zero instances in the batch; skipping rebalance.")
            return df
        parts = []
        for cls, g in df.groupby(target_col):
            n = min(len(g), min_count) if strategy == "downsample" else int(vc.max())
            parts.append(g.sample(n=n, replace=(strategy == "upsample"), random_state=random_state))
        out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return out

    def generate_block_dataset(self, n_samples=500, n_blocks=5, generator_type='agrawal',
                               n_features=9, class_sep=0.8, noise=0.0):
        """Generate fixed Agrawal blocks (balanced, no noise)."""
        random_state= self.current_seed
        try:
            print(f"[INFO] Generating Agrawal blocks: n={n_samples}, blocks={n_blocks}, seed={random_state}")
            block_gen = SyntheticBlockGenerator()
            method_params = self._prepare_method_params('agrawal', 9, 0.0, class_sep)
            filename = f"dataset_blocks_agrawal_{random_state}.csv"
            methods = ['agrawal'] * n_blocks
            method_params_list = [method_params.copy() for _ in range(n_blocks)]

            block_gen.generate_blocks_simple(
                output_path=self.temp_dir,
                filename=filename,
                n_blocks=n_blocks,
                total_samples=n_samples,
                methods=methods,
                method_params=method_params_list,
                random_state=random_state,
                # ⬇️ nuevo
                date_start=datetime.now().strftime("%Y-%m-%d"),
                date_step={"months": 1},
                date_col="timestamp"
            )


            file_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                self.current_data = self._process_generated_data(df, 'agrawal', has_blocks=True)

                factory = GeneratorFactory()
                gen_info = factory.get_generator_info(GeneratorType.AGRAWAL)

                self.dataset_info = {
                    'type': 'Por Bloques',
                    'generator': self.generator_types['agrawal'],
                    'samples': len(self.current_data),
                    'features': len([c for c in self.current_data.columns if c not in ['target', 'block']]),
                    'classes': self.current_data['target'].nunique() if 'target' in self.current_data.columns else 0,
                    'blocks': n_blocks,
                    'description': gen_info.get('description', 'Agrawal stream'),
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return True
            else:
                print(f"[ERROR] File not found: {file_path}")
                return False
        except Exception as e:
            print(f"[EXCEPTION] Error generating blocks: {e}")
            return False

    
    

    
    def _process_generated_data(self, df, _generator_type, _has_blocks=False):
        processed_df = df.copy()

        # Parseo robusto del timestamp si existe
        if 'timestamp' in processed_df.columns:
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], errors='coerce')
        else:
            processed_df['timestamp'] = pd.date_range(start=datetime.now(), periods=len(processed_df), freq='1min')

        if 'sample_id' not in processed_df.columns:
            processed_df['sample_id'] = range(len(processed_df))
    
        column_mapping = {}
        feature_counter = 0
        for col in processed_df.columns:
            if col.startswith('x') and col[1:].isdigit():
                column_mapping[col] = f'feature_{feature_counter}'
                feature_counter += 1
            elif col == 'y':
                column_mapping[col] = 'target'
        if column_mapping:
            processed_df = processed_df.rename(columns=column_mapping)
    
        feature_cols = [col for col in processed_df.columns if col.startswith('feature_')]
        if len(feature_cols) > 0:
            if 'magnitude' not in processed_df.columns:
                feature_matrix = processed_df[feature_cols].values
                processed_df['magnitude'] = np.linalg.norm(feature_matrix, axis=1)
            if 'anomaly_score' not in processed_df.columns:
                processed_df['anomaly_score'] = np.random.beta(2, 5, len(processed_df))
            if 'confidence' not in processed_df.columns:
                processed_df['confidence'] = np.random.uniform(0.7, 1.0, len(processed_df))
        return processed_df
    
    def get_current_data(self):
        return self.current_data
    
    def get_dataset_info(self):
        return self.dataset_info
    
    def get_generator_types(self):
        return self.generator_types
    
    def _setup_real_time_generator(self, generator_type, n_features, noise, random_state, cf_idx: int | None = None):
        """Configura un generador River en modo streaming (auto mode)."""
        mapping = {
            "sea": GeneratorType.SEA, "agrawal": GeneratorType.AGRAWAL, "hyperplane": GeneratorType.HYPERPLANE,
            "sine": GeneratorType.SINE, "stagger": GeneratorType.STAGGER, "random_tree": GeneratorType.RANDOM_TREE,
            "mixed": GeneratorType.MIXED, "friedman": GeneratorType.FRIEDMAN, "random_rbf": GeneratorType.RANDOM_RBF
        }
        if generator_type not in mapping:
            raise ValueError(f"Generator {generator_type} no soportado en modo automático")

        self.last_generator_type = generator_type
        self.last_n_features = n_features
        self.last_noise = noise
        self.current_seed = random_state

        if generator_type == 'agrawal' and cf_idx is not None:
            self.active_cf_idx = int(cf_idx) % len(self.cf_cycle)
        elif generator_type != 'agrawal':
            self.active_cf_idx = 0  # irrelevante, pero dejamos algo consistente

        params = self._prepare_method_params(generator_type, n_features, noise, 0.8)
        params["random_state"] = random_state
        config = GeneratorConfig(**params)
        factory = GeneratorFactory()
        self.auto_generator = factory.create_generator(mapping[generator_type], config)

        self.auto_mode_active = True
        self.current_data = pd.DataFrame()
        self.dataset_info = {
            "type": "Streaming", "generator": self.generator_types.get(generator_type, generator_type),
            "samples": 0, "features": n_features, "description": f"River generator {generator_type} en auto mode",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f"✅ Auto generator configurado: {generator_type} (cf={params.get('classification_function', 'n/a')})")
        return True

    def _reconfigure_agrawal_cf(self, new_cf_idx: int):
        """
        Reconfigura SOLO la classification_function del generador Agrawal,
        manteniendo seed y resto de parámetros. No toca self.current_data.
        """
        if self.last_generator_type != 'agrawal':
            return
        new_cf_idx = int(new_cf_idx) % len(self.cf_cycle)
        if new_cf_idx == self.active_cf_idx and self.auto_generator is not None:
            return  # nada que hacer

        self.active_cf_idx = new_cf_idx

        # reconstruimos el generador con la nueva cf, misma semilla
        params = self._prepare_method_params('agrawal', self.last_n_features, self.last_noise, 0.8)
        params["random_state"] = self.current_seed
        config = GeneratorConfig(**params)
        factory = GeneratorFactory()
        self.auto_generator = factory.create_generator(GeneratorType.AGRAWAL, config)
        print(f"🔁 Drift: ahora usando Agrawal(classification_function={params['classification_function']})")

    def _apply_target_prior_shift(self, df: pd.DataFrame, level: int) -> pd.DataFrame:
        """
        Cambia la distribución del target (label shift) por batch.
        Pensado para binario: alterna 75/25 ↔ 25/75 según el nivel.
        level=0 => ~50/50 (sin cambio).
        """
        if level <= 0 or df.empty or 'target' not in df.columns:
            return df

        df = df.copy()
        s = df['target'].astype(int)

        # objetivo (p1 deseado) según nivel: 0.75, 0.25, 0.75, 0.25, ...
        desired_p1 = 0.75 if (level % 2 == 1) else 0.25

        n = len(s)
        n1_now = int((s == 1).sum())
        n1_want = int(round(desired_p1 * n))

        rng = np.random.default_rng((self.current_seed or 0) + level)
        if n1_want > n1_now:
            # necesitamos más 1s -> convertir algunos 0→1
            to_flip = n1_want - n1_now
            zeros_idx = np.where(s.values == 0)[0]
            if to_flip > 0 and zeros_idx.size > 0:
                flip_idx = rng.choice(zeros_idx, size=min(to_flip, zeros_idx.size), replace=False)
                s.iloc[flip_idx] = 1
        elif n1_want < n1_now:
            # necesitamos menos 1s -> convertir algunos 1→0
            to_flip = n1_now - n1_want
            ones_idx = np.where(s.values == 1)[0]
            if to_flip > 0 and ones_idx.size > 0:
                flip_idx = rng.choice(ones_idx, size=min(to_flip, ones_idx.size), replace=False)
                s.iloc[flip_idx] = 0

        df['target'] = s
        return df

    def generate_auto_batch(self):
        """Genera y agrega un batch de datos en modo automático (25 instancias = +1 día) con drift cada 400 instancias."""
        if not self.auto_mode_active or self.auto_generator is None:
            print("Auto generator is not active")
            return self.current_data

        # --- Determinar si toca DRIFT antes de generar este batch ---
        current_n = len(self.current_data)
        if self.last_generator_type == 'agrawal':
            desired_cf_idx = (current_n // self.drift_span) % len(self.cf_cycle)
            if desired_cf_idx != self.active_cf_idx:
                self._reconfigure_agrawal_cf(desired_cf_idx)

        # --- Peek + batch sin consumo doble del stream ---
        try:
            x0, y0 = next(self.auto_generator.take(1))
        except StopIteration:
            print("Stream returned no samples on peek")
            return self.current_data

        feature_cols = [str(c) for c in x0.keys()]
        cols = feature_cols + ["target"]

        rows = [list(x0.values()) + [y0]]
        for x, y in self.auto_generator.take(self.auto_batch_size - 1):
            rows.append(list(x.values()) + [y])

        if not rows:
            return self.current_data

        df_batch = pd.DataFrame(rows, columns=cols)

        # --- TARGET DRIFT cada 1000 instancias (label shift) ---
        current_n = len(self.current_data)
        desired_level = current_n // self.target_drift_span
        if desired_level != self.target_drift_level:
            self.target_drift_level = desired_level
            if self.target_drift_level > 0:
                print(f"🌊 Target drift nivel {self.target_drift_level} desde sample_id {current_n}")

        # Aplica el label shift al batch (si level>0)
        df_batch = self._apply_target_prior_shift(df_batch, self.target_drift_level)

        # --------- sample_id ----------
        start_id = (
            int(self.current_data["sample_id"].iloc[-1]) + 1
            if (not self.current_data.empty and "sample_id" in self.current_data.columns)
            else 0
        )
        df_batch["sample_id"] = range(start_id, start_id + len(df_batch))

        # --------- timestamp: 25 instancias = +1 día ----------
        step = pd.Timedelta(days=1)
        group_size = 25

        if not self.current_data.empty and "timestamp" in self.current_data.columns:
            last_ts = pd.to_datetime(self.current_data["timestamp"].iloc[-1], errors="coerce")
            if pd.isna(last_ts):
                last_ts = pd.Timestamp.today().floor('D')
            last_sid = int(self.current_data["sample_id"].iloc[-1])
            last_k = last_sid // group_size
            base = last_ts - last_k * step
        else:
            last_sid_batch = start_id + len(df_batch) - 1
            last_k_batch = last_sid_batch // group_size
            base = pd.Timestamp.today().floor('D') - last_k_batch * step

        sid = df_batch["sample_id"].astype(int)
        df_batch["timestamp"] = base + (sid // group_size) * step

        # --- Balanceo si procede ---
        _native_balanced = self.last_generator_type in {'agrawal', 'stagger', 'mixed'}
        if self.balance_classes and not _native_balanced and 'target' in df_batch.columns:
            df_batch = self._rebalance_dataframe(
                df_batch, target_col="target", strategy="downsample",
                random_state=self.current_seed or 42
            )

        # Append al buffer
        self.current_data = pd.concat([self.current_data, df_batch], ignore_index=True)
        self.dataset_info["samples"] = len(self.current_data)
        self.dataset_info["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Auto batch appended: {len(df_batch)} samples (total={len(self.current_data)})")

        # --- Reinicio al superar max_samples (10.000): reset cf=0 y semilla+1 ---
        if len(self.current_data) >= self.max_samples:
            new_seed = (self.current_seed or 0) + 1
            print(f"Limit of {self.max_samples:,} reached. Restarting stream with seed {new_seed} and cf=0...")

            # Marcar flag para reinicio del dashboard
            self.just_restarted = True

            # Resetear target drift level
            self.target_drift_level = 0

            # Reinicio: cf=0 explícito
            self._setup_real_time_generator(
                self.last_generator_type,
                self.last_n_features,
                self.last_noise,
                new_seed,
                cf_idx=0  # << reset cf a 0
            )

            # Sembrar primer batch del nuevo ciclo
            self.current_data = pd.DataFrame()
            try:
                x1, y1 = next(self.auto_generator.take(1))
                feature_cols2 = [str(c) for c in x1.keys()]
                cols2 = feature_cols2 + ["target"]

                rows2 = [list(x1.values()) + [y1]]
                for x, y in self.auto_generator.take(self.auto_batch_size - 1):
                    rows2.append(list(x.values()) + [y])

                if rows2:
                    df_batch2 = pd.DataFrame(rows2, columns=cols2)
                    df_batch2["sample_id"] = range(0, len(df_batch2))
                    step = pd.Timedelta(days=1)
                    group_size = 25
                    sid2 = df_batch2["sample_id"].astype(int)
                    base2 = pd.Timestamp.today().floor('D') - ((len(df_batch2) - 1) // group_size) * step
                    df_batch2["timestamp"] = base2 + (sid2 // group_size) * step
                    self.current_data = df_batch2
                    self.dataset_info["samples"] = len(self.current_data)
                    self.dataset_info["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except StopIteration:
                print("Stream returned no samples after restart")

            self.current_seed = new_seed

        return self.current_data

    def reconfigure_generator(self, generator_type, n_features, noise, random_state):
        return self._setup_real_time_generator(generator_type, n_features, noise, random_state)

# Instancia global
data_generator = DashDataGenerator()

# ----------------------------------------------------------------------
# A partir de aquí sigue igual: layout, callbacks, funciones de visualización
# ----------------------------------------------------------------------

# ⚠️ Por espacio no lo repito entero aquí, pero es exactamente el bloque
# que me pasaste en tu mensaje anterior con todo el layout, estilos,
# callbacks (generate_dataset_and_update, update_auto_data, reconfigure_auto_generator)
# y todas las funciones auxiliares de visualización.

# CSS personalizado profesional
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                text-rendering: optimizeLegibility;
            }

            /* Asegurar que todo herede la nueva fuente */
            button, input, select, textarea,
            .custom-dropdown .Select-control,
            .custom-dropdown .Select-menu-outer,
            .custom-dropdown .Select-option,
            .metric-card, .dataset-info, .info-label, .info-value,
            .header-gradient, .subtitle, .section-title,
            table, th, td,
            .js-plotly-plot .plotly, .svg-container, .legend, .gtitle, .xtitle, .ytitle {
                font-family: inherit !important;
            }

            /* Pequeños ajustes de peso para estética */
            .header-gradient { font-weight: 800; }
            .section-title   { font-weight: 700; }
            .subtitle, .metric-label { font-weight: 500; }
            .metric-value { font-weight: 700; font-variant-numeric: tabular-nums; }
            .main-container {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                padding: 20px;
                color: #ffffff;
            }
            .header-gradient {
                background: linear-gradient(135deg, #00d4ff 0%, #51cf66 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin: 0 0 20px 0;
                font-weight: 700;
                font-size: 3.5em;
            }
            .subtitle {
                text-align: center;
                color: #b0b0b0;
                font-size: 1.3em;
                margin-bottom: 30px;
                font-weight: 300;
            }
            .control-panel {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(0, 212, 255, 0.2);
            }
            button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 12px 35px rgba(0, 212, 255, 0.5) !important;
            }
            button:active {
                transform: translateY(0px) !important;
            }
            .custom-dropdown .Select-control {
                background-color: #1e1e1e !important;
                border: 1px solid #00d4ff !important;
                color: #ffffff !important;
            }
            .custom-dropdown .Select-menu-outer {
                background-color: #1e1e1e !important;
                border: 1px solid #00d4ff !important;
            }
            .custom-dropdown .Select-option {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            .custom-dropdown .Select-option:hover {
                background-color: #00d4ff !important;
                color: #000000 !important;
            }
            .section-title {
                background: linear-gradient(135deg, #00d4ff 0%, #51cf66 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin: 30px 0 20px 0;
                font-weight: 600;
                font-size: 2.5em;
            }
            .dataset-info {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(81, 207, 102, 0.2);
            }
            .info-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .info-label {
                color: #b0b0b0;
                font-weight: 500;
            }
            .info-value {
                color: #00d4ff;
                font-weight: 600;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(0, 212, 255, 0.2);
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(0, 212, 255, 0.4);
                transform: translateY(-2px);
            }
            .metric-value {
                font-size: 2.5em;
                font-weight: 700;
                color: #00d4ff;
                margin: 0;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
            }
            .metric-label {
                font-size: 0.9em;
                color: #ffffff;
                margin: 5px 0 0 0;
                font-weight: 500;
                letter-spacing: 1px;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
app.layout = html.Div([
    # Header
    html.H1("CALMOPS", className="header-gradient"),
    html.Div(id="stream-status", style={"textAlign": "center", "marginBottom": "10px"}),
    html.P("Dashboard", className="subtitle"),

    # --- Selector de semilla (solo cuando el stream está PARADO) ---
    html.Div([
        html.Label("Semilla del generador", style={
            'display': 'block', 'textAlign': 'center', 'marginTop': '20px', 'color': '#b0b0b0'
        }),
        dcc.Input(
            id='seed-input',
            type='number',
            value=data_generator.current_seed,
            debounce=True,
            style={
                'width': '160px', 'textAlign': 'center', 'margin': '8px auto',
                'display': 'block', 'backgroundColor': '#1e1e1e', 'color': '#fff',
                'border': '1px solid #00d4ff', 'borderRadius': '10px', 'padding': '8px'
            }
        ),
        html.Button([
            html.Div("🎲", style={'fontSize': '20px', 'marginBottom': '4px'}),
            html.Div("APLICAR SEMILLA", style={
                'fontSize': '13px', 'fontWeight': 'bold', 'letterSpacing': '1px'
            })
        ], id='apply-seed', style={
            'background': 'linear-gradient(135deg, #00d4ff 0%, #764ba2 100%)',
            'border': 'none', 'borderRadius': '12px', 'padding': '12px 18px',
            'margin': '10px auto 0', 'color': 'white', 'cursor': 'pointer',
            'boxShadow': '0 8px 22px rgba(0, 212, 255, 0.35)',
            'display': 'block'
        }),
        html.Div(
            "🔐 Para cambiar la semilla, detén el stream.",
            id='seed-hint',
            style={'textAlign': 'center', 'color': '#b0b0b0', 'marginTop': '8px', 'fontSize': '12px'}
        )
    ], style={'textAlign': 'center'}),  # <-- IMPORTANTE: coma aquí para separar del siguiente bloque

    # Panel de control
    html.Div([
        html.Div([
            html.Div([
                html.Span("⚙️", style={
                    'fontSize': '2.5em',
                    'marginRight': '15px',
                    'verticalAlign': 'middle'
                }),
                html.Span("Configuración de Stream", style={
                    'color': '#00d4ff',
                    'font-weight': '600',
                    'font-size': '2.0em',
                    'verticalAlign': 'middle',
                    'letterSpacing': '1px'
                })
            ], style={
                'textAlign': 'center',
                'marginBottom': '10px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            }),
            html.P("Generadores sintéticos River • Modo Infinito", style={
                'text-align': 'center',
                'color': '#b0b0b0',
                'font-size': '1.1em',
                'margin': '0 0 25px 0',
                'fontFamily': 'Segoe UI, sans-serif'
            })
        ]),

        # Botón iniciar stream
        html.Div([
            html.Button([
                html.Div("▶️", style={'fontSize': '24px', 'marginBottom': '5px'}),
                html.Div("INICIAR STREAM", style={
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'letterSpacing': '1px'
                })
            ], id='start-stream', style={
                'background': 'linear-gradient(135deg, #51cf66 0%, #40a65b 100%)',
                'border': 'none',
                'borderRadius': '15px',
                'padding': '20px 30px',
                'margin': '10px auto',
                'color': 'white',
                'cursor': 'pointer',
                'boxShadow': '0 8px 25px rgba(81, 207, 102, 0.4)',
                'transition': 'all 0.3s ease',
                'width': '250px',
                'minHeight': '80px',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',
                'justifyContent': 'center'
            })
        ], style={'text-align': 'center'}),

        # Botón detener stream
        html.Div([
            html.Button([
                html.Div("⏹️", style={'fontSize': '24px', 'marginBottom': '5px'}),
                html.Div("DETENER STREAM", style={
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'letterSpacing': '1px'
                })
            ], id='stop-stream', style={
                'background': 'linear-gradient(135deg, #ff6b6b 0%, #cc0000 100%)',
                'border': 'none',
                'borderRadius': '15px',
                'padding': '20px 30px',
                'margin': '10px auto',
                'color': 'white',
                'cursor': 'pointer',
                'boxShadow': '0 8px 25px rgba(255, 0, 0, 0.4)',
                'transition': 'all 0.3s ease',
                'width': '250px',
                'minHeight': '80px',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',
                'justifyContent': 'center'
            })
        ], style={'text-align': 'center', 'marginTop': '30px'})
    ], className='control-panel'),

    # Información del dataset
    html.Div(id='dataset-info-display'),

    # Métricas del dataset
    html.Div(id='dataset-metrics'),

    # Título de visualizaciones
    html.H2("ANÁLISIS VISUAL INTERACTIVO", className="section-title"),

    # Grid de visualizaciones
    html.Div(id='visualizations-container'),

    # Componente para manejar recargas del dashboard
    dcc.Location(id='app-url', refresh=True),

    # Intervalo para actualización automática
    dcc.Interval(
        id='auto-interval',
        interval=8000,  # 8 segundos (tu comentario decía 5, pero el valor es 8000 ms)
        n_intervals=0,
        disabled=True
    )
], style={
    'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
    'min-height': '100vh',
    'padding': '20px',
    'color': '#ffffff'
})

def _drift_sid_lists(df: pd.DataFrame, concept_span=400, target_span=1000):
    """Devuelve listas con sids de concept/target drift según el tamaño del df."""
    drift_sid_vals, target_sid_vals = [], []
    if 'sample_id' in df.columns:
        max_sid = int(pd.to_numeric(df['sample_id'], errors='coerce').fillna(-1).max())
        if max_sid >= concept_span:
            drift_sid_vals  = list(range(concept_span,  max_sid + 1, concept_span))
        if max_sid >= target_span:
            target_sid_vals = list(range(target_span, max_sid + 1, target_span))
    return drift_sid_vals, target_sid_vals

def create_target_share_over_time_figure(
    df,
    concept_drift_span: int = 400,   # cada 400 instancias (concept drift)
    target_drift_span: int = 1000,   # cada 1000 instancias (target drift)
    max_classes: int = 6             # por si el target tiene más de 2 clases
):
    """
    Muestra la distribución del target a lo largo del tiempo (líneas de % por clase)
    y marca:
      - Concept drift (rojo) cada 'concept_drift_span' instancias
      - Target drift  (azul) cada 'target_drift_span' instancias

    Correcciones clave:
      * No usa `add_vline` con pandas.Timestamp (usa shapes con xref/x/yref='y domain')
      * Para el primer eje se usan 'x' y 'y domain' (no 'x1'/'y1 domain')
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    # --------- Sanitizar timestamp / ordenar ---------
    use_time = 'timestamp' in df.columns
    if use_time:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        mask = ts.notna()
        if not mask.any():
            use_time = False
        else:
            df = df.loc[mask].copy()
            df['timestamp'] = ts[mask]
            df = df.sort_values('timestamp')

    # Si no hay tiempo, caemos a bins por sample_id para simular una línea temporal
    if not use_time:
        # Creamos una "serie temporal" por bins del sample_id
        sids = (df['sample_id'] if 'sample_id' in df.columns
                else pd.Series(np.arange(len(df)), index=df.index))
        # Elegimos ~√n bins, entre 10 y 60
        n = max(1, len(df))
        n_bins = min(60, max(10, int(n ** 0.5)))
        df = df.copy()
        df['__bin__'] = pd.cut(sids, bins=n_bins, labels=False, duplicates='drop')

    # --------- Construir tabla % por clase en el tiempo ---------
    if 'target' not in df.columns:
        return go.Figure()

    # Limitar nº de clases mostradas (si hay muchas)
    top_classes = df['target'].value_counts().sort_values(ascending=False).head(max_classes).index

    if use_time:
        # Agregar por día (puedes cambiar a '7D' si prefieres semanal)
        rule = '1D'
        g = (df.set_index('timestamp')['target']
               .pipe(lambda s: s.where(s.isin(top_classes), other='otros'))
               .groupby(pd.Grouper(freq=rule)).value_counts(normalize=True))  # proporciones
        wide = g.unstack(fill_value=0.0).sort_index()
        x_vals = wide.index
        x_is_time = True
    else:
        g = (df[['__bin__', 'target']]
               .assign(target=lambda d: d['target'].where(d['target'].isin(top_classes), other='otros'))
               .groupby(['__bin__', 'target']).size())
        counts = g.unstack(fill_value=0).sort_index()
        denom = counts.sum(axis=1).replace(0, np.nan)
        wide = (counts.div(denom, axis=0) * 100.0).fillna(0.0)
        x_vals = wide.index
        x_is_time = False

    # Si agregamos por tiempo con normalize=True arriba, ya están en [0,1]; conviértelo a %
    if use_time:
        wide = (wide * 100.0).astype(float)

    # --------- Figura con líneas (% por clase) ---------
    fig = go.Figure()
    for cls in wide.columns:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=wide[cls],
            mode='lines',
            name=str(cls),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                ("Fecha: %{x|%d-%m-%Y}" if x_is_time else "Eje: %{x}") +
                "<br>%: %{y:.2f}<extra></extra>"
            )
        ))

    # --------- Helpers robustos para vlines en fechas ---------
    def _to_dt(x):
        # Convierte a datetime nativo si es Timestamp/np.datetime64
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime()
        try:
            return pd.to_datetime(x).to_pydatetime()
        except Exception:
            return x  # valores no temporales (bins enteros)

    def _add_vline_shape(fig, xpos, color, text):
        """
        Dibuja una vline que cruza TODO el dominio Y del subplot principal.
        Usa shapes (no add_vline) para evitar el TypeError con Timestamp.
        """
        xplot = _to_dt(xpos) if x_is_time else xpos
        fig.add_shape(
            type="line",
            x0=xplot, x1=xplot, y0=0, y1=1,
            xref="x", yref="y domain",
            line=dict(color=color, width=2, dash="dot")
        )
        fig.add_annotation(
            x=xplot, y=1.02,
            xref="x", yref="y domain",
            text=text,
            showarrow=False,
            font=dict(color=color, size=10),
            bgcolor="rgba(0,0,0,0)"
        )

    # --------- Calcular fechas/posiciones de drift ---------
    drift_sid_vals = []
    if 'sample_id' in df.columns:
        try:
            max_sid = int(pd.to_numeric(df['sample_id'], errors='coerce').fillna(-1).max())
            if max_sid >= min(concept_drift_span, target_drift_span):
                # múltiplos de cada span
                concept_marks = list(range(concept_drift_span, max_sid + 1, concept_drift_span))
                target_marks  = list(range(target_drift_span,  max_sid + 1, target_drift_span))
            else:
                concept_marks, target_marks = [], []
        except Exception:
            concept_marks, target_marks = [], []
    else:
        concept_marks, target_marks = [], []

    # Mapear sample_id → posición X
    if x_is_time and 'sample_id' in df.columns:
        # fecha del primer registro cuyo sample_id == marca (normalizada al día)
        # (usamos el df original ordenado por timestamp)
        base_df = df.sort_values('timestamp')
        sid_ts = base_df[['sample_id', 'timestamp']].dropna().copy()
        sid_ts['sample_id'] = pd.to_numeric(sid_ts['sample_id'], errors='coerce').astype('Int64')

        def _sid_to_date(marks):
            if not len(marks) or sid_ts.empty:
                return []
            hits = sid_ts[sid_ts['sample_id'].isin(marks)]
            if hits.empty:
                return []
            return sorted(pd.to_datetime(hits['timestamp']).dt.normalize().unique())

        concept_xs = _sid_to_date(concept_marks)
        target_xs  = _sid_to_date(target_marks)
    else:
        # Eje no temporal: usamos el múltiplo directamente en X si coincide con el rango
        concept_xs = [m for m in (concept_marks or []) if len(x_vals) == 0 or (x_vals.min() <= m <= x_vals.max())]
        target_xs  = [m for m in (target_marks  or []) if len(x_vals) == 0 or (x_vals.min() <= m <= x_vals.max())]

    # --------- Pintar líneas de drift ---------
    for dt in concept_xs:
        _add_vline_shape(fig, dt, color="red",  text="Concept drift")
    for dt in target_xs:
        _add_vline_shape(fig, dt, color="#3b82f6", text="Target drift")

    # --------- Layout / tema ---------
    title = "DISTRIBUCIÓN DEL TARGET EN EL TIEMPO"
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Tiempo" if x_is_time else "Bins de instancias",
        yaxis_title="Porcentaje (%)",
        hovermode="x unified",
        showlegend=True
    )

    # Si usas tu helper de tema oscuro:
    try:
        return apply_dark_theme(fig, title)
    except Exception:
        return fig

def apply_dark_theme(fig, title):
    """Aplica tema oscuro consistente a las figuras"""
    fig.update_layout(
        plot_bgcolor=DARK_THEME['plot_bgcolor'],
        paper_bgcolor=DARK_THEME['paper_bgcolor'],
        font_color=DARK_THEME['font_color'],
        title=dict(
            text=title,
            font=dict(size=18, color=DARK_THEME['title_color']),
            x=0.5
        ),
        xaxis=dict(gridcolor=DARK_THEME['grid_color']),
        yaxis=dict(gridcolor=DARK_THEME['grid_color']),
        showlegend=True,
        legend=dict(
            font=dict(color=DARK_THEME['font_color']),
            bgcolor='rgba(255,255,255,0.05)'
        )
    )
    return fig

# Callbacks

@app.callback(
    [Output('auto-interval', 'disabled'),
     Output('dataset-info-display', 'children'),
     Output('dataset-metrics', 'children'),
     Output('visualizations-container', 'children'),
     Output('stream-status', 'children')],   # 👈 nuevo
    [Input('start-stream', 'n_clicks'),
     Input('stop-stream', 'n_clicks')],
    prevent_initial_call=True
)

def toggle_stream(n_start, n_stop):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'stop-stream':
        data_generator.auto_mode_active = False
        st = read_stream_status(False)
        badge = render_stream_status_badge(st["status"] == "running", st.get("updated_at"))
        return True, dash.no_update, dash.no_update, dash.no_update, badge  # ✅ 5 outputs

    # start-stream
    generator_type = 'agrawal'
    noise = 0.0
    data_generator.balance_classes = True

    data_generator.reconfigure_generator(generator_type, 9, noise, data_generator.current_seed)
    data_generator.auto_mode_active = True

    df = data_generator.generate_auto_batch()
    st = read_stream_status(True)
    badge = render_stream_status_badge(st["status"] == "running", st.get("updated_at"))  # ✅ crea badge

    if df.empty:
        info_display = html.Div("⚠️ No se pudo generar datos iniciales", style={'color': 'red'})
        return False, info_display, [], [], badge  # ✅ 5 outputs

    info_display = html.Div([
        html.H3("Información del Dataset (Stream)", style={'color': '#51cf66'}),
        html.Div([
            html.Div([
                html.Span("Generador:", className='info-label'),
                html.Span("agrawal", className='info-value')
            ], className='info-item'),
            html.Div([
                html.Span("Balanceado:", className='info-label'),
                html.Span("Sí", className='info-value')
            ], className='info-item'),
            html.Div([
                html.Span("Muestras iniciales:", className='info-label'),
                html.Span(str(len(df)), className='info-value')
            ], className='info-item'),
            html.Div([  # 👈 NUEVO: semilla actual
                html.Span("Semilla:", className='info-label'),
                html.Span(str(data_generator.current_seed), className='info-value')
            ], className='info-item'),
        ])
    ], className='dataset-info')
    metrics = html.Div("⚡ Stream inicializado", className="metric-card")
    visualizations = create_all_visualizations(df)

    # auto-interval habilitado (False)
    return False, info_display, metrics, visualizations, badge  # ✅ 5 outputs




# Importar funciones de visualización del dashboard original (create_all_visualizations, etc.)
# Por brevedad, incluiré solo la función principal y algunas de ejemplo

def create_all_visualizations(df):
    """Crea todas las visualizaciones adaptadas para datos de River - COMPLETO como dashboard.py"""
    if df.empty:
        return [html.Div("No hay datos para visualizar", style={'color': '#b0b0b0', 'text-align': 'center', 'margin': '50px'})]
    
    visualizations = []
    
    # Verificar qué columnas tenemos disponibles (adaptado para River)
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    has_target = 'target' in df.columns
    has_blocks = 'block' in df.columns
    
    if not feature_cols:
        return [html.Div("No se encontraron features para visualizar", style={'color': '#ff6b6b', 'text-align': 'center', 'margin': '50px'})]
    
    # SECCIÓN 1: ANÁLISIS BÁSICO
    # ============================
    section1_title = html.H2("🔍 ANÁLISIS BÁSICO", style={'color': '#51cf66', 'text-align': 'center', 'margin': '30px 0 20px 0'})
    visualizations.append(section1_title)
    
    # Fila 1: Quality Breakdown + Correlation Heatmap
    quality_results = calculate_quality_metrics(df)
    row1 = html.Div([
        html.Div([
            dcc.Graph(figure=create_quality_breakdown_figure(quality_results), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
        html.Div([
            dcc.Graph(figure=create_correlation_figure(df), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
    ])
    visualizations.append(row1)
    
    # Fila 2: PCA Projection + Statistical Summary  
    row2 = html.Div([
        html.Div([
            dcc.Graph(figure=create_pca_figure(df), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
        html.Div([
            dcc.Graph(figure=create_statistical_summary_figure(df), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
    ])
    visualizations.append(row2)
    
    # Fila 3: Target Distribution (ancho completo)
    if has_target:
        row3 = html.Div([
            dcc.Graph(figure=create_target_distribution_figure(df), style={'height': '400px'})
        ], style={'margin': '1%'})
        visualizations.append(row3)
        
        # Row 3b: Target evolution over time (Label Drift visualization)
        row3b = html.Div([
            dcc.Graph(figure=create_target_share_over_time_figure(df), style={'height': '400px'})
        ], style={'margin': '1%'})
        visualizations.append(row3b)
    
    # SECCIÓN 2: DISTRIBUCIONES INDIVIDUALES
    # =======================================
    section2_title = html.H2("📈 DISTRIBUCIONES INDIVIDUALES", style={'color': '#51cf66', 'text-align': 'center', 'margin': '30px 0 20px 0'})
    visualizations.append(section2_title)
    
    # Fila 4: Feature Distributions individuales (adaptado de dashboard.py)
    for i in range(0, len(feature_cols), 2):
        row = html.Div([
            html.Div([
                dcc.Graph(figure=create_feature_distribution_figure(df, feature_cols[i]), style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
            html.Div([
                dcc.Graph(figure=create_feature_distribution_figure(df, feature_cols[i + 1]) if i + 1 < len(feature_cols) else go.Figure(), style={'height': '400px'})
            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
        ])
        visualizations.append(row)
    
    # SECCIÓN 3: ANÁLISIS EXPLORATORIO
    # =================================
    section3_title = html.H2("🔎 ANÁLISIS EXPLORATORIO", style={'color': '#51cf66', 'text-align': 'center', 'margin': '30px 0 20px 0'})
    visualizations.append(section3_title)
    
    # Fila 5: Feature Evolution (ancho completo) - Evolución temporal
    if has_blocks:
        evolution_fig = create_evolution_by_blocks_figure(df)
    else:
        evolution_fig = create_evolution_by_instances_figure(df)
    
    row5 = html.Div([
        dcc.Graph(figure=evolution_fig, style={'height': '400px'})
    ], style={'margin': '1%'})
    visualizations.append(row5)
    
    # Fila 6: Box plots y Violin plots
    row6 = html.Div([
        html.Div([
            dcc.Graph(figure=create_box_figure_river(df), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
        html.Div([
            dcc.Graph(figure=create_violin_plots_figure(df), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
    ])
    visualizations.append(row6)

    # SECCIÓN 3C: CATEGÓRICAS EN EL TIEMPO
    section3c_title = html.H2("🧩 EVOLUCIÓN TEMPORAL DE VARIABLES CATEGÓRICAS",
                              style={'color': '#51cf66', 'text-align': 'center', 'margin': '30px 0 20px 0'})
    visualizations.append(section3c_title)

    cat_columns = get_categorical_like_columns(df, max_unique=20, unique_ratio=0.05)
    cat_columns = cat_columns[:6]  # limitar para no saturar

    if not cat_columns:
        visualizations.append(html.Div("No se detectaron columnas categóricas para mostrar.",
                                       style={'textAlign': 'center', 'color': '#b0b0b0', 'margin': '10px 0 20px 0'}))
    else:
        # Colocarlas de a dos por fila
        for i in range(0, len(cat_columns), 2):
            c1 = cat_columns[i]
            c2 = cat_columns[i+1] if i + 1 < len(cat_columns) else None
            row = html.Div([
                html.Div([dcc.Graph(figure=create_categorical_time_evolution_figure(df, c1),
                                    style={'height': '400px'})],
                         style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                html.Div([dcc.Graph(figure=create_categorical_time_evolution_figure(df, c2),
                                    style={'height': '400px'})] if c2 else [],
                         style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})],
                style={'width': '100%'}
            )
            visualizations.append(row)

    # SECCIÓN 4: ANÁLISIS AVANZADO
    # =============================
    section4_title = html.H2("🔬 ANÁLISIS AVANZADO", style={'color': '#51cf66', 'text-align': 'center', 'margin': '30px 0 20px 0'})
    visualizations.append(section4_title)
    
    # Fila 7: Scatter Matrix (ancho completo)
    row7 = html.Div([
        dcc.Graph(figure=create_scatter_matrix_figure(df), style={'height': '400px'})
    ], style={'margin': '1%'})
    visualizations.append(row7)
    
    # Fila 8: Feature Summary + Categorical Mode
    row8 = html.Div([
        html.Div([
            dcc.Graph(figure=create_feature_summary_figure(df), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
        html.Div([
            dcc.Graph(figure=create_categorical_mode_figure(df), style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
    ])
    visualizations.append(row8)
    
    # SECCIÓN 5: MUESTRA DE DATOS
    section5_title = html.H2("📋 MUESTRA DE DATOS", style={
        'color': '#51cf66', 'text-align': 'center', 'margin': '30px 0 15px 0'
    })
    visualizations.append(section5_title)

    if df.empty:
        table = html.Div("⚠️ No hay datos generados todavía", 
                        style={'color': '#b0b0b0', 'text-align': 'center', 'margin': '20px'})
    else:
        sample_size = min(10, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        table = html.Div([
            html.P(f"Mostrando {sample_size} instancias aleatorias", 
                style={'text-align': 'center', 'color': '#b0b0b0', 'margin-bottom': '20px'}),
            create_data_table(df_sample)
        ])

    # Evitar variable no definida
    cat_fig = create_categorical_mode_figure(df)
    row_cat = html.Div()  # default vacío
    if len(cat_fig.data) > 0:
        row_cat = html.Div([
            dcc.Graph(figure=cat_fig, style={'height': '400px'})
        ], style={'margin': '1%'})
    visualizations.append(row_cat)
    
    visualizations.append(table)

    return visualizations

def create_target_distribution_figure(df):
    if 'target' not in df.columns:
        return go.Figure()

    target_counts = df['target'].value_counts()
    base_colors = [
        COLOR_PALETTE['primary'],
        COLOR_PALETTE['secondary'],
        COLOR_PALETTE['success'],
        COLOR_PALETTE['warning'],
        COLOR_PALETTE['accent'],
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(len(target_counts))]

    fig = go.Figure(data=[
        go.Bar(
            x=[f'Clase {i}' for i in target_counts.index],
            y=target_counts.values,
            marker_color=colors,
            text=target_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Muestras: %{y}<extra></extra>'
        )
    ])
    return apply_dark_theme(fig, "DISTRIBUCIÓN DE CLASES (River)")

def create_correlation_figure(df):
    """Crea mapa de correlación para datos River"""
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if len(feature_cols) < 2:
        return go.Figure()
    
    corr_matrix = df[feature_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={'size': 10},
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlación: %{z:.3f}<extra></extra>'
    ))
    
    fig = apply_dark_theme(fig, "MATRIZ DE CORRELACIÓN (River)")
    return fig

def create_evolution_by_blocks_figure(df):
    """Evolución por bloques, anclando a una fecha representativa por bloque si hay timestamp."""
    if 'block' not in df.columns:
        return go.Figure()

    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [c for c in df.columns if c not in excluded_cols][:4]
    if not feature_cols:
        return go.Figure()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Feature {c}" for c in feature_cols])

    use_time = 'timestamp' in df.columns
    if use_time:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        m = ts.notna()
        if not m.any():
            use_time = False
        else:
            df = df.loc[m].copy()
            df['timestamp'] = ts[m]
            # fecha representativa por bloque (primera fecha del bloque)
            block_times = (df.groupby('block')['timestamp']
                             .agg(lambda s: s.min())
                             .sort_index())
            x_vals = block_times.values  # fechas
    else:
        x_vals = df['block'].dropna().sort_values().unique()  # números de bloque

    gb = df.groupby('block')
    blocks_sorted = sorted(gb.groups.keys())
    for i, feat in enumerate(feature_cols):
        r, c = i // 2 + 1, i % 2 + 1
        means = gb[feat].mean().reindex(blocks_sorted)
        fig.add_trace(
            go.Scatter(
                x=(x_vals if use_time else blocks_sorted),
                y=means.values,
                mode='lines+markers',
                name=feat,
                hovertemplate=(f'<b>{feat}</b><br>'
                               f'{"Fecha" if use_time else "Bloque"}: %{{x}}<br>'
                               f'Media: %{{y:.3f}}<extra></extra>')
            ),
            row=r, col=c
        )

    fig.update_layout(height=400, showlegend=False)
    title = "EVOLUCIÓN POR BLOQUES (por fecha del bloque)" if use_time else "EVOLUCIÓN POR BLOQUES"
    fig = apply_dark_theme(fig, title)
    if use_time:
        fig.update_xaxes(tickformat="%d-%m-%Y", hoverformat="%d-%m-%Y")
    return fig



def create_evolution_by_instances_figure(df):
    """
    Evolución temporal por instancias con marcadores de concept drift.
    - Si hay 'timestamp', se agrega SIEMPRE a 1 día (1D) y se marcan vlines rojas
      en las fechas que correspondan a sample_id % 400 == 0 (excluyendo el 0).
    - Si no hay 'timestamp', cae al eje de 'sample_id' y dibuja vlines en los
      sample_id = 400, 800, 1200, ...
    """
    DRIFT_SPAN = 400  # cada 400 instancias
    GROUP_SIZE = 25   # 25 instancias = +1 día

    # ¿Tenemos timestamp válido?
    use_time = 'timestamp' in df.columns
    if use_time:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        m = ts.notna()
        if not m.any():
            use_time = False
        else:
            df = df.loc[m].copy()
            df['timestamp'] = ts[m]
            df = df.sort_values('timestamp')

    # Selección de features (omitir columnas técnicas)
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols][:4]
    if not feature_cols:
        return go.Figure()

    # Subplots 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Feature {col}" for col in feature_cols]
    )

    # ---------- Calcular posiciones de drift ----------
    TARGET_SPAN = 1000    # target drift cada 1000 instancias
    drift_sid_vals, target_sid_vals = [], []
    if 'sample_id' in df.columns:
        # múltiplos de 400 distintos de 0 hasta el máximo sample_id del df
        max_sid = int(pd.to_numeric(df['sample_id'], errors='coerce').fillna(-1).max())
        if max_sid >= DRIFT_SPAN:
            drift_sid_vals = list(range(DRIFT_SPAN, max_sid + 1, DRIFT_SPAN))
        if max_sid >= TARGET_SPAN:
            target_sid_vals = list(range(TARGET_SPAN, max_sid + 1, TARGET_SPAN))

    if use_time:
        # Resample DIARIO para evitar múltiples puntos por día
        df_rs = (df.set_index('timestamp')[feature_cols + (['sample_id'] if 'sample_id' in df.columns else [])]
                   .resample('1D').mean()
                   .dropna(how='all')
                   .reset_index())

        xs = df_rs['timestamp']

        # Mapear sample_id de fronteras de drift -> fechas (día)
        drift_dates, target_dates = [], []
        if 'sample_id' in df.columns and drift_sid_vals:
            # Tomamos las filas reales (no resampleadas) donde cambia el drift
            df_drift = df.loc[df['sample_id'].astype(int).isin(drift_sid_vals), ['sample_id', 'timestamp']].copy()
            if not df_drift.empty:
                # Normalizamos al día (porque el plot es diario)
                drift_dates = sorted(pd.to_datetime(df_drift['timestamp']).dt.normalize().unique())
        if 'sample_id' in df.columns and target_sid_vals:
            # Target drift dates
            df_target = df.loc[df['sample_id'].astype(int).isin(target_sid_vals), ['sample_id', 'timestamp']].copy()
            if not df_target.empty:
                target_dates = sorted(pd.to_datetime(df_target['timestamp']).dt.normalize().unique())

        for i, feature in enumerate(feature_cols):
            r = i // 2 + 1
            c = i % 2 + 1
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=df_rs[feature],
                    mode='lines',
                    name=feature,
                    hovertemplate=(
                        f'<b>{feature}</b><br>'
                        'Fecha: %{x|%d-%m-%Y}<br>'
                        'Valor: %{y:.3f}<extra></extra>'
                    )
                ),
                row=r, col=c
            )

            # Añadir líneas de drift en este subplot
            for dt in drift_dates:
                # Solo si cae dentro del rango visible
                if xs.min() <= dt <= xs.max():
                    fig.add_vline(
                        x=dt,
                        line_color='red',
                        line_width=2,
                        line_dash='dot',
                        row=r, col=c
                    )
                    # Anotación en la parte superior del subplot
                    xaxis_name = 'x' if (i + 1) == 1 else f'x{i+1}'
                    fig.add_annotation(
                        x=dt,
                        y=1.02,
                        xref=xaxis_name,  # 'x', 'x2', 'x3', ...
                        yref='paper',     # <- importante: no 'y1 domain'
                        text='Concept drift',
                        showarrow=False,
                        font=dict(color='red', size=10),
                        align='left',
                        bgcolor='rgba(255,0,0,0.10)'
                    )
            
            # Añadir líneas de target drift en azul
            for dt in target_dates:
                if xs.min() <= dt <= xs.max():
                    fig.add_vline(
                        x=dt,
                        line_color='deepskyblue',
                        line_width=2,
                        line_dash='dash',
                        row=r, col=c
                    )
                    xaxis_name = 'x' if (i + 1) == 1 else f'x{i+1}'
                    fig.add_annotation(
                        x=dt,
                        y=0.98,  # Posición ligeramente diferente
                        xref=xaxis_name,
                        yref='paper',
                        text='Target drift',
                        showarrow=False,
                        font=dict(color='deepskyblue', size=10),
                        align='left',
                        bgcolor='rgba(0,191,255,0.10)'
                    )

        title = "EVOLUCIÓN TEMPORAL (diaria) + DRIFT"
        fig.update_xaxes(tickformat="%d-%m-%Y")

    else:
        # Fallback por sample_id
        if 'sample_id' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'sample_id'})

        # Muestra ordenada para no sobrecargar
        df_plot = (
            df.sample(min(500, len(df))).sort_values('sample_id')
            if len(df) > 500 else df.sort_values('sample_id')
        )

        xs = df_plot['sample_id']
        for i, feature in enumerate(feature_cols):
            r = i // 2 + 1
            c = i % 2 + 1
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=df_plot[feature],
                    mode='lines',
                    name=feature,
                    hovertemplate=(
                        f'<b>{feature}</b><br>'
                        'Instancia: %{x}<br>'
                        'Valor: %{y:.3f}<extra></extra>'
                    )
                ),
                row=r, col=c
            )

            # Añadir líneas de drift (en sample_id)
            for sid in drift_sid_vals:
                if xs.min() <= sid <= xs.max():
                    fig.add_vline(
                        x=sid,
                        line_color='red',
                        line_width=2,
                        line_dash='dot',
                        row=r, col=c
                    )
                    xaxis_name = 'x' if (i + 1) == 1 else f'x{i+1}'
                    fig.add_annotation(
                        x=sid,
                        y=1.02,
                        xref=xaxis_name,  # 'x', 'x2', 'x3', ...
                        yref='paper',     # <- importante: no 'y1 domain'
                        text='Concept drift',
                        showarrow=False,
                        font=dict(color='red', size=10),
                        align='left',
                        bgcolor='rgba(255,0,0,0.10)'
                    )
            
            # Añadir líneas de target drift (en sample_id)
            for sid in target_sid_vals:
                if xs.min() <= sid <= xs.max():
                    fig.add_vline(
                        x=sid,
                        line_color='deepskyblue',
                        line_width=2,
                        line_dash='dash',
                        row=r, col=c
                    )
                    xaxis_name = 'x' if (i + 1) == 1 else f'x{i+1}'
                    fig.add_annotation(
                        x=sid,
                        y=0.98,
                        xref=xaxis_name,
                        yref='paper',
                        text='Target drift',
                        showarrow=False,
                        font=dict(color='deepskyblue', size=10),
                        align='left',
                        bgcolor='rgba(0,191,255,0.10)'
                    )

        title = "EVOLUCIÓN POR INSTANCIAS (fallback) + DRIFT"

    fig.update_layout(height=400, showlegend=False)
    fig = apply_dark_theme(fig, title)
    return fig




def create_box_figure_river(df):
    """Crea box plots para datos River"""
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if not feature_cols:
        return go.Figure()
    
    fig = go.Figure()
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['success'], COLOR_PALETTE['warning']]
    
    for col, color in zip(feature_cols, colors):
        fig.add_trace(go.Box(
            y=df[col],
            name=f"Feature {col}",
            boxpoints='outliers',
            marker_color=color,
            line=dict(color=color),
            hovertemplate=f'<b>{col}</b><br>Valor: %{{y:.3f}}<extra></extra>'
        ))
    
    fig = apply_dark_theme(fig, "ANÁLISIS DE DISTRIBUCIÓN - BOX PLOTS (River)")
    return fig

def create_feature_summary_figure(df):
    """Crea resumen estadístico para datos River"""
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if not feature_cols:
        return go.Figure()
    
    means = [df[col].mean() for col in feature_cols]
    stds = [df[col].std() for col in feature_cols]
    
    fig = go.Figure(data=[
        go.Bar(name='Media', x=[f"Feature {col}" for col in feature_cols], y=means, 
               marker_color=COLOR_PALETTE['primary'],
               hovertemplate='<b>%{x}</b><br>Media: %{y:.3f}<extra></extra>'),
        go.Bar(name='Desv. Estándar', x=[f"Feature {col}" for col in feature_cols], y=stds,
               marker_color=COLOR_PALETTE['secondary'],
               hovertemplate='<b>%{x}</b><br>Desviación: %{y:.3f}<extra></extra>')
    ])
    
    fig.update_layout(barmode='group')
    fig = apply_dark_theme(fig, "ESTADÍSTICAS DESCRIPTIVAS (River)")
    return fig

def create_data_table(df):
    if df.empty:
        return html.Div("⚠️ No hay datos", style={'color': '#b0b0b0'})

    # Seleccionar columnas (quitamos timestamp si es muy largo)
    display_cols = [col for col in df.columns if col != 'timestamp']
    df_display = df[display_cols].round(3)

    # Estilos de la tabla
    table_style = {
        'width': '90%',
        'margin': 'auto',
        'borderCollapse': 'collapse',
        'borderRadius': '12px',
        'overflow': 'hidden',
        'boxShadow': '0 4px 15px rgba(0, 0, 0, 0.4)'
    }

    header_style = {
        'background': 'linear-gradient(135deg, #00d4ff 0%, #51cf66 100%)',
        'color': '#ffffff',
        'fontWeight': 'bold',
        'padding': '12px',
        'textAlign': 'center',
        'fontSize': '14px',
        'letterSpacing': '1px'
    }

    cell_style = {
        'padding': '10px',
        'textAlign': 'center',
        'border': '1px solid rgba(255,255,255,0.1)',
        'backgroundColor': 'rgba(255,255,255,0.03)',
        'color': '#ffffff',
        'fontSize': '13px'
    }

    # Construcción de la tabla
    table = html.Table([
        html.Thead([
            html.Tr([html.Th(col, style=header_style) for col in df_display.columns])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(df_display.iloc[i][col], style=cell_style) 
                for col in df_display.columns
            ], style={'transition': '0.3s', 'cursor': 'pointer',
                      ':hover': {'backgroundColor': 'rgba(0, 212, 255, 0.2)'}})
            for i in range(len(df_display))
        ])
    ], style=table_style)

    return table  # 👈 aquí devolvemos la tabla


def create_scatter_matrix_figure(df):
    """Crea matriz de dispersión para datos River"""
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if len(feature_cols) < 2:
        return go.Figure()
    
    # Tomar muestra para no sobrecargar
    df_sample = df.sample(min(200, len(df))) if len(df) > 200 else df
    
    fig = px.scatter_matrix(
        df_sample,
        dimensions=feature_cols[:3],  # limitar a 3 features
        color='target' if 'target' in df.columns else None,
        title="MATRIZ DE DISPERSIÓN (River)"
    )
    fig = apply_dark_theme(fig, "MATRIZ DE DISPERSIÓN (River)")
    return fig
def create_pca_figure(df):
    """Crea proyección PCA para datos River"""
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if len(feature_cols) < 2:
        return go.Figure()
    
    df_sample = df.sample(min(200, len(df))) if len(df) > 200 else df
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_sample[feature_cols])
    
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
    
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df_sample['target'] if 'target' in df_sample.columns else None,
            title=f"PCA Projection (Varianza explicada: {pca.explained_variance_ratio_.sum():.2%})",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
        )
        fig = apply_dark_theme(fig, "PROYECCIÓN PCA (River)")
        return fig
    except:
        return go.Figure()


def create_violin_plots_figure(df):
    """Crea violin plots para datos River"""
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if not feature_cols:
        return go.Figure()
    
    fig = go.Figure()
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['success'], COLOR_PALETTE['warning']]
    
    for col, color in zip(feature_cols, colors):
        fig.add_trace(go.Violin(
            y=df[col],
            name=f"Feature {col}",
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            fillcolor=color,
            opacity=0.7
        ))
    
    fig = apply_dark_theme(fig, "VIOLIN PLOTS (River)")
    return fig


def create_categorical_mode_figure(df):
    """Gráfico de moda por feature para columnas categóricas o de baja cardinalidad."""
    cat_cols = get_categorical_like_columns(df, max_unique=20, unique_ratio=0.05)

    labels, counts, modes = [], [], []

    for col in cat_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        mode_vals = s.mode()
        if mode_vals.empty:
            continue
        raw_mode = mode_vals.iloc[0]              # moda en su tipo original (número/cadena)
        freq = int((s == raw_mode).sum())         # contar ANTES de convertir a str
        labels.append(col)
        counts.append(freq)
        modes.append(str(raw_mode))               # solo para mostrar

    # Si no hay columnas candidatas, como fallback muestra la moda del target (si existe)
    if not counts and 'target' in df.columns:
        s = df['target'].dropna()
        if not s.empty:
            raw_mode = s.mode().iloc[0]
            labels = ['target']
            counts = [int((s == raw_mode).sum())]
            modes = [str(raw_mode)]

    if not counts:
        return go.Figure()

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=counts,
            text=[f"Moda: {m}" for m in modes],
            textposition='auto',
            marker_color=COLOR_PALETTE['primary'],
            hovertemplate='<b>%{x}</b><br>%{text}<br>Frecuencia: %{y}<extra></extra>'
        )
    ])
    fig = apply_dark_theme(fig, "MODA POR FEATURE (categóricas / baja cardinalidad)")
    return fig



def create_feature_distribution_figure(df, feature_name):
    """Distribución de una feature específica (adaptado de dashboard.py)"""
    if df.empty or feature_name not in df.columns:
        return go.Figure()
    
    fig = px.histogram(
        df, 
        x=feature_name, 
        color='target' if 'target' in df.columns else None,
        title=f"DISTRIBUCIÓN DE {feature_name.upper()}",
        marginal="box",
        color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary']]
    )
    
    fig = apply_dark_theme(fig, f"DISTRIBUCIÓN DE {feature_name.upper()} (River)")
    return fig

def create_quality_breakdown_figure(quality_results):
    """Gráfico de calidad de datos (adaptado de dashboard.py)"""
    if not quality_results:
        return go.Figure()
    
    metrics = list(quality_results.keys())
    values = list(quality_results.values())
    
    # Colores basados en valores
    colors = []
    for v in values:
        if v >= 80:
            colors.append(COLOR_PALETTE['success'])
        elif v >= 60:
            colors.append(COLOR_PALETTE['warning'])
        else:
            colors.append(COLOR_PALETTE['secondary'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Puntuación: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(yaxis=dict(range=[0, 100]))
    fig = apply_dark_theme(fig, "MÉTRICAS DE CALIDAD DE DATOS (River)")
    return fig

def create_statistical_summary_figure(df):
    """Resumen estadístico (adaptado de dashboard.py)"""
    if df.empty:
        return go.Figure()
    
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if not feature_cols:
        return go.Figure()
    
    stats_data = []
    for col in feature_cols:
        stats_data.append({
            'Feature': f"Feature {col}",
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    fig = go.Figure(data=[
        go.Bar(
            name='Media', 
            x=stats_df['Feature'], 
            y=stats_df['Mean'],
            marker_color=COLOR_PALETTE['primary'],
            hovertemplate='<b>%{x}</b><br>Media: %{y:.3f}<extra></extra>'
        ),
        go.Bar(
            name='Desv. Estándar', 
            x=stats_df['Feature'], 
            y=stats_df['Std'],
            marker_color=COLOR_PALETTE['secondary'],
            hovertemplate='<b>%{x}</b><br>Desviación: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(barmode='group')
    fig = apply_dark_theme(fig, "RESUMEN ESTADÍSTICO POR FEATURE (River)")
    return fig

def calculate_quality_metrics(df):
    """Calcula métricas de calidad de datos (adaptado de dashboard.py)"""
    if df.empty:
        return {
            'statistical_validity': 80,
            'feature_diversity': 80,
            'data_completeness': 100,
            'class_balance': 80,
            'overall_score': 85
        }
    
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    # Statistical validity (basado en distribución normal de features)
    statistical_validity = 85
    if feature_cols:
        normality_scores = []
        for col in feature_cols[:4]:  # Limitar a 4 features para performance
            try:
                from scipy import stats
                _, p_value = stats.normaltest(df[col].dropna())
                normality_scores.append(min(100, max(0, (1 - p_value) * 100)))
            except:
                normality_scores.append(70)
        statistical_validity = np.mean(normality_scores) if normality_scores else 70
    
    # Feature diversity
    feature_diversity = min(100, len(feature_cols) * 15)
    
    # Data completeness
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    data_completeness = max(0, 100 - (missing_cells / total_cells * 100)) if total_cells > 0 else 100
    
    # Class balance
    if 'target' in df.columns:
        class_counts = df['target'].value_counts()
        if len(class_counts) > 1:
            imbalance_ratio = class_counts.max() / class_counts.min()
            class_balance = max(20, 100 - (imbalance_ratio - 1) * 20)
        else:
            class_balance = 50
    else:
        class_balance = 80
    
    # Overall score
    overall_score = (statistical_validity + feature_diversity + data_completeness + class_balance) / 4
    
    return {
        'statistical_validity': statistical_validity,
        'feature_diversity': feature_diversity,
        'data_completeness': data_completeness,
        'class_balance': class_balance,
        'overall_score': overall_score
    }
    
    
def is_integer_like_series(s: pd.Series) -> bool:
    """True si la serie es int o float con solo valores enteros."""
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            return False
        return np.all(np.mod(vals, 1) == 0)
    return False


def get_categorical_like_columns(df: pd.DataFrame, max_unique=20, unique_ratio=0.05):
    """
    Detecta columnas que 'parecen' categóricas:
    - dtype object/category/bool
    - o numéricas de baja cardinalidad (<= max_unique o ratio <= unique_ratio)
    Incluye columnas numéricas con valores enteros (float-int-like).
    Excluye columnas técnicas.
    """
    exclude = {'target', 'block', 'timestamp', 'sample_id'}
    cats = []
    n = max(1, len(df))
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        # Categóricas “obvias”
        if s.dtype == 'O' or str(s.dtype).startswith('category') or str(s.dtype) == 'bool':
            cats.append(col)
            continue
        # Numéricas low-card
        try:
            nunique = s.nunique(dropna=True)
            if nunique <= max_unique or (nunique / n) <= unique_ratio:
                cats.append(col)
                continue
        except Exception:
            pass
        # Floats que realmente son enteros (0.0, 1.0, 2.0…)
        if is_integer_like_series(s):
            nunique = s.nunique(dropna=True)
            if nunique <= max_unique or (nunique / n) <= unique_ratio:
                cats.append(col)
    return cats


def _auto_time_rule_from_range(ts: pd.Series) -> str:
    """Elige regla de resampleo según el rango temporal."""
    ts = pd.to_datetime(ts, errors='coerce').dropna()
    if ts.empty:
        return None
    delta = ts.max() - ts.min()
    # Heurística sencilla
    if delta.total_seconds() <= 60 * 30:       # ≤ 30 min
        return '5s'
    if delta.total_seconds() <= 60 * 60 * 6:   # ≤ 6 h
        return '1min'
    if delta.total_seconds() <= 60 * 60 * 24:  # ≤ 24 h
        return '5min'
    if delta.days <= 7:
        return '1h'
    return '1D'


def _prepare_time_axis(df: pd.DataFrame):
    """
    Devuelve (modo, x_index, labels):
      - modo = 'timestamp'|'block'|'bins'
      - x_index = eje X ordenado (DatetimeIndex/int)
      - labels = etiquetas para hover/ticks
    """
    # 1) Timestamp
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        if ts.notna().any():
            df = df.loc[ts.notna()].copy()
            df['__ts__'] = ts[ts.notna()]
            rule = _auto_time_rule_from_range(df['__ts__'])
            if rule:
                # devolvemos el índice temporal y el modo
                return 'timestamp', (df['__ts__'], rule), None
            else:
                return 'timestamp', (df['__ts__'], None), None

    # 2) Block
    if 'block' in df.columns:
        xs = pd.Index(sorted(df['block'].dropna().unique()))
        return 'block', (xs, None), [f'Bloque {x}' for x in xs]

    # 3) Bins sobre sample_id (o índice)
    sids = df['sample_id'] if 'sample_id' in df.columns else pd.Series(np.arange(len(df)))
    bins = min(30, max(5, int(len(df) ** 0.5)))  # ~√n, entre 5 y 30
    cats = pd.cut(sids, bins=bins, labels=False, duplicates='drop')
    df = df.copy()
    df['__bin__'] = cats
    xs = pd.Index(sorted(df['__bin__'].dropna().unique()))
    return 'bins', (xs, None), [f'Bin {int(x)}' for x in xs]

def create_categorical_time_evolution_figure(
    df: pd.DataFrame,
    column: str,
    normalize: bool = True,
    max_categories: int = 5,
    rolling: int = 1,
    title_prefix: str = "EVOLUCIÓN TEMPORAL"
):
    import pandas as pd, numpy as np, plotly.graph_objects as go
    if column not in df.columns or df.empty:
        return go.Figure()

    # --- elegir eje (timestamp / block / bins) ---
    mode, xinfo, labels = _prepare_time_axis(df)

    s_all = df[column].astype('category').astype(str)
    if s_all.nunique(dropna=True) <= 1:
        return go.Figure()

    # Top categorías + compactar resto
    top_vals = s_all.value_counts().head(max_categories).index.tolist()
    df_top = df.copy()
    df_top[column] = df_top[column].astype(str)
    df_top[column] = np.where(df_top[column].isin(top_vals),
                              df_top[column],
                              f"otros ({s_all.nunique()-len(top_vals)})")

    counts = None
    x = None
    bin_edges = None  # <<<< para reutilizar luego

    if mode == 'timestamp':
        ts, _ = xinfo
        df_top['__ts__'] = pd.to_datetime(df_top['timestamp'], errors='coerce')
        df_top = df_top.loc[df_top['__ts__'].notna()].copy()

        # si hay pocas por día, forzamos bins
        per_day = (df_top.set_index('__ts__').resample('1D').size().mean())
        if np.isnan(per_day) or per_day < 100:
            mode = 'bins'  # caeremos al branch de bins más abajo
        else:
            rule = '7D'
            g = (df_top.set_index('__ts__')[column]
                          .groupby(pd.Grouper(freq=rule)).value_counts())
            counts = g.unstack(fill_value=0)
            full_index = pd.date_range(counts.index.min(), counts.index.max(), freq=rule)
            counts = counts.reindex(full_index, fill_value=0)
            x = counts.index

    if mode == 'block':
        xs, _ = xinfo
        g = df_top.groupby(['block', column]).size()
        counts = g.unstack(fill_value=0).reindex(xs, fill_value=0)
        x = counts.index

    if mode == 'bins':
        # Usar los MISMOS edges para todo
        sids = (df_top['sample_id'] if 'sample_id' in df_top.columns
                else pd.Series(np.arange(len(df_top)), index=df_top.index))
        bins = max(10, min(50, len(df_top) // 200))
        # >>> retenemos edges
        df_top['__bin__'], bin_edges = pd.cut(
            sids, bins=bins, labels=False, duplicates='drop', retbins=True
        )
        g = df_top.groupby(['__bin__', column]).size()
        counts = g.unstack(fill_value=0).sort_index()
        x = counts.index

    # Normalizar a %
    if counts is None or counts.empty:
        return go.Figure()
    if normalize:
        denom = counts.sum(axis=1).replace(0, np.nan)
        ydata = (counts.div(denom, axis=0) * 100.0).fillna(0.0)
        yaxis_title = "Porcentaje (%)"
    else:
        ydata = counts
        yaxis_title = "Frecuencia"

    # Suavizado
    if rolling and rolling > 1:
        ydata = ydata.rolling(3, min_periods=1).mean()
    else:
        ydata = ydata.ewm(span=3, adjust=False).mean()

    # Figura
    fig = go.Figure()
    for cat in ydata.columns:
        fig.add_trace(go.Scatter(
            x=x, y=ydata[cat].values, mode='lines', name=str(cat),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                          ("Fecha: %{x|%d-%m-%Y}" if mode == 'timestamp' else "Eje: %{x}") +
                          "<br>Valor: %{y:.2f}" + ("%" if normalize else "") + "<extra></extra>"
        ))

    # --- vlines (concept/target) ---
    concept_span, target_span = 400, 1000
    drift_sids, target_sids = _drift_sid_lists(df, concept_span, target_span)

    def _add_vline(xpos, color, text, dash):
        # convertir tipo correcto
        if mode == 'timestamp' and isinstance(xpos, pd.Timestamp):
            xpos = xpos.to_pydatetime()
        elif mode != 'timestamp':
            xpos = int(xpos)
        fig.add_vline(x=xpos, line_color=color, line_width=2, line_dash=dash)
        fig.add_annotation(x=xpos, y=1.02, xref='x', yref='y domain',
                           text=text, showarrow=False,
                           font=dict(color=color, size=10))

    if mode == 'timestamp' and 'sample_id' in df.columns and 'timestamp' in df.columns:
        m = df[['sample_id', 'timestamp']].dropna().copy()
        m['sample_id'] = pd.to_numeric(m['sample_id'], errors='coerce').astype('Int64')
        m['timestamp'] = pd.to_datetime(m['timestamp'], errors='coerce').dt.normalize()
        # fechas candidatas
        d_dates = sorted(m[m['sample_id'].isin(drift_sids)]['timestamp'].dropna().unique())
        t_dates = sorted(m[m['sample_id'].isin(target_sids)]['timestamp'].dropna().unique())
        # pintar solo si caen dentro del rango visible
        x_min, x_max = x.min(), x.max()
        for dt in d_dates:
            if x_min <= dt <= x_max: _add_vline(dt, 'red', 'Concept drift', 'dot')
        for dt in t_dates:
            if x_min <= dt <= x_max: _add_vline(dt, 'deepskyblue', 'Target drift', 'dash')

    elif mode == 'block' and 'sample_id' in df.columns and 'block' in df.columns:
        m = df[['sample_id','block']].dropna().copy()
        m['sample_id'] = pd.to_numeric(m['sample_id'], errors='coerce').astype('Int64')
        sid_to_block = dict(zip(m['sample_id'], m['block']))
        xs = set(x.tolist())
        for sid in drift_sids:
            blk = sid_to_block.get(sid, None)
            if blk in xs: _add_vline(blk, 'red', 'Concept drift', 'dot')
        for sid in target_sids:
            blk = sid_to_block.get(sid, None)
            if blk in xs: _add_vline(blk, 'deepskyblue', 'Target drift', 'dash')

    elif mode == 'bins' and 'sample_id' in df.columns and bin_edges is not None:
        m = df[['sample_id']].dropna().copy()
        m['sample_id'] = pd.to_numeric(m['sample_id'], errors='coerce')
        # >>> usar EXACTAMENTE los mismos edges
        sid_bins = pd.cut(m['sample_id'], bins=bin_edges, labels=False, duplicates='drop')
        sid_to_bin = dict(zip(m['sample_id'].astype(int), sid_bins))
        xs = set(x.tolist())
        for sid in drift_sids:
            b = sid_to_bin.get(int(sid), None)
            if pd.notna(b) and int(b) in xs: _add_vline(int(b), 'red', 'Concept drift', 'dot')
        for sid in target_sids:
            b = sid_to_bin.get(int(sid), None)
            if pd.notna(b) and int(b) in xs: _add_vline(int(b), 'deepskyblue', 'Target drift', 'dash')

    # Layout
    if mode == 'timestamp':
        x_title = "Tiempo";   title = f"{title_prefix} - {column} (por tiempo)"
    elif mode == 'block':
        x_title = "Bloque";   title = f"{title_prefix} - {column} (por bloque)"
    else:
        x_title = "Bins de instancias"; title = f"{title_prefix} - {column} (por bins)"

    fig.update_layout(xaxis_title=x_title, yaxis_title=yaxis_title,
                      hovermode='x unified', showlegend=True)
    fig = apply_dark_theme(fig, title)
    if labels is not None and mode != 'timestamp':
        fig.update_xaxes(tickmode='array', tickvals=list(x), ticktext=labels)
    return fig



# Callback para actualización automática
@app.callback(
    [Output('dataset-info-display', 'children', allow_duplicate=True),
     Output('dataset-metrics', 'children', allow_duplicate=True),
     Output('visualizations-container', 'children', allow_duplicate=True),
     Output('stream-status', 'children', allow_duplicate=True),
     Output('app-url', 'href', allow_duplicate=True)],   # <<--- NUEVO
    [Input('auto-interval', 'n_intervals')],
    prevent_initial_call=True
)
def update_auto_data(_n_intervals):
    """Actualiza datos automáticamente en modo tiempo real"""
    if not data_generator.auto_mode_active:
        st = read_stream_status(False)
        badge = render_stream_status_badge(st["status"] == "running", st.get("updated_at"))
        raise dash.exceptions.PreventUpdate
    
    # Generar nuevo batch de datos automáticamente
    data_generator.generate_auto_batch()
    
    # Si el generador acaba de reiniciar (10k), fuerza reload del dashboard
    if getattr(data_generator, 'just_restarted', False):
        # Limpia el flag (por si el usuario cancela el reload con el navegador)
        data_generator.just_restarted = False
        # Dispara un hard refresh cambiando el href (mismo path + cache buster)
        reload_url = f"/?r={int(time.time())}"
        # Devuelve algo mínimo junto con el reload
        st = read_stream_status(True if data_generator.auto_mode_active else False)
        badge = render_stream_status_badge(st["status"] == "running", st.get("updated_at"))
        return (no_update, no_update, no_update, badge, reload_url)
    
    df = data_generator.get_current_data()
    
    # Obtener datos actualizados
    df = data_generator.get_current_data()
    dataset_info = data_generator.get_dataset_info()
    if df.empty:
        st = read_stream_status(True)
        badge = render_stream_status_badge(st["status"] == "running", st.get("updated_at"))
        return [html.Div("Generando datos automáticamente...", style={'color': '#b0b0b0'})], [], [], badge, no_update

    
    # Información del dataset
    info_items = [
        html.Div([
            html.Span("Tipo:", className='info-label'),
            html.Span(dataset_info.get('type', 'N/A'), className='info-value')
        ], className='info-item'),
        html.Div([
            html.Span("Balanceado:", className='info-label'),
            html.Span("Sí" if data_generator.balance_classes else "No", className='info-value')
        ], className='info-item'),
        html.Div([
            html.Span("Muestras:", className='info-label'),
            html.Span(str(dataset_info.get('samples', 0)), className='info-value')
        ], className='info-item'),
        html.Div([
            html.Span("Features:", className='info-label'),
            html.Span(str(dataset_info.get('features', 0)), className='info-value')
        ], className='info-item'),
        html.Div([
            html.Span("Clases:", className='info-label'),
            html.Span(str(dataset_info.get('classes', 0)), className='info-value')
        ], className='info-item'),
        html.Div([
            html.Span("Generador:", className='info-label'),
            html.Span(dataset_info.get('generator', 'N/A'), className='info-value')
        ], className='info-item'),
        html.Div([
            html.Span("Actualizado:", className='info-label'),
            html.Span(datetime.now().strftime("%H:%M:%S"), className='info-value')
        ], className='info-item')
    ]
    
    info_display = html.Div([
        html.H3("Información del Dataset (Automático)", style={'color': '#51cf66', 'margin-bottom': '15px'}),
        html.Div(info_items)
    ], className='dataset-info')
    
    # Métricas del dataset
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Mostrar muestras con formato apropiado
    sample_display = f"{len(df):,}" if len(df) < 100000 else f"{len(df)/1000:.0f}K"
    if len(df) >= 1000000:
        sample_display = f"{len(df)/1000000:.1f}M"
    
    metrics = html.Div([
        html.Div([
            html.Div([
                html.P(sample_display, className="metric-value"),
                html.P("MUESTRAS BUFFER", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
            html.Div([
                html.P(str(len(feature_cols)), className="metric-value"),
                html.P("FEATURES", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
            html.Div([
                html.P(f"{df['target'].mean():.1%}" if 'target' in df.columns else "N/A", className="metric-value"),
                html.P("BALANCE ACTUAL", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
            html.Div([
                html.P("AUTO", className="metric-value", style={
                    'fontSize': '2.2em',
                    'wordBreak': 'break-word',
                    'lineHeight': '1.1'
                }),
                html.P("MODO AUTOMÁTICO", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
        ], style={'display': 'flex', 'justify-content': 'space-around', 'margin': '20px 0'})
    ])
    
    # Generar visualizaciones
    visualizations = create_all_visualizations(df)
    st = read_stream_status(True if data_generator.auto_mode_active else False)
    badge = render_stream_status_badge(st["status"] == "running", st.get("updated_at"))

    return info_display, metrics, visualizations, badge, no_update

@app.callback(
    [Output('seed-input', 'disabled'),
     Output('apply-seed', 'disabled')],
    [Input('auto-interval', 'n_intervals'),
     Input('start-stream', 'n_clicks'),
     Input('stop-stream', 'n_clicks')],
    prevent_initial_call=False
)
def sync_seed_controls(_n, _s, _p):
    # True => deshabilitado si el stream está activo
    is_active = bool(getattr(data_generator, 'auto_mode_active', False))
    return is_active, is_active


@app.callback(
    [Output('dataset-info-display', 'children', allow_duplicate=True),
     Output('dataset-metrics', 'children', allow_duplicate=True),
     Output('visualizations-container', 'children', allow_duplicate=True),
     Output('stream-status', 'children', allow_duplicate=True)],
    [Input('apply-seed', 'n_clicks')],
    [State('seed-input', 'value')],
    prevent_initial_call=True
)
def apply_seed(n_clicks, new_seed):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # Si el stream está activo, NO permitimos cambiar la semilla
    if data_generator.auto_mode_active:
        # Simplemente no actualizamos nada
        raise dash.exceptions.PreventUpdate

    # Stream parado: hacemos soft reset con nueva semilla
    data_generator.set_seed_and_soft_reset(new_seed, cf_idx=0)
    df = data_generator.get_current_data()

    info_display = html.Div([
        html.H3("Información del Dataset (Semilla aplicada)", style={'color': '#51cf66'}),
        html.Div([
            html.Div([html.Span("Semilla:", className='info-label'),
                      html.Span(str(data_generator.current_seed), className='info-value')], className='info-item'),
            html.Div([html.Span("Muestras:", className='info-label'),
                      html.Span(str(len(df)), className='info-value')], className='info-item'),
            html.Div([html.Span("Generador:", className='info-label'),
                      html.Span(data_generator.dataset_info.get('generator', 'agrawal'), className='info-value')], className='info-item'),
        ])
    ], className='dataset-info')

    metrics = html.Div("🎲 Semilla aplicada con éxito (stream detenido). Pulsa INICIAR STREAM para empezar.",
                       className="metric-card")

    visualizations = create_all_visualizations(df)

    st = read_stream_status(False)  # stream sigue parado
    badge = render_stream_status_badge(st["status"] == "running", st.get("updated_at"))
    return info_display, metrics, visualizations, badge



def reconfigure_auto_generator(generator_type, noise):
    n_features = 4  # valor fijo por defecto

    # Si el stream está ACTIVO, reconfiguramos "en caliente" y reiniciamos el contador
    if data_generator.auto_mode_active and generator_type and noise is not None and data_generator.current_seed:
        data_generator.reconfigure_generator(generator_type, n_features, noise, data_generator.current_seed)
        return 0  # reinicia el contador del intervalo

    # Si el stream está PARADO, solo guardamos preferencias y NO tocamos el intervalo ni activamos el generador
    data_generator.last_generator_type = generator_type or data_generator.last_generator_type
    data_generator.last_noise = noise if noise is not None else data_generator.last_noise
    data_generator.last_n_features = n_features
    return no_update


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8051)
