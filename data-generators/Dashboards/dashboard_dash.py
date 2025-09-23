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
from dash import no_update
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

# Import Synthetic generators
from Synthetic.SyntheticGenerator import SyntheticGenerator
from Synthetic.SyntheticBlockGenerator import SyntheticBlockGenerator
from Synthetic.GeneratorFactory import GeneratorFactory, GeneratorType, GeneratorConfig

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
    """Generador de datasets usando infraestructura Synthetic de CALMOPS"""
    
    def __init__(self):
        self.max_samples = 10000    # umbral de reinicio
        self.current_seed = 42            # semilla actual
        self.last_generator_type = 'agrawal'
        self.last_n_features = 4
        self.last_noise = 0.1
        self.balance_classes = True 
        self.current_data = pd.DataFrame()
        self.dataset_info = {}
        self.temp_dir = './plots'
        os.makedirs(self.temp_dir, exist_ok=True)

        # --- Auto mode (streaming) ---
        self.auto_mode_active = False
        self.auto_generator = None
        self.auto_batch_size = 100
        
        # Tipos de generadores River disponibles
        self.generator_types = {
            'agrawal': 'Agrawal (9 features)',
            'sea': 'SEA Concepts (3 features)',
            'stagger': 'STAGGER (3 features)',
            'random_tree': 'Random Tree',
        }
        
    def generate_standard_dataset(self, n_samples=500, generator_type='sea', 
                                  n_features=4, class_sep=0.8, noise=0.1, random_state=42):
        """Genera dataset estándar usando SyntheticGenerator"""
        try:
            print(f"🎯 Generando dataset estándar: {n_samples} muestras con {generator_type}")
            
            syn_gen = SyntheticGenerator(enable_auto_visualization=False)
            method_params = self._prepare_method_params(generator_type, n_features, noise, class_sep)
            
            filename = f"dataset_{generator_type}_{random_state}.csv"
            syn_gen.generate(
                output_path=self.temp_dir,
                filename=filename,
                n_samples=n_samples,
                method=generator_type,
                method_params=method_params,
                drift_type="none",
                random_state=random_state
            )
            
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                self.current_data = self._process_generated_data(df, generator_type)
                
                factory = GeneratorFactory()
                try:
                    gen_type_enum = getattr(GeneratorType, generator_type.upper())
                    gen_info = factory.get_generator_info(gen_type_enum)
                except AttributeError:
                    gen_info = {'description': f'Generador {generator_type}'}
                
                self.dataset_info = {
                    'type': 'Estándar',
                    'generator': self.generator_types.get(generator_type, generator_type),
                    'samples': len(self.current_data),
                    'features': len([col for col in self.current_data.columns if col not in ['target']]),
                    'classes': len(self.current_data['target'].unique()) if 'target' in self.current_data.columns else 0,
                    'description': gen_info.get('description', 'Generador River'),
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return True
            else:
                print(f"Error: Archivo {file_path} no encontrado")
                return False
        except Exception as e:
            print(f"Error generando dataset estándar: {e}")
            return False
    
    def _rebalance_dataframe(self, df: pd.DataFrame, target_col: str = "target",
                         strategy: str = "downsample", random_state: int = 42) -> pd.DataFrame:
        """
        Rebalance a dataframe per-batch keeping class proportions equal.
        strategy: 'downsample' (safe for streams) or 'upsample' (may duplicate).
        """
        if target_col not in df.columns:
            return df

        vc = df[target_col].value_counts(dropna=False)
        if len(vc) <= 1:
            return df  # nothing to do

        min_count = int(vc.min())
        max_count = int(vc.max())
        if min_count == 0:
            logger.warning("One class has zero instances in the batch; skipping rebalance.")
            return df

        rng = np.random.default_rng(random_state)
        parts = []

        for cls, g in df.groupby(target_col):
            if strategy == "downsample":
                n = min(len(g), min_count)
                parts.append(g.sample(n=n, replace=False, random_state=random_state))
            elif strategy == "upsample":
                n = max_count
                if len(g) == n:
                    parts.append(g)
                else:
                    reps = int(np.ceil(n / len(g)))
                    g_rep = pd.concat([g] * reps, ignore_index=True).iloc[:n]
                    parts.append(g_rep)
            else:
                return df  # unknown strategy, skip

        out = pd.concat(parts, ignore_index=True)
        out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)  # shuffle
        return out

    def generate_block_dataset(self, n_samples=500, n_blocks=5, generator_type='sea',
                               n_features=4, class_sep=0.8, noise=0.1, random_state=42):
        """Genera dataset dividido en bloques usando SyntheticBlockGenerator"""
        try:
            print(f"📊 Generando dataset por bloques: {n_samples} muestras en {n_blocks} bloques con {generator_type}")
            
            block_gen = SyntheticBlockGenerator()
            method_params = self._prepare_method_params(generator_type, n_features, noise, class_sep)
            filename = f"dataset_blocks_{generator_type}_{random_state}.csv"
            
            methods = [generator_type] * n_blocks
            method_params_list = [method_params.copy() for _ in range(n_blocks)]
            
            block_gen.generate_blocks_simple(
                output_path=self.temp_dir,
                filename=filename,
                n_blocks=n_blocks,
                total_samples=n_samples,
                methods=methods,
                method_params=method_params_list,
                random_state=random_state
            )
            
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                self.current_data = self._process_generated_data(df, generator_type, has_blocks=True)
                
                factory = GeneratorFactory()
                try:
                    gen_type_enum = getattr(GeneratorType, generator_type.upper())
                    gen_info = factory.get_generator_info(gen_type_enum)
                except AttributeError:
                    gen_info = {'description': f'Generador {generator_type}'}
                
                self.dataset_info = {
                    'type': 'Por Bloques',
                    'generator': self.generator_types.get(generator_type, generator_type),
                    'samples': len(self.current_data),
                    'features': len([col for col in self.current_data.columns if col not in ['target', 'block']]),
                    'classes': len(self.current_data['target'].unique()) if 'target' in self.current_data.columns else 0,
                    'blocks': n_blocks,
                    'description': gen_info.get('description', 'Generador River'),
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return True
            else:
                print(f"Error: Archivo {file_path} no encontrado")
                return False
        except Exception as e:
            print(f"Error generando dataset por bloques: {e}")
            return False
    
    def _prepare_method_params(self, generator_type, n_features, noise, _class_sep):
        """Prepara parámetros específicos para cada tipo de generador River"""
        params = {}
        if generator_type == 'sea':
            params = {'function': 0, 'noise_percentage': noise}
        elif generator_type == 'agrawal':
            params = {'classification_function': 0, 'balance_classes': self.balance_classes, 'perturbation': noise}
        elif generator_type == 'hyperplane':
            params = {'n_features': n_features, 'mag_change': 0.0, 'sigma': noise}
        elif generator_type == 'sine':
            params = {'has_noise': noise > 0, 'noise_percentage': noise if noise > 0 else 0.1}
        elif generator_type == 'stagger':
            params = {'classification_function_stagger': 0, 'balance_classes_stagger': self.balance_classes}
        elif generator_type == 'random_tree':
            params = {'n_num_features': min(n_features, 5), 'n_cat_features': 0,
                      'n_categories_per_cat_feature': 2, 'max_tree_depth': 5}
        elif generator_type == 'mixed':
            params = {'classification_function_mixed': 0, 'balance_classes_mixed': self.balance_classes}
        elif generator_type == 'friedman':
            params = {'n_features_friedman': n_features}
        elif generator_type == 'random_rbf':
            params = {'n_features_rbf': n_features, 'n_centroids': min(50, n_features * 5)}
        return params
    
    def _process_generated_data(self, df, _generator_type, _has_blocks=False):
        """Procesa datos generados para adaptarlos al formato del dashboard"""
        processed_df = df.copy()
        if 'timestamp' not in processed_df.columns:
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
    
    def _setup_real_time_generator(self, generator_type, n_features, noise, random_state):
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
        print(f"✅ Auto generator configurado: {generator_type}")
        return True

    def generate_auto_batch(self):
        """Genera y agrega un batch de datos en modo automático."""
        if not self.auto_mode_active or self.auto_generator is None:
           print("Auto generator is not active")
           return self.current_data

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

        # Identidad temporal/secuencial
        start_id = len(self.current_data)
        df_batch["sample_id"] = range(start_id, start_id + len(df_batch))
        df_batch["timestamp"] = pd.date_range(start=datetime.now(), periods=len(df_batch), freq="1s")

        # --- Balanceo fallback solo si el generador no es nativamente balanceado ---
        _native_balanced = self.last_generator_type in {'agrawal', 'stagger', 'mixed'}
        if self.balance_classes and not _native_balanced and 'target' in df_batch.columns:
            before = df_batch['target'].value_counts().to_dict()
            df_batch = self._rebalance_dataframe(
                df_batch, target_col="target", strategy="downsample",
                random_state=self.current_seed or 42
            )
            after = df_batch['target'].value_counts().to_dict()
            print(f"Rebalanced batch (downsample). Before={before} After={after}")

        # Append al buffer
        self.current_data = pd.concat([self.current_data, df_batch], ignore_index=True)
        self.dataset_info["samples"] = len(self.current_data)
        self.dataset_info["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Auto batch appended: {len(df_batch)} samples (total={len(self.current_data)})")

        # --- Chequear umbral y reiniciar si corresponde ---
        if len(self.current_data) >= self.max_samples:
            new_seed = (self.current_seed or 0) + 1
            print(f"Limit of {self.max_samples:,} reached. Restarting stream with seed {new_seed}...")

            # Reconfigurar generador con nueva semilla
            self._setup_real_time_generator(
                self.last_generator_type,
                self.last_n_features,
                self.last_noise,
                new_seed
            )

            # Limpiar buffer y sembrar primer batch del nuevo ciclo (SIN consumo doble)
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
                    df_batch2["timestamp"] = pd.date_range(start=datetime.now(), periods=len(df_batch2), freq="1s")
                    self.current_data = df_batch2
                    self.dataset_info["samples"] = len(self.current_data)
                    self.dataset_info["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except StopIteration:
                print("Stream returned no samples after restart")

            # Actualiza semilla actual
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
    html.P("Dashboard", className="subtitle"),
    
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
        
        # Configuración básica
        html.Div([
            # Tipo de generador
            html.Div([
                html.Label("Generador River:", style={'color': '#b0b0b0', 'fontSize': '16px',
                                                      'margin-bottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='generator-type-dropdown',
                    options=[
                        {'label': '🌊 SEA Concepts (3 features)', 'value': 'sea'},
                        {'label': '🧬 Agrawal (9 features)', 'value': 'agrawal'},
                        {'label': '📊 STAGGER (3 features)', 'value': 'stagger'},
                        {'label': '🌳 Random Tree', 'value': 'random_tree'}
                    ],
                    value='agrawal',
                    style={
                        'backgroundColor': '#1e1e1e',
                        'color': '#ffffff',
                        'border': '1px solid #00d4ff',
                        'borderRadius': '8px'
                    },
                    className='custom-dropdown'
                )
            ], style={'width': '48%', 'display': 'inline-block', 'margin-right': '4%'}),
        ], style={'margin-bottom': '20px'}),
        
        # Parámetros avanzados
        html.Div([
            html.Div([
                html.Label("Nivel de Ruido:", style={'color': '#b0b0b0',
                                                     'fontSize': '16px',
                                                     'margin-bottom': '10px',
                                                     'display': 'block'}),
                dcc.Slider(
                    id='noise-slider',
                    min=0.0,
                    max=0.5,
                    step=0.05,
                    value=0.1,
                    marks={i/20: f'{i/20:.2f}' for i in range(0, 11, 2)},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ], style={'width': '48%', 'display': 'inline-block'})
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label("Balance de clases:", style={'color': '#b0b0b0','fontSize': '16px','margin-bottom': '10px','display': 'block'}),
            dcc.RadioItems(
                id='balance-toggle',
                options=[
                    {'label': 'Balanceado (igualar clases)', 'value': 'balanced'},
                    {'label': 'No balanceado', 'value': 'unbalanced'},
                ],
                value='balanced',
                inputStyle={'margin-right': '6px', 'margin-left': '12px'},
                labelStyle={'display': 'inline-block', 'margin-right': '18px', 'color': '#ffffff'}
            ),
        ], style={'margin-bottom': '20px'}),
        # Semilla
        html.Div([
        html.Div([
            html.Button([
                html.Div("▶️", style={'fontSize': '24px', 'marginBottom': '5px'}),
                html.Div("INICIAR STREAM", style={'fontSize': '14px',
                                                'fontWeight': 'bold',
                                                'letterSpacing': '1px'})
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
            html.Div([
                html.Button([
                    html.Div("⏹️", style={'fontSize': '24px', 'marginBottom': '5px'}),
                    html.Div("DETENER STREAM", style={'fontSize': '14px',
                                                      'fontWeight': 'bold',
                                                      'letterSpacing': '1px'})
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
            ], style={'text-align': 'center'})
        ], style={'marginTop': '30px'})
    ], className='control-panel'),
    
    # Información del dataset
    html.Div(id='dataset-info-display'),
    
    # Métricas del dataset
    html.Div(id='dataset-metrics'),
    
    # Título de visualizaciones
    html.H2("ANÁLISIS VISUAL INTERACTIVO", className="section-title"),
    
    # Grid de visualizaciones
    html.Div(id='visualizations-container'),
    
    # Intervalo para actualización automática
    dcc.Interval(
        id='auto-interval',
        interval=5000,  # 5 segundos
        n_intervals=0,
        disabled=True
      
    )
    
], style={'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
          'min-height': '100vh', 'padding': '20px', 'color': '#ffffff'})
])

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
     Output('balance-toggle', 'disabled')],   # ⬅️ nuevo
    [Input('start-stream', 'n_clicks'),
     Input('stop-stream', 'n_clicks')],
    [State('generator-type-dropdown', 'value'),
     State('noise-slider', 'value'),
     State('balance-toggle', 'value')],       # ⬅️ nuevo
    prevent_initial_call=True
)
def toggle_stream(n_start, n_stop, generator_type, noise, balance_value):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'stop-stream':
        data_generator.auto_mode_active = False
        return True, dash.no_update, dash.no_update, dash.no_update, False

    # start-stream
    data_generator.balance_classes = (balance_value == 'balanced')
    data_generator.reconfigure_generator(generator_type, 4, noise, data_generator.current_seed)
    data_generator.auto_mode_active = True

    df = data_generator.generate_auto_batch()
    if df.empty:
        info_display = html.Div("⚠️ No se pudo generar datos iniciales", style={'color': 'red'})
        return False, info_display, [], [], True

    info_display = html.Div([
        html.H3("Información del Dataset (Stream)", style={'color': '#51cf66'}),
        html.Div([
            html.Div([html.Span("Generador:"), html.Span(generator_type)]),
            html.Div([html.Span("Balanceado:"), html.Span("Sí" if data_generator.balance_classes else "No")]),
            html.Div([html.Span("Muestras iniciales:"), html.Span(str(len(df)))]),
        ])
    ], className='dataset-info')

    metrics = html.Div("⚡ Stream inicializado", className="metric-card")
    visualizations = create_all_visualizations(df)

    return False, info_display, metrics, visualizations, True


def generate_dataset_and_update(n_standard, n_blocks, sample_mode, n_samples, generator_type, 
                                n_features, noise, random_seed, n_block_count):
    """Genera dataset y actualiza visualizaciones usando Synthetic"""
    ctx = callback_context
    if not ctx.triggered:
        return [], [], []
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Determinar número de muestras
    actual_samples = 1000000 
    
    try:
        print(f"\\n🚀 INICIANDO GENERACIÓN - Botón: {button_id}")
        print(f"📊 Parámetros: {actual_samples} muestras, generador: {generator_type}")
        
        success = False
        if button_id == 'generate-standard' and n_standard:
            success = data_generator.generate_standard_dataset(
                n_samples=actual_samples,
                generator_type=generator_type,
                n_features=n_features,
                class_sep=0.8,
                noise=noise,
                random_state=random_seed
            )
        elif button_id == 'generate-blocks' and n_blocks:
            success = data_generator.generate_block_dataset(
                n_samples=actual_samples,
                n_blocks=n_block_count,
                generator_type=generator_type,
                n_features=n_features,
                class_sep=0.8,
                noise=noise,
                random_state=random_seed
            )
        
        # Reconfigurar generador automático con nuevos parámetros
        data_generator.reconfigure_generator(generator_type, n_features, noise, random_seed)
        data_generator.auto_mode_active = True
        if not success:
            print("❌ Error en la generación del dataset")
            return [html.Div("Error generando dataset", style={'color': 'red'})], [], []
        
        print("✅ Dataset generado correctamente, creando visualizaciones...")
        
    except Exception as e:
        print(f"❌ Error en callback: {e}")
        import traceback
        traceback.print_exc()
        return [html.Div(f"Error: {str(e)}", style={'color': 'red'})], [], []
    
    # Obtener datos
    df = data_generator.get_current_data()
    dataset_info = data_generator.get_dataset_info()
    
    # Si no hay datos generados manualmente, usar datos automáticos
    if df.empty and data_generator.auto_mode_active:
        # Generar un batch inicial para mostrar algo
        data_generator.generate_auto_batch()
        df = data_generator.get_current_data()
        dataset_info = data_generator.get_dataset_info()
    
    if df.empty:
        return [html.Div("No hay datos disponibles", style={'color': '#b0b0b0'})], [], []
    
    # Información del dataset
    info_items = [
        html.Div([
            html.Span("Tipo:", className='info-label'),
            html.Span(dataset_info.get('type', 'N/A'), className='info-value')
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
            html.Span("Generado:", className='info-label'),
            html.Span(dataset_info.get('generated_at', 'N/A'), className='info-value')
        ], className='info-item')
    ]
    
    # Agregar información de bloques si aplica
    if 'blocks' in dataset_info:
        info_items.insert(-1, html.Div([
            html.Span("Bloques:", className='info-label'),
            html.Span(str(dataset_info.get('blocks', 0)), className='info-value')
        ], className='info-item'))
    
    
    info_display = html.Div([
        html.H3("Información del Dataset", style={'color': '#51cf66', 'margin-bottom': '15px'}),
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
                html.P("MUESTRAS TOTALES", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
            html.Div([
                html.P(str(len(feature_cols)), className="metric-value"),
                html.P("FEATURES", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
            html.Div([
                html.P(f"{df['target'].mean():.1%}" if 'target' in df.columns else "N/A", className="metric-value"),
                html.P("BALANCE PROMEDIO", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
            html.Div([
                html.P(generator_type.upper(), className="metric-value", style={
                    'fontSize': '1.8em' if len(generator_type) > 8 else '2.5em',
                    'wordBreak': 'break-word',
                    'lineHeight': '1.1'
                }),
                html.P("GENERADOR RIVER", className="metric-label")
            ], className="metric-card", style={'flex': '1'}),
        ], style={'display': 'flex', 'justify-content': 'space-around', 'margin': '20px 0'})
    ])
    
    # Generar visualizaciones (muestrear si es muy grande)
    df_for_viz = df.sample(min(5000, len(df))) if len(df) > 5000 else df
    visualizations = create_all_visualizations(df_for_viz)
    
    return info_display, metrics, visualizations

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
        print("[DEBUG] df está vacío en sección muestra de datos")
        table = html.Div("⚠️ No hay datos generados todavía", 
                        style={'color': '#b0b0b0', 'text-align': 'center', 'margin': '20px'})
    else:
        sample_size = min(10, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)

        print(f"[DEBUG] Mostrando {sample_size} instancias de muestra")
        print(df_sample.head())  # para confirmar que se seleccionan filas

        table = html.Div([
            html.P(f"Mostrando {sample_size} instancias aleatorias", 
                style={'text-align': 'center', 'color': '#b0b0b0', 'margin-bottom': '20px'}),
            create_data_table(df_sample)
        ])

    cat_fig = create_categorical_mode_figure(df)
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
    """Crea evolución por bloques para datos River"""
    if 'block' not in df.columns:
        return go.Figure()
    
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols][:4]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Feature {col}" for col in feature_cols]
    )
    
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['success'], COLOR_PALETTE['warning']]
    
    for i, (feature, color) in enumerate(zip(feature_cols, colors)):
        row = i // 2 + 1
        col = i % 2 + 1
        
        block_means = df.groupby('block')[feature].mean()
        
        fig.add_trace(
            go.Scatter(
                x=block_means.index,
                y=block_means.values,
                mode='lines+markers',
                name=feature,
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{feature}</b><br>Bloque: %{{x}}<br>Media: %{{y:.3f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=400, showlegend=False)
    fig = apply_dark_theme(fig, "EVOLUCIÓN POR BLOQUES (River)")
    return fig

def create_evolution_by_instances_figure(df):
    """Crea evolución por instancias para datos River"""
    if 'sample_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'sample_id'})
    
    excluded_cols = ['target', 'block', 'timestamp', 'sample_id']
    feature_cols = [col for col in df.columns if col not in excluded_cols][:4]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Feature {col}" for col in feature_cols]
    )
    
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['success'], COLOR_PALETTE['warning']]
    
    # Tomar muestra para rendimiento
    df_sample = df.sample(min(500, len(df))).sort_values('sample_id') if len(df) > 500 else df.sort_values('sample_id')
    
    for i, (feature, color) in enumerate(zip(feature_cols, colors)):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Scatter(
                x=df_sample['sample_id'],
                y=df_sample[feature],
                mode='lines',
                name=feature,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{feature}</b><br>Instancia: %{{x}}<br>Valor: %{{y:.3f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=400, showlegend=False)
    fig = apply_dark_theme(fig, "EVOLUCIÓN POR INSTANCIAS (River)")
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




# Callback para actualización automática
@app.callback(
    [Output('dataset-info-display', 'children', allow_duplicate=True),
     Output('dataset-metrics', 'children', allow_duplicate=True),
     Output('visualizations-container', 'children', allow_duplicate=True)],
    [Input('auto-interval', 'n_intervals')],
    prevent_initial_call=True
)
def update_auto_data(_n_intervals):
    """Actualiza datos automáticamente en modo tiempo real"""
    if not data_generator.auto_mode_active:
        raise dash.exceptions.PreventUpdate
    
    # Generar nuevo batch de datos automáticamente
    data_generator.generate_auto_batch()
    
    # Obtener datos actualizados
    df = data_generator.get_current_data()
    dataset_info = data_generator.get_dataset_info()
    
    if df.empty:
        return [html.Div("Generando datos automáticamente...", style={'color': '#b0b0b0'})], [], []
    
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
    
    return info_display, metrics, visualizations

# Callback para reconfigurar generador automático

@app.callback(
    Output('auto-interval', 'n_intervals'),
    [Input('generator-type-dropdown', 'value'),
     Input('noise-slider', 'value')],
    prevent_initial_call=True
)
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