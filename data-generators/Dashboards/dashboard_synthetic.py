#!/usr/bin/env python3
"""
CALMOPS - Synthetic Viewer (Informational, Read-only)
=====================================================

- Does NOT generate data.
- Does NOT accept directory paths.
- Always reads from the fixed folder: 'salida_tiempo_real/'.
- Rebuilds interactive plots from the latest CSV found there.
- Reads 'report.json' from the same folder and shows a compact summary.

UI language: English.
"""

import os
import json
import glob
import time
import socket
import webbrowser
from contextlib import closing

import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Fixed location (no routes passed)
# -----------------------------
BASE_DIR = "salida_tiempo_real"
REPORT_FILE = "report.json"

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "CALMOPS Synthetic Viewer"

# -----------------------------
# Helpers
# -----------------------------
def latest_csv(base_dir: str) -> str | None:
    """Return path to the newest CSV in base_dir (non-recursive)."""
    pattern = os.path.join(base_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def load_df() -> pd.DataFrame | None:
    """Load the most recent CSV from BASE_DIR."""
    if not os.path.isdir(BASE_DIR):
        return None
    path = latest_csv(BASE_DIR)
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_report() -> dict | None:
    path = os.path.join(BASE_DIR, REPORT_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def info_row(label: str, value: str):
    return html.Div([
        html.Span(label, className="info-label"),
        html.Span(value, className="info-value"),
    ], className="info-item")

def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"target"}]

# -----------------------------
# Plot builders (dark theme)
# -----------------------------
DARK = {
    "plot_bgcolor": "#1e1e1e",
    "paper_bgcolor": "#1e1e1e",
    "font_color": "#ffffff",
    "grid_color": "#404040",
    "title_color": "#00d4ff",
}

def apply_dark(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=DARK["plot_bgcolor"],
        paper_bgcolor=DARK["paper_bgcolor"],
        font_color=DARK["font_color"],
        title=dict(text=title, font=dict(size=18, color=DARK["title_color"]), x=0.5),
        xaxis=dict(gridcolor=DARK["grid_color"]),
        yaxis=dict(gridcolor=DARK["grid_color"]),
        legend=dict(font=dict(color=DARK["font_color"]), bgcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

def fig_target_distribution(df: pd.DataFrame) -> go.Figure:
    if "target" not in df.columns:
        return apply_dark(go.Figure(), "Target Distribution (missing 'target')")
    vc = df["target"].value_counts().sort_index()
    fig = go.Figure([go.Bar(x=[str(i) for i in vc.index], y=vc.values, text=vc.values, textposition="auto")])
    return apply_dark(fig, "Target Distribution")

def fig_correlation(df: pd.DataFrame) -> go.Figure:
    feats = numeric_feature_cols(df)
    if len(feats) < 2:
        return apply_dark(go.Figure(), "Correlation Matrix (need ≥2 numeric features)")
    corr = df[feats].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 3), texttemplate="%{text}", textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r=%{z:.3f}<extra></extra>",
    ))
    return apply_dark(fig, "Correlation Matrix")

def fig_pca(df: pd.DataFrame) -> go.Figure:
    feats = numeric_feature_cols(df)
    if len(feats) < 2:
        return apply_dark(go.Figure(), "PCA Projection (need ≥2 numeric features)")
    sample = df[feats + (["target"] if "target" in df.columns else [])].copy()
    if len(sample) > 3000:
        sample = sample.sample(3000, random_state=42)
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        X = StandardScaler().fit_transform(sample[feats].values)
        Xp = PCA(n_components=2).fit_transform(X)
        fig = px.scatter(
            x=Xp[:, 0], y=Xp[:, 1],
            color=sample["target"].astype(str) if "target" in sample.columns else None,
            labels={"x": "PC1", "y": "PC2"},
            title="PCA Projection"
        )
        return apply_dark(fig, "PCA Projection")
    except Exception:
        return apply_dark(go.Figure(), "PCA Projection (sklearn not available)")

def fig_feature_hist(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, color=("target" if "target" in df.columns else None), marginal="box")
    return apply_dark(fig, f"Distribution of {col}")

def fig_box(df: pd.DataFrame) -> go.Figure:
    feats = numeric_feature_cols(df)[:6]
    if not feats:
        return apply_dark(go.Figure(), "Box Plots (no numeric features)")
    fig = go.Figure()
    for c in feats:
        fig.add_trace(go.Box(y=df[c], name=c, boxpoints="outliers"))
    return apply_dark(fig, "Box Plots (Numeric Features)")

def fig_violin(df: pd.DataFrame) -> go.Figure:
    feats = numeric_feature_cols(df)[:6]
    if not feats:
        return apply_dark(go.Figure(), "Violin Plots (no numeric features)")
    fig = go.Figure()
    for c in feats:
        fig.add_trace(go.Violin(y=df[c], name=c, box_visible=True, meanline_visible=True))
    return apply_dark(fig, "Violin Plots (Numeric Features)")

def fig_scatter_matrix(df: pd.DataFrame) -> go.Figure:
    feats = numeric_feature_cols(df)[:4]
    if len(feats) < 2:
        return apply_dark(go.Figure(), "Scatter Matrix (need ≥2 numeric features)")
    df_sample = df.sample(min(400, len(df)), random_state=42) if len(df) > 400 else df
    fig = px.scatter_matrix(
        df_sample,
        dimensions=feats,
        color=("target" if "target" in df_sample.columns else None),
        title="Scatter Matrix"
    )
    return apply_dark(fig, "Scatter Matrix")

def fig_top_correlations(df: pd.DataFrame, top_k: int = 10) -> go.Figure:
    feats = numeric_feature_cols(df)
    if len(feats) < 2:
        return apply_dark(go.Figure(), "Top Correlations (need ≥2 numeric features)")
    corr = df[feats].corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((f"{cols[i]} vs {cols[j]}", corr.iloc[i, j]))
    if not pairs:
        return apply_dark(go.Figure(), "Top Correlations")
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:top_k]
    fig = go.Figure(go.Bar(x=[p[0] for p in pairs], y=[p[1] for p in pairs], text=[f"{p[1]:.3f}" for p in pairs], textposition="auto"))
    return apply_dark(fig, "Top Correlations (|r|)")

def fig_block_sizes(df: pd.DataFrame) -> go.Figure:
    if "block" not in df.columns:
        return apply_dark(go.Figure(), "Block Sizes (no 'block' column)")
    vc = df["block"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=[str(i) for i in vc.index], y=vc.values, text=vc.values, textposition="auto"))
    return apply_dark(fig, "Block Sizes")

# -----------------------------
# Build static content once (no auto-refresh, no last-updated, no base folder shown)
# -----------------------------
_df = load_df()
if _df is None or _df.empty:
    _summary = html.Div([
        html.Div([info_row("Status", "No data found")], className="dataset-info"),
        html.Div("Place a CSV in 'salida_tiempo_real/'.", className="empty-note")
    ])
    _fig_target = apply_dark(go.Figure(), "No data")
    _fig_corr = apply_dark(go.Figure(), "No data")
    _fig_pca = apply_dark(go.Figure(), "No data")
    _fig_h0 = apply_dark(go.Figure(), "No data")
    _fig_h1 = apply_dark(go.Figure(), "No data")
    _fig_h2 = apply_dark(go.Figure(), "No data")
    _fig_box = apply_dark(go.Figure(), "No data")
    _fig_violin = apply_dark(go.Figure(), "No data")
    _fig_scatter = apply_dark(go.Figure(), "No data")
    _fig_topcorr = apply_dark(go.Figure(), "No data")
    _fig_blocks = apply_dark(go.Figure(), "No data")
    _random_table = html.Div("No random instances (no data).", className="empty-note")
    _report_block = html.Div("No report.json found.", className="empty-note")
else:
    n_rows, n_cols = _df.shape
    facts = [
        info_row("Rows", f"{n_rows:,}"),
        info_row("Columns", str(n_cols)),
        info_row("Has target", "Yes" if "target" in _df.columns else "No"),
    ]
    # Random instances (10 rows)
    df_rand = _df.sample(min(10, len(_df)), random_state=42).reset_index(drop=True)
    def make_table(df_show: pd.DataFrame):
        return html.Table([
            html.Thead(html.Tr([html.Th(c) for c in df_show.columns])),
            html.Tbody([html.Tr([html.Td(df_show.iloc[i][c]) for c in df_show.columns]) for i in range(len(df_show))])
        ], style={"width": "100%", "borderCollapse": "collapse"})

    head = _df.head(5)
    _table_preview = make_table(head)
    _random_table = html.Div([
        html.Div("Random sample (10 rows)", className="info-label", style={"margin": "8px 0"}),
        make_table(df_rand)
    ], className="panel")

    _summary = html.Div([
        html.Div(facts, className="dataset-info"),
        html.Div([html.Div("Preview (top 5 rows)", className="info-label", style={"margin": "8px 0"}), _table_preview], className="panel")
    ])

    _fig_target = fig_target_distribution(_df)
    _fig_corr = fig_correlation(_df)
    _fig_pca = fig_pca(_df)
    _feats = numeric_feature_cols(_df)
    _fig_h0 = fig_feature_hist(_df, _feats[0]) if _feats else apply_dark(go.Figure(), "No numeric features")
    _fig_h1 = fig_feature_hist(_df, _feats[1]) if len(_feats) > 1 else apply_dark(go.Figure(), "—")
    _fig_h2 = fig_feature_hist(_df, _feats[2]) if len(_feats) > 2 else apply_dark(go.Figure(), "—")
    _fig_box = fig_box(_df)
    _fig_violin = fig_violin(_df)
    _fig_scatter = fig_scatter_matrix(_df)
    _fig_topcorr = fig_top_correlations(_df, top_k=10)
    _fig_blocks = fig_block_sizes(_df)

    _rep = load_report()
    if _rep:
        items = []
        meta = _rep.get("meta", {})
        quality = _rep.get("quality", {})
        target_sec = _rep.get("target", {})
        schema = _rep.get("schema", {})
        drift = _rep.get("drift", {})

        def add_if(label, value):
            if value is not None:
                items.append(info_row(label, str(value)))

        add_if("Rows", meta.get("rows"))
        add_if("Columns", meta.get("cols"))
        add_if("Target column", meta.get("target_col"))
        add_if("Block dataset", meta.get("is_block_dataset"))
        add_if("Declared drift", meta.get("drift_type"))
        add_if("Position of drift", meta.get("position_of_drift"))
        add_if("Data completeness (%)", quality.get("data_completeness"))
        add_if("Missing values", quality.get("missing_values"))
        add_if("Duplicate rows", quality.get("duplicate_rows"))
        add_if("Classes (target)", target_sec.get("classes_count"))
        add_if("Class balance score", target_sec.get("class_balance_score"))
        add_if("Numeric features", schema.get("features_count"))

        _report_block = html.Div([
            html.Div("Report Summary (report.json)", className="info-label", style={"marginBottom": "8px"}),
            html.Div(items or [html.Div("Report loaded, but no summary keys found.", className="empty-note")], className="dataset-info")
        ])
    else:
        _report_block = html.Div("No report.json found.", className="empty-note")

# -----------------------------
# Layout (English UI)
# -----------------------------
app.index_string = """
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
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #ffffff; margin: 0; padding: 0;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
            .header-gradient {
                background: linear-gradient(135deg, #00d4ff 0%, #51cf66 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                font-weight: 800; font-size: 42px; text-align: center; margin: 8px 0 6px;
            }
            .subtitle { text-align: center; color: #b0b0b0; font-size: 16px; margin-bottom: 24px; font-weight: 500; }
            .panel { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,212,255,0.2); border-radius: 16px; padding: 20px; backdrop-filter: blur(10px); }
            .dataset-info { margin-top: 14px; background: rgba(255,255,255,0.05); border: 1px solid rgba(81,207,102,0.2); border-radius: 12px; padding: 12px; }
            .info-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
            .info-item:last-child { border-bottom: none; }
            .info-label { color: #b0b0b0; font-weight: 600; }
            .info-value { color: #00d4ff; font-weight: 700; }
            .section-title { font-weight: 700; font-size: 28px; text-align: center; margin: 28px 0 12px; background: linear-gradient(135deg, #00d4ff 0%, #51cf66 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
            .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
            .empty-note { color: #b0b0b0; text-align: center; padding: 16px; }
            .footer-note { color: #b0b0b0; font-size: 12px; text-align: center; margin-top: 24px; }
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
"""

app.layout = html.Div([
    html.Div([
        html.Div("CALMOPS Synthetic Viewer", className="header-gradient"),
        html.Div("Read-only dashboard that renders plots from the latest generated CSV and shows the JSON report.", className="subtitle"),

        html.Div(id="dataset-summary", children=_summary, style={"marginTop": "16px"}),

        html.Div("Visual Analysis", className="section-title"),
        html.Div([
            dcc.Graph(figure=_fig_target, style={"height": "380px"}),
        ], className="panel", style={"marginBottom": "16px"}),

        html.Div(className="grid2", children=[
            html.Div([dcc.Graph(figure=_fig_corr, style={"height": "420px"})], className="panel"),
            html.Div([dcc.Graph(figure=_fig_pca,  style={"height": "420px"})], className="panel"),
        ]),

        html.Div("Feature Distributions", className="section-title"),
        html.Div(className="grid3", children=[
            html.Div([dcc.Graph(figure=_fig_h0, style={"height": "360px"})], className="panel"),
            html.Div([dcc.Graph(figure=_fig_h1, style={"height": "360px"})], className="panel"),
            html.Div([dcc.Graph(figure=_fig_h2, style={"height": "360px"})], className="panel"),
        ]),

        html.Div("More Exploratory Plots", className="section-title"),
        html.Div(className="grid2", children=[
            html.Div([dcc.Graph(figure=_fig_violin, style={"height": "420px"})], className="panel"),
            html.Div([dcc.Graph(figure=_fig_scatter, style={"height": "420px"})], className="panel"),
        ]),
        html.Div([dcc.Graph(figure=_fig_topcorr, style={"height": "420px"})], className="panel", style={"marginTop": "16px"}),

        html.Div("Blocks Overview", className="section-title"),
        html.Div([dcc.Graph(figure=_fig_blocks, style={"height": "360px"})], className="panel"),

        html.Div("Random Instances", className="section-title"),
        html.Div(_random_table, className="", style={"marginTop": "8px"}),

        html.Div("Report", className="section-title"),
        html.Div(id="report-view", children=_report_block, className="panel"),

        html.Div("Tip: The viewer reads the latest CSV and 'report.json' from 'salida_tiempo_real/'.", className="footer-note")
    ], className="container")
])

# -----------------------------
# Port helpers
# -----------------------------
def port_in_use(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8061)
    args = parser.parse_args()

    # Simple fallback if port is busy (up to 10 tries)
    chosen_port = args.port
    for _ in range(10):
        if not port_in_use(chosen_port):
            break
        chosen_port += 1

    url = f"http://127.0.0.1:{chosen_port}"
    try:
        webbrowser.open(url)
    except Exception:
        pass

    app.run(debug=False, host="0.0.0.0", port=chosen_port)
