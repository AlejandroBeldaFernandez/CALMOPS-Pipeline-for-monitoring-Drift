#!/usr/bin/env python3
"""
CALMOPS - Informational Dashboard (Viewer)
=========================================

This Dash app is a **read-only** viewer:
- It does NOT generate data.
- It scans a base folder for previously generated assets (CSV, plots, reports).
- It displays plot galleries and key report information.

Conventions it understands (best-effort):
- Base folder: env var `CALMOPS_OUTPUT_DIR` or `./plots` by default.
- Inside the base folder, each dataset can either be:
  - A subfolder with plots (png/jpg/svg/html), CSV(s), and optional report files
    like `report.json`/`synthetic_report.json` or `report.html`.
  - Or the base folder itself may just contain artifacts; then it's shown as
    a single dataset named "default".

You can adapt the glob patterns or file names below to match your reporter.

Author: CalmOps Team
Version: 1.0 (Viewer-only)
"""

import os
import io
import json
import base64
import glob
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, no_update
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.environ.get("CALMOPS_OUTPUT_DIR", "./plots")
VALID_IMG_EXT = (".png", ".jpg", ".jpeg", ".svg")
VALID_HTML_EXT = (".html", ".htm")
VALID_CSV_EXT = (".csv",)

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "CALMOPS Viewer"

# -----------------------------
# Utilities
# -----------------------------

def discover_datasets(base_dir: str) -> list:
    """Return a list of dataset entries as (label, path)."""
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        return []

    # Look for subdirectories with content
    subdirs = [d for d in sorted(os.listdir(base_dir))
               if os.path.isdir(os.path.join(base_dir, d))]

    datasets = []
    for d in subdirs:
        full = os.path.join(base_dir, d)
        # Check if contains any artifact files
        if glob.glob(os.path.join(full, "**", "*"), recursive=True):
            datasets.append((d, full))

    # If base_dir itself has artifacts, add a default entry
    if glob.glob(os.path.join(base_dir, "*")) and (not datasets):
        datasets.append(("default", base_dir))

    return datasets


def _file_newest_mtime(path_pattern: str) -> float:
    files = glob.glob(path_pattern)
    return max((os.path.getmtime(f) for f in files), default=0)


def dataset_last_updated(dataset_path: str) -> str:
    """Return a human-readable 'last updated' time for a dataset path."""
    newest = 0
    for pattern in ("*.csv", "*.png", "*.jpg", "*.jpeg", "*.svg", "*.html", "*.htm", "*.json"):
        newest = max(newest, _file_newest_mtime(os.path.join(dataset_path, "**", pattern)))
    if newest == 0:
        return "N/A"
    return datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M:%S")


def load_image_cards(dataset_path: str) -> list:
    """Return a list of image/plot cards (Dash html) encoded as base64 or iframe for HTML plots."""
    cards = []

    # Images
    for img_path in sorted(glob.glob(os.path.join(dataset_path, "**", "*"), recursive=True)):
        ext = os.path.splitext(img_path)[1].lower()
        if ext in VALID_IMG_EXT:
            try:
                b64 = base64.b64encode(open(img_path, "rb").read()).decode("ascii")
                src = f"data:image/{ext[1:]};base64,{b64}"
                cards.append(
                    html.Div([
                        html.Img(src=src, style={"width": "100%", "borderRadius": "12px"}),
                        html.Div(os.path.relpath(img_path, dataset_path), className="card-caption")
                    ], className="plot-card")
                )
            except Exception:
                continue

    # Standalone HTML plots (e.g., Plotly saved as HTML)
    for html_path in sorted(glob.glob(os.path.join(dataset_path, "**", "*"), recursive=True)):
        ext = os.path.splitext(html_path)[1].lower()
        if ext in VALID_HTML_EXT:
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_src = f.read()
                cards.append(
                    html.Div([
                        html.Iframe(srcDoc=html_src, style={"width": "100%", "height": "360px", "border": "0", "borderRadius": "12px"}),
                        html.Div(os.path.relpath(html_path, dataset_path), className="card-caption")
                    ], className="plot-card")
                )
            except Exception:
                continue

    if not cards:
        cards = [html.Div("No plots found in this dataset.", className="empty-note")]

    return cards


def find_report(dataset_path: str):
    """Try to load a report JSON or HTML; return (report_dict, html_str)."""
    # JSON candidates
    for name in ("report.json", "synthetic_report.json", "report_summary.json"):
        cand = os.path.join(dataset_path, name)
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f), None
            except Exception:
                pass

    # HTML candidates
    for name in ("report.html", "synthetic_report.html"):
        cand = os.path.join(dataset_path, name)
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8", errors="ignore") as f:
                    return None, f.read()
            except Exception:
                pass

    return None, None


def load_any_csv(dataset_path: str) -> pd.DataFrame | None:
    """Load the first CSV found, best-effort (for preview/summary)."""
    for csv_path in sorted(glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            continue
    return None


def build_report_summary(report_dict: dict) -> list:
    """Render a compact summary from a report dictionary."""
    if not report_dict:
        return [html.Div("No JSON report found.", className="empty-note")]

    items = []
    # Heuristics for common keys
    key_map = [
        ("drift_type", "Drift Type"),
        ("position_of_drift", "Position of Drift"),
        ("is_block_dataset", "Block Dataset"),
        ("quality_score", "Quality Score"),
        ("overall_score", "Overall Score"),
        ("class_balance", "Class Balance"),
        ("data_completeness", "Data Completeness"),
        ("feature_diversity", "Feature Diversity"),
        ("statistical_validity", "Statistical Validity"),
    ]

    for k, label in key_map:
        if k in report_dict:
            items.append(info_row(label, str(report_dict[k])))

    # Fallback: show top-level keys briefly
    if not items:
        for k, v in list(report_dict.items())[:10]:
            items.append(info_row(k, str(v)))

    return [html.Div(items, className="dataset-info")]


def info_row(label: str, value: str):
    return html.Div([
        html.Span(label, className="info-label"),
        html.Span(value, className="info-value"),
    ], className="info-item")


# -----------------------------
# Layout
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
                color: #ffffff;
                margin: 0; padding: 0;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
            .header-gradient {
                background: linear-gradient(135deg, #00d4ff 0%, #51cf66 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                font-weight: 800; font-size: 42px; text-align: center; margin: 8px 0 6px;
            }
            .subtitle { text-align: center; color: #b0b0b0; font-size: 16px; margin-bottom: 24px; font-weight: 500; }
            .panel { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,212,255,0.2); border-radius: 16px; padding: 20px; backdrop-filter: blur(10px); }
            .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: center; }
            .label { color: #b0b0b0; font-weight: 600; }
            .dataset-info { margin-top: 14px; background: rgba(255,255,255,0.05); border: 1px solid rgba(81,207,102,0.2); border-radius: 12px; padding: 12px; }
            .info-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
            .info-item:last-child { border-bottom: none; }
            .info-label { color: #b0b0b0; font-weight: 600; }
            .info-value { color: #00d4ff; font-weight: 700; }
            .section-title { font-weight: 700; font-size: 28px; text-align: center; margin: 28px 0 12px; background: linear-gradient(135deg, #00d4ff 0%, #51cf66 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
            .plot-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(0,212,255,0.2); border-radius: 14px; padding: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.25); }
            .card-caption { margin-top: 8px; font-size: 12px; color: #b0b0b0; word-break: break-all; }
            .empty-note { color: #b0b0b0; text-align: center; padding: 16px; }
            .footer-note { color: #b0b0b0; font-size: 12px; text-align: center; margin-top: 24px; }
            .inline { display: inline-block; }
            .dropdown { background: #1e1e1e; color: #fff; border: 1px solid #00d4ff; border-radius: 8px; padding: 6px 10px; }
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


def make_dataset_options():
    opts = []
    for label, path in discover_datasets(BASE_DIR):
        last_upd = dataset_last_updated(path)
        opts.append({"label": f"{label}  ·  {last_upd}", "value": path})
    return opts or [{"label": "No datasets found", "value": ""}]


app.layout = html.Div([
    html.Div([
        html.Div("CALMOPS Viewer", className="header-gradient"),
        html.Div("Read-only dashboard to browse plots and reports generated by the Synthetic pipeline.", className="subtitle"),

        html.Div([
            html.Div([
                html.Span("Base folder:", className="label inline"),
                html.Span(os.path.abspath(BASE_DIR), style={"marginLeft": "8px", "color": "#74c0fc"})
            ]),
            html.Div(className="controls", children=[
                html.Div([
                    html.Div("Select dataset", className="label"),
                    dcc.Dropdown(id="dataset-dropdown", options=make_dataset_options(), value=(make_dataset_options()[0]["value"] if make_dataset_options() else ""), clearable=False)
                ]),
                html.Div([
                    html.Div("Auto-refresh (seconds)", className="label"),
                    dcc.Slider(id="refresh-slider", min=0, max=60, step=5, value=0, tooltip={"always_visible": False})
                ])
            ])
        ], className="panel"),

        html.Div(id="dataset-summary"),

        html.Div("Plots", className="section-title"),
        html.Div(id="plots-gallery", className="gallery"),

        html.Div("Report", className="section-title"),
        html.Div(id="report-view"),

        dcc.Interval(id="refresh-timer", interval=0, disabled=True, n_intervals=0),
        html.Div("Tip: Place your generated assets under the base folder. The viewer will show images (png/jpg/svg), standalone HTML plots, and basic report JSON/HTML if present.", className="footer-note")
    ], className="container")
])


# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    Output("refresh-timer", "interval"),
    Output("refresh-timer", "disabled"),
    Input("refresh-slider", "value"),
)
def configure_refresh(seconds):
    if not seconds:
        return 0, True
    return int(seconds * 1000), False


@app.callback(
    Output("dataset-summary", "children"),
    Output("plots-gallery", "children"),
    Output("report-view", "children"),
    Input("dataset-dropdown", "value"),
    Input("refresh-timer", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_view(selected_path, _tick):
    if not selected_path or not os.path.isdir(selected_path):
        return (
            html.Div("No dataset selected or folder missing.", className="empty-note"),
            [html.Div("—", className="empty-note")],
            [html.Div("—", className="empty-note")],
        )

    # Summary section (DataFrame preview + basic facts)
    df = load_any_csv(selected_path)
    facts = []
    if df is not None and not df.empty:
        n_rows, n_cols = df.shape
        facts.append(info_row("Rows", f"{n_rows:,}"))
        facts.append(info_row("Columns", str(n_cols)))
        if "target" in df.columns:
            cls = df["target"].nunique(dropna=True)
            facts.append(info_row("Classes (target)", str(cls)))
        facts.append(info_row("Last Updated", dataset_last_updated(selected_path)))
        # Sample preview (first 5 rows), as a small HTML table
        preview = df.head(5).to_dict(orient="records")
        table_header = [html.Th(c) for c in df.columns]
        table_rows = [html.Tr([html.Td(row.get(c, "")) for c in df.columns]) for row in preview]
        preview_table = html.Table([
            html.Thead(html.Tr(table_header)),
            html.Tbody(table_rows)
        ], style={"width": "100%", "borderCollapse": "collapse"})
        summary_block = html.Div([
            html.Div(facts, className="dataset-info"),
            html.Div([html.Div("Preview (top 5 rows)", className="label", style={"margin": "8px 0"}), preview_table], className="panel",)
        ])
    else:
        summary_block = html.Div([
            html.Div([info_row("Last Updated", dataset_last_updated(selected_path))], className="dataset-info"),
            html.Div("No CSV found for preview.", className="empty-note")
        ])

    # Plots gallery
    plots = load_image_cards(selected_path)

    # Report view: prefer JSON summary; fallback to HTML iframe
    rep_json, rep_html = find_report(selected_path)
    if rep_json:
        report_children = build_report_summary(rep_json)
    elif rep_html:
        report_children = [html.Iframe(srcDoc=rep_html, style={"width": "100%", "height": "720px", "border": "0", "borderRadius": "12px"})]
    else:
        report_children = [html.Div("No report found in this dataset.", className="empty-note")]

    return summary_block, plots, report_children


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    app.run_server(debug=False, host="0.0.0.0", port=8061)
