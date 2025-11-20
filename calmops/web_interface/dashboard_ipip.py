# web_interface/dashboard_ipip.py
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import streamlit as st
import joblib

# Import shared utilities
from calmops.utils import get_project_root
from calmops.IPIP.ipip_model import IpipModel
from utils import (
    _load_any_dataset,
    dashboard_data_loader,
    load_log,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')
# =========================
# Sanitizers (from dashboard_block)
# =========================
def _safe_table(df: pd.DataFrame, *, use_container_width: bool = True):
    st.dataframe(df, use_container_width=use_container_width)

def _safe_table_static(df: pd.DataFrame):
    st.table(df)

def _safe_plot(fig: go.Figure, *, use_container_width: bool = True):
    st.plotly_chart(fig, use_container_width=use_container_width)

def _safe_json_display(obj: Any):
    st.json(obj)

# =========================
# Caching & Loading
# =========================
def _mtime(path: str | Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def _read_json_cached(path: str, stamp: float) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _read_text_cached(path: str, stamp: float) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def _load_joblib_cached(path: str, stamp: float):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def _load_any_dataset_cached(path: str, stamp: float) -> pd.DataFrame:
    return _load_any_dataset(path)

# =========================
# Block Helpers
# =========================
def _detect_block_col(pipeline_name: str, df: pd.DataFrame, default: str = "block") -> str | None:
    project_root = get_project_root()
    cfg_path = project_root / "pipelines" / pipeline_name / "config" / "config.json"
    try:
        if cfg_path.exists():
            cfg = _read_json_cached(str(cfg_path), _mtime(str(cfg_path))) or {}
            if cfg.get("block_col") and cfg["block_col"] in df.columns:
                return cfg["block_col"]
    except Exception:
        pass
    if default in df.columns:
        return default
    for c in df.columns:
        if "block" in c.lower() or "chunk" in c.lower():
            return c
    return None

def _sorted_blocks(series: pd.Series):
    vals = series.dropna().unique().tolist()
    try:
        # Try sorting numerically
        return sorted(vals, key=float)
    except (ValueError, TypeError):
        # Fallback to string sorting
        return sorted(vals, key=str)

# =========================
# Dataset Tab
# =========================
def show_dataset_section(df: pd.DataFrame, last_file: str, pipeline_name: str, block_col: Optional[str]):
    st.subheader("Dataset Inspector")

    if df.empty or not last_file:
        st.warning("No processed dataset found yet.")
        return

    st.write(f"*Last processed dataset:* `{last_file}`")

    df_display = df

    if block_col and block_col in df.columns:
        st.markdown(f"Block column detected: **{block_col}**")
        blocks = _sorted_blocks(df[block_col])
        counts = df[block_col].value_counts().reindex(blocks, fill_value=0)

        st.markdown("### Blocks Overview")
        c1, c2 = st.columns([2, 1])
        with c1:
            _safe_table(pd.DataFrame({"block": [str(b) for b in counts.index], "rows": counts.values}))
        with c2:
            fig = px.bar(x=[str(b) for b in counts.index], y=counts.values, labels={"x": "Block", "y": "Rows"}, title="Rows per Block")
            _safe_plot(fig)
        
        selected_block = st.selectbox('Select a block to inspect', options=["(All)"] + blocks, index=0, key="dataset_block_selector")
        if selected_block != "(All)":
            df_display = df[df[block_col].astype(str) == str(selected_block)]
    else:
        st.markdown("No block column detected. Showing global info.")

    st.markdown("### Preview")
    _safe_table(df_display.head(10))

    st.markdown("### Descriptive Statistics")
    _safe_table(df_display.describe())

    st.markdown("### Missingness")
    nan_frac = df_display.isna().mean().sort_values(ascending=False)
    if not nan_frac.empty and nan_frac.max() > 0:
        fig_nan = px.bar(x=nan_frac.index, y=nan_frac.values, labels={"x": "Column", "y": "NaN fraction"}, title="NaN fraction by column")
        _safe_plot(fig_nan)
    else:
        st.info("No missing values detected.")

    st.download_button(
        "Download preview (CSV)",
        data=df_display.to_csv(index=False).encode("utf-8"),
        file_name="preview.csv",
        mime="text/csv",
    )

    st.markdown("### Dataset Info")
    info_df = pd.DataFrame({
        "Column": df_display.columns,
        "Non-Null Count": df_display.notnull().sum().values,
        "Unique Values": df_display.nunique(dropna=True).values,
        "Dtype": df_display.dtypes.astype(str).values,
    })
    _safe_table(info_df)

    # Categorical Variable Analysis
    st.markdown("### Categorical Variable Analysis")
    cat_cols = df_display.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        max_cols_cat = st.slider("Max. categorical columns to display", 1, min(20, len(cat_cols)), min(5, len(cat_cols)), key="cat_slider_dataset")
        show_cols_cat = list(cat_cols[:max_cols_cat])
        
        for col in show_cols_cat:
            st.markdown(f"#### `{col}`")
            
            # Basic stats
            num_unique = df_display[col].nunique()
            mode_val = df_display[col].mode().iloc[0] if not df_display[col].mode().empty else "N/A"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Categories", num_unique)
            with col2:
                st.metric("Mode (Most Frequent)", str(mode_val))

            # Proportions
            proportions = df_display[col].value_counts(normalize=True)
            
            # Display as pie chart if few categories, otherwise as bar chart
            if num_unique <= 10:
                fig = px.pie(proportions, values=proportions.values, names=proportions.index, title=f"Proportions for `{col}`")
                _safe_plot(fig)
            else:
                st.write("**Category Proportions (Top 10)**")
                _safe_table(proportions.head(10))
    else:
        st.info("No categorical columns to analyze.")

    st.markdown("### Processed Files History")
    project_root = get_project_root()
    control_file_path = project_root / "pipelines" / pipeline_name / "control" / "control_file.txt"
    if control_file_path.exists():
        processed_files_data = []
        with open(control_file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    fname, mtime = parts
                    try:
                        processed_files_data.append({"File Name": fname, "Processed At": pd.to_datetime(float(mtime), unit='s')})
                    except ValueError:
                        processed_files_data.append({"File Name": fname, "Processed At": "Invalid Timestamp"})
        if processed_files_data:
            df_processed = pd.DataFrame(processed_files_data).sort_values(by="Processed At", ascending=False)
            _safe_table(df_processed)
        else:
            st.info("No files recorded in control_file.txt yet.")
    else:
        st.info("control_file.txt not found.")

# =========================
# Train/Retrain Tab
# =========================
def show_train_section(pipeline_name: str):
    st.subheader("Train / Retrain Results")
    project_root = get_project_root()
    metrics_dir = project_root / "pipelines" / pipeline_name / "metrics"
    tr_path = metrics_dir / "train_results.json"
    rt_path = metrics_dir / "retrain_results.json"

    use_path = tr_path if tr_path.exists() else (rt_path if rt_path.exists() else None)
    if not use_path:
        st.info("No train/retrain results found in `metrics/`.")
        return

    results = _read_json_cached(str(use_path), _mtime(str(use_path))) or {}
    
    st.markdown("#### General Information")
    st.write(f"**Type:** {results.get('type', 'N/A')}")
    st.write(f"**Timestamp:** {results.get('timestamp', 'N/A')}")
    st.write(f"**File:** `{results.get('file', 'N/A')}`")

    st.markdown("### IPIP Model Structure")
    if results.get("model_type") == "IpipModel":
        p = results.get("p", "N/A")
        b = results.get("b", "N/A")
        num_ensembles = results.get("num_ensembles", "N/A")
        
        cols = st.columns(3)
        cols[0].metric("Target Ensembles (p)", p)
        cols[1].metric("Max Base Models (b)", b)
        cols[2].metric("Final Ensembles", num_ensembles)

        models_per_ensemble = results.get("models_per_ensemble")
        if models_per_ensemble:
            df_struct = pd.DataFrame({
                "Ensemble ID": range(len(models_per_ensemble)),
                "Number of Models": models_per_ensemble
            })
            _safe_table(df_struct)
            fig = px.bar(df_struct, x="Ensemble ID", y="Number of Models", title="Models per Ensemble")
            _safe_plot(fig)

        ensemble_validation_metrics = results.get("ensemble_validation_metrics")
        if ensemble_validation_metrics:
            st.markdown("### Ensemble Internal Validation Performance (Balanced Accuracy)")
            df_val_metrics = pd.DataFrame({
                "Ensemble ID": range(len(ensemble_validation_metrics)),
                "Balanced Accuracy": ensemble_validation_metrics
            })
            _safe_table(df_val_metrics)
            fig_val = px.bar(df_val_metrics, x="Ensemble ID", y="Balanced Accuracy", title="Balanced Accuracy per Ensemble (Internal Validation)")
            _safe_plot(fig_val)

    else:
        st.info("Training results are not from an IPIP model.")

    with st.expander("Raw JSON Output"):
        _safe_json_display(results)

# =========================
# Evaluator Tab
# =========================
def show_evaluator_section(pipeline_name: str):
    st.subheader("Model Evaluation & Approval")
    project_root = get_project_root()
    metrics_dir = project_root / "pipelines" / pipeline_name / "metrics"
    eval_path = metrics_dir / "eval_results.json"

    if not eval_path.exists():
        st.info("No evaluation results found (`eval_results.json`).")
        st.warning("The pipeline may not have completed its first evaluation, or it is not configured to save one.")
        return

    results = _read_json_cached(str(eval_path), _mtime(str(eval_path))) or {}
    
    st.markdown("### Approval Status")
    is_approved = results.get('approved')
    
    if is_approved is not None:
        if is_approved:
            st.success("APPROVED - The model exceeded performance thresholds.")
        else:
            st.error("NOT APPROVED - The model did not meet performance thresholds.")
    
    st.markdown("#### Global Metrics on Evaluation Set")
    metrics = results.get("metrics", {})
    thresholds = results.get("thresholds", {})
    
    metric_data = []
    for m, v in metrics.items():
        # Ignore complex values like 'classification_report' which are dicts
        if not isinstance(v, dict):
            thr = thresholds.get(m, 'N/A')
            metric_data.append({"Metric": m, "Value": v, "Threshold": str(thr)}) # Convert thr to string
    
    if metric_data:
        _safe_table_static(pd.DataFrame(metric_data))
    else:
        st.info("No global metrics found.")

    with st.expander("Raw Evaluation JSON"):
        _safe_json_display(results)

# =========================
# Historical Performance Tab
# =========================
def show_historical_performance_section(pipeline_name: str):
    st.subheader("Historical Performance")
    project_root = get_project_root()
    history_dir = project_root / "pipelines" / pipeline_name / "metrics" / "eval_history"

    if not history_dir.exists():
        st.info("No evaluation history directory found (`metrics/eval_history/`).")
        st.warning("The pipeline may not be configured to store historical evaluation results.")
        return

    history_files = sorted([p.name for p in history_dir.glob("eval_results_*.json")])
    if not history_files:
        st.info("No historical evaluation files found in `metrics/eval_history/`.")
        st.warning("The pipeline has run, but no historical results have been saved yet.")
        return

    all_data = []
    for fname in history_files:
        fpath = history_dir / fname
        data = _read_json_cached(str(fpath), _mtime(fpath))
        if data:
            ts = pd.to_datetime(fname.replace("eval_results_", "").replace(".json", ""), format="%Y%m%d_%H%M%S")
            per_block = data.get("blocks", {}).get("per_block_metrics_full", {})
            if not per_block and "transitions" in data: # IPIP format
                 per_block = {t.get("to"): t.get("metrics", {}) for t in data["transitions"]}

            for block, metrics in per_block.items():
                row = {"timestamp": ts, "block": str(block), **metrics}
                all_data.append(row)

    if not all_data:
        st.warning("Could not load any historical performance data.")
        return

    df_history = pd.DataFrame(all_data).sort_values("timestamp")
    
    metric_cols = [c for c in df_history.columns if c not in ["timestamp", "block"]]
    if not metric_cols:
        st.info("No metrics found in historical data.")
        return
        
    metric_to_plot = st.selectbox("Select Metric", options=metric_cols)
    fig = px.line(df_history, x="timestamp", y=metric_to_plot, color="block", markers=True, title=f"Historical {metric_to_plot} by Block")
    _safe_plot(fig)

# =========================
# IPIP Model Tab
# =========================
def show_ipip_section(pipeline_name: str):
    st.subheader("IPIP Model Inspector")
    project_root = get_project_root()
    model_path = project_root / "pipelines" / pipeline_name / "models" / f"{pipeline_name}.pkl"
    evolution_path = project_root / "pipelines" / pipeline_name / "metrics" / "evolution_results.json"

    if not model_path.exists():
        st.info("No persisted model found.")
        return

    model = _load_joblib_cached(str(model_path), _mtime(model_path))
    
    if hasattr(model, "ensembles_"):
        sizes = [len(Ek) for Ek in getattr(model, "ensembles_", [])]
        cols = st.columns(3)
        cols[0].metric("Ensembles (p)", len(sizes))
        cols[1].metric("Max Models/Ensemble (b)", max(sizes) if sizes else 0)
        cols[2].metric("Total Base Models", sum(sizes))
        
        df_ensembles = pd.DataFrame({"ensemble_id": range(len(sizes)), "num_models": sizes})
        _safe_table(df_ensembles)
        fig = px.bar(df_ensembles, x="ensemble_id", y="num_models", title="Models per Ensemble")
        _safe_plot(fig)
    else:
        st.info("The loaded model does not appear to be an IPIP model (missing `ensembles_` attribute).")

    st.subheader("Model Evolution")
    if evolution_path.exists():
        evolution_info = _read_json_cached(str(evolution_path), _mtime(evolution_path))
        if evolution_info:
            if evolution_info.get("changed"):
                st.warning("Model has changed.")
                for detail in evolution_info.get("details", []):
                    st.write(detail)
            else:
                st.success("Model has not changed.")
        else:
            st.info("No evolution information found.")
    else:
        st.info("No evolution results file found.")

# =========================
# Logs Tab
# =========================
def parse_logs_to_df(log_text: str) -> pd.DataFrame:
    """
    Parse log lines into a structured DataFrame with columns:
    date, pipeline, level, message.
    """
    log_pattern = re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - "
        r"(?P<pipeline>[\w_]+) - (?P<level>\w+) - "
        r"(?P<message>.*)"
    )
    rows = []
    for line in log_text.splitlines():
        match = log_pattern.match(line)
        if match:
            rows.append(match.groupdict())
        else:
            rows.append(
                {"date": None, "pipeline": None, "level": None, "message": line}
            )
    return pd.DataFrame(rows)


def show_logs_section(pipeline_name: str):
    st.subheader("Pipeline Logs")
    project_root = get_project_root()
    logs_path = project_root / "pipelines" / pipeline_name / "logs" / "pipeline.log"
    error_logs_path = (
        project_root / "pipelines" / pipeline_name / "logs" / "pipeline_errors.log"
    )
    warning_logs_path = (
        project_root / "pipelines" / pipeline_name / "logs" / "pipeline_warnings.log"
    )

    # Function to read logs and delete DeltaGenerator messages
    def clean_log_content(log_text):
        # Ignore lines that start with DeltaGenerator
        lines = [
            line
            for line in log_text.splitlines()
            if not line.strip().startswith("DeltaGenerator")
        ]
        return "\n".join(lines).strip()

    # General Log
    st.markdown("### General Log")
    general_log_text = clean_log_content(_read_text_cached(str(logs_path), _mtime(logs_path)))
    df_general = parse_logs_to_df(general_log_text)
    if not df_general.empty:
        st.dataframe(
            df_general.style.map(
                lambda v: "color: red;"
                if v == "ERROR"
                else "color: orange;"
                if v == "WARNING"
                else "color: green;",
                subset=["level"],
            ),
            use_container_width=True,
        )
    else:
        st.info("No general logs found.")

    # Error Log
    st.markdown("### Error Log")
    error_log_text = clean_log_content(_read_text_cached(str(error_logs_path), _mtime(error_logs_path)))
    if error_log_text and "No logs found" not in error_log_text:
        st.code(error_log_text, language="bash")
    else:
        st.success("No errors found.")

    # Warning Log
    st.markdown("### Warning Log")
    warning_log_text = clean_log_content(_read_text_cached(str(warning_logs_path), _mtime(warning_logs_path)))
    if warning_log_text and "No logs found" not in warning_log_text:
        st.code(warning_log_text, language="bash")
    else:
        st.success("No warnings found.")

# =========================
# Main App
# =========================
st.set_page_config(page_title="CalmOps IPIP Dashboard", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_name", type=str, required=True)
try:
    args, _ = parser.parse_known_args()
    pipeline_name = args.pipeline_name
except SystemExit: # Handles Streamlit's internal argument passing
    st.error("Please provide a pipeline name, e.g., `streamlit run ... -- --pipeline_name my_pipeline`")
    st.stop()

st.title(f"Dashboard — IPIP Pipeline: `{pipeline_name}`")

project_root = get_project_root()
# Auto-refresh logic
control_file = project_root / "pipelines" / pipeline_name / "control" / "control_file.txt"
if 'control_mtime' not in st.session_state:
    st.session_state.control_mtime = _mtime(control_file)

current_mtime = _mtime(control_file)
if current_mtime > st.session_state.control_mtime:
    st.session_state.control_mtime = current_mtime
    st.rerun()

# Data Loading
data_dir = project_root / "pipelines" / pipeline_name / "config" / "runner_config.json"
df, last_file = pd.DataFrame(), None
runner_cfg = {}
if data_dir.exists():
    runner_cfg = _read_json_cached(str(data_dir), _mtime(data_dir))
    if runner_cfg and runner_cfg.get("data_dir"):
        df, last_file = dashboard_data_loader(runner_cfg["data_dir"], project_root / "pipelines" / pipeline_name / "control")

block_col = _detect_block_col(pipeline_name, df)

# Tabs
tab_names = ["Dataset", "Train/Retrain", "Evaluator", "Historical Performance", "IPIP Model", "Logs"]
tabs = st.tabs(tab_names)

with tabs[0]:
    show_dataset_section(df, last_file, pipeline_name, block_col)
with tabs[1]:
    show_train_section(pipeline_name)
with tabs[2]:
    show_evaluator_section(pipeline_name)
with tabs[3]:
    show_historical_performance_section(pipeline_name)
with tabs[4]:
    show_ipip_section(pipeline_name)
with tabs[5]:
    show_logs_section(pipeline_name)