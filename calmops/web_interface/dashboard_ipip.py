import os
import json
import re
import argparse
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Import shared utilities
from calmops.utils import get_pipelines_root
from utils import (
    _load_any_dataset,
    dashboard_data_loader,
    show_evolution_section,  # We reuse the generic one but wrap it
)
from dashboard_common import (
    _mtime,
    _read_json_cached,
    _read_text_cached,
    _read_csv_cached,
    _load_joblib_cached,
    _load_any_dataset_cached,
    _sanitize_text,
    _safe_table,
    _safe_table_static,
    _safe_plot,
    _safe_json_display,
    _sorted_blocks,
    _get_pipeline_base_dir,
    _detect_block_col,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf


# tf.compat.v1.logging.set_verbosity(# tf.compat.v1.logging.ERROR)

# tf.get_logger().setLevel("ERROR")


# Sanitizers, caching helpers, and block sorting logic have been moved to dashboard_common.py


# =========================
# Dataset Tab
# =========================
def show_dataset_section(
    df: pd.DataFrame, last_file: str, pipeline_name: str, block_col: Optional[str]
):
    """
    Displays the Dataset tab, including preview, statistics, missingness, and block distribution.

    Args:
        df: The loaded dataframe of the current/latest data.
        last_file: Filename of the last processed file.
        pipeline_name: Name of the current pipeline (for finding control files).
        block_col: Name of the block column if detected.
    """
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
            _safe_table(
                pd.DataFrame(
                    {"block": [str(b) for b in counts.index], "rows": counts.values}
                )
            )
        with c2:
            fig = px.bar(
                x=[str(b) for b in counts.index],
                y=counts.values,
                labels={"x": "Block", "y": "Rows"},
                title="Rows per Block",
            )
            _safe_plot(fig)

        selected_block = st.selectbox(
            "Select a block to inspect",
            options=["(All)"] + blocks,
            index=0,
            key="dataset_block_selector",
        )
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
        fig_nan = px.bar(
            x=nan_frac.index,
            y=nan_frac.values,
            labels={"x": "Column", "y": "NaN fraction"},
            title="NaN fraction by column",
        )
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
    info_df = pd.DataFrame(
        {
            "Column": df_display.columns,
            "Non-Null Count": df_display.notnull().sum().values,
            "Unique Values": df_display.nunique(dropna=True).values,
            "Dtype": df_display.dtypes.astype(str).values,
        }
    )
    _safe_table(info_df)

    # Categorical Variable Analysis
    st.markdown("### Categorical Variable Analysis")
    cat_cols = df_display.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        if len(cat_cols) > 1:
            max_cols_cat = st.slider(
                "Max. categorical columns to display",
                1,
                min(20, len(cat_cols)),
                min(5, len(cat_cols)),
                key="cat_slider_dataset",
            )
        else:
            max_cols_cat = 1
        show_cols_cat = list(cat_cols[:max_cols_cat])

        for col in show_cols_cat:
            st.markdown(f"#### `{col}`")

            # Basic stats
            num_unique = df_display[col].nunique()
            mode_val = (
                df_display[col].mode().iloc[0]
                if not df_display[col].mode().empty
                else "N/A"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Categories", num_unique)
            with col2:
                st.metric("Mode (Most Frequent)", str(mode_val))

            # Proportions
            proportions = df_display[col].value_counts(normalize=True)

            # Display as pie chart if few categories, otherwise as bar chart
            if num_unique <= 10:
                fig = px.pie(
                    proportions,
                    values=proportions.values,
                    names=proportions.index,
                    title=f"Proportions for `{col}`",
                )
                _safe_plot(fig)
            else:
                st.write("**Category Proportions (Top 10)**")
                _safe_table(proportions.head(10))
    else:
        st.info("No categorical columns to analyze.")

    st.markdown("### Processed Files History")
    project_root = get_pipelines_root()
    control_file_path = (
        project_root / "pipelines" / pipeline_name / "control" / "control_file.txt"
    )
    if control_file_path.exists():
        processed_files_data = []
        with open(control_file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    fname, mtime = parts
                    try:
                        processed_files_data.append(
                            {
                                "File Name": fname,
                                "Processed At": pd.to_datetime(float(mtime), unit="s"),
                            }
                        )
                    except ValueError:
                        processed_files_data.append(
                            {"File Name": fname, "Processed At": "Invalid Timestamp"}
                        )
        if processed_files_data:
            df_processed = pd.DataFrame(processed_files_data).sort_values(
                by="Processed At", ascending=False
            )
            _safe_table(df_processed)
        else:
            st.info("No files recorded in control_file.txt yet.")
    else:
        st.info("control_file.txt not found.")


# =========================
# Train/Retrain Tab
# =========================
def show_train_section(pipeline_name: str):
    """
    Displays training and retraining results, including IPIP model structure and ensemble composition.
    Reads from `train_results.json` or `retrain_results.json`.
    """
    st.subheader("Train / Retrain Results")
    project_root = get_pipelines_root()
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
            df_struct = pd.DataFrame(
                {
                    "Ensemble ID": range(len(models_per_ensemble)),
                    "Number of Models": models_per_ensemble,
                }
            )
            _safe_table(df_struct)
            fig = px.bar(
                df_struct,
                x="Ensemble ID",
                y="Number of Models",
                title="Models per Ensemble",
            )
            _safe_plot(fig)

        ensemble_validation_metrics = results.get("ensemble_validation_metrics")
        if ensemble_validation_metrics:
            st.markdown(
                "### Ensemble Internal Validation Performance (Balanced Accuracy)"
            )
            df_val_metrics = pd.DataFrame(
                {
                    "Ensemble ID": range(len(ensemble_validation_metrics)),
                    "Balanced Accuracy": ensemble_validation_metrics,
                }
            )
            _safe_table(df_val_metrics)
            fig_val = px.line(
                df_val_metrics,
                x="Ensemble ID",
                y="Balanced Accuracy",
                title="Balanced Accuracy per Ensemble (Internal Validation)",
                markers=True,
            )
            _safe_plot(fig_val)

    else:
        st.info("Training results are not from an IPIP model.")

    with st.expander("Raw JSON Output"):
        _safe_json_display(results)

    # Note: Training history plot moved to 'Evolution' tab for better organization


# =========================
# Evaluator Tab
# =========================
def show_evaluator_section(pipeline_name: str):
    """
    Displays evaluation metrics for the current model.
    Reads from `eval_results.json` and shows metrics, thresholds, and predictions.
    """
    st.subheader("Evaluation Results")
    project_root = get_pipelines_root()
    metrics_dir = project_root / "pipelines" / pipeline_name / "metrics"
    eval_path = metrics_dir / "eval_results.json"

    if not eval_path.exists():
        st.info("No evaluation results found (`eval_results.json`).")
        st.warning(
            "The pipeline may not have completed its first evaluation, or it is not configured to save one."
        )
        return

    results = _read_json_cached(str(eval_path), _mtime(str(eval_path))) or {}

    st.markdown("#### Global Metrics on Evaluation Set")
    metrics = results.get("metrics", {})
    thresholds = results.get("thresholds", {})

    metric_data = []
    for m, v in metrics.items():
        # Ignore complex values like 'classification_report' which are dicts
        if not isinstance(v, dict):
            thr = thresholds.get(m, "N/A")
            metric_data.append(
                {"Metric": m, "Value": v, "Threshold": str(thr)}
            )  # Convert thr to string

    if metric_data:
        _safe_table_static(pd.DataFrame(metric_data))
    else:
        st.info("No global metrics found.")

    with st.expander("Raw Evaluation JSON"):
        _safe_json_display(results)

    # NEW: Download Predictions
    preds_path = metrics_dir / "predictions.csv"
    if preds_path.exists():
        st.markdown("### Predictions")
        with open(preds_path, "r") as f:
            st.download_button(
                label="Download Full Predictions (CSV)",
                data=f,
                file_name="predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("No `predictions.csv` found.")

    # NEW: Probability Distribution Plot
    if preds_path.exists():
        try:
            df_preds = pd.read_csv(preds_path)
            if "y_pred_proba" in df_preds.columns and "y_true" in df_preds.columns:
                st.markdown("### Probability Distribution")

                # Create histogram
                fig_hist = px.histogram(
                    df_preds,
                    x="y_pred_proba",
                    color="y_true",
                    nbins=50,
                    opacity=0.7,
                    labels={
                        "y_pred_proba": "Predicted Probability (Positive Class)",
                        "y_true": "True Class",
                    },
                    title="Distribution of Predicted Probabilities by Class",
                    marginal="box",  # Adds a box plot on top
                )
                fig_hist.update_layout(barmode="overlay")
                _safe_plot(fig_hist)

                # Optional: Calibration curve (Reliability diagram) could be added here too
            else:
                st.info(
                    "Predictions file does not contain probability data (`y_pred_proba`)."
                )
        except Exception as e:
            st.warning(f"Could not load predictions for visualization: {e}")


# =========================
# Historical Performance Tab
# =========================
def show_historical_performance_section(pipeline_name: str):
    """
    Visualizes historical performance metrics over time/blocks.
    Reads from `metrics/eval_history/` and plots selected metrics with optional CI bands.
    """
    st.subheader("Historical Performance")
    project_root = get_pipelines_root()
    history_dir = (
        project_root / "pipelines" / pipeline_name / "metrics" / "eval_history"
    )

    if not history_dir.exists():
        st.info("No evaluation history directory found (`metrics/eval_history/`).")
        st.warning(
            "The pipeline may not be configured to store historical evaluation results."
        )
        return

    history_files = sorted([p.name for p in history_dir.glob("eval_results_*.json")])
    if not history_files:
        st.info("No historical evaluation files found in `metrics/eval_history/`.")
        st.warning(
            "The pipeline has run, but no historical results have been saved yet."
        )
        return

    all_data = []
    for fname in history_files:
        fpath = history_dir / fname
        data = _read_json_cached(str(fpath), _mtime(fpath))
        if data:
            ts = pd.to_datetime(
                fname.replace("eval_results_", "").replace(".json", ""),
                format="%Y%m%d_%H%M%S",
            )
            per_block = data.get("blocks", {}).get("per_block_metrics_full", {})
            if not per_block and "transitions" in data:  # IPIP format
                per_block = {
                    t.get("to"): t.get("metrics", {}) for t in data["transitions"]
                }

            for block, metrics in per_block.items():
                row = {"timestamp": ts, "block": str(block), **metrics}
                all_data.append(row)

    if not all_data:
        st.warning("Could not load any historical performance data.")
        return

    # Merge with Training History (for IPIP parameters)
    metrics_dir = project_root / "pipelines" / pipeline_name / "metrics"
    train_hist_path = metrics_dir / "training_history.json"

    if train_hist_path.exists():
        try:
            train_hist_data = _read_json_cached(
                str(train_hist_path), _mtime(str(train_hist_path))
            )
            if train_hist_data:
                df_train_hist = pd.DataFrame(train_hist_data)
                # Ensure we have a join key, usually 'block'
                if "block" in df_train_hist.columns and not df_history.empty:
                    # Filter useful cols
                    train_cols = [
                        c
                        for c in df_train_hist.columns
                        if c
                        in [
                            "block",
                            "p",
                            "b",
                            "num_ensembles",
                            "replacement_percentage",
                        ]
                    ]
                    df_train_subset = df_train_hist[train_cols].copy()
                    df_train_subset["block"] = df_train_subset["block"].astype(str)

                    df_history = pd.merge(
                        df_history, df_train_subset, on="block", how="left"
                    )
        except Exception as e:
            st.warning(f"Could not load training parameters from history: {e}")

    # Plot metrics
    # Exclude non-numeric or non-relevant columns
    exclude_cols = ["timestamp", "block", "error_y", "datetime"]
    metric_cols = [c for c in df_history.columns if c not in exclude_cols]

    if not metric_cols:
        st.info("No metrics found in historical data.")
        return

    # Default to balanced_accuracy if available, else first metric
    default_ix = (
        metric_cols.index("balanced_accuracy")
        if "balanced_accuracy" in metric_cols
        else 0
    )

    metric_to_plot = st.selectbox(
        "Select Metric or Parameter to Plot", options=metric_cols, index=default_ix
    )

    # Calculate 95% CI if metric is accuracy-like and support is available
    # ONLY for actual performance metrics, not parameters like 'p' or 'b'
    is_performance_metric = metric_to_plot in [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
    ]

    if is_performance_metric and "support" in df_history.columns:
        # Standard Error = sqrt(p * (1-p) / n)
        # 95% CI = 1.96 * SE
        df_history["error_y"] = 1.96 * np.sqrt(
            df_history[metric_to_plot]
            * (1 - df_history[metric_to_plot])
            / df_history["support"]
        )
        # Handle cases where support is 0 or NaN
        df_history["error_y"] = df_history["error_y"].fillna(0)
    else:
        df_history["error_y"] = None

    # Create figure with shaded CI
    fig = go.Figure()

    # Get unique blocks to plot
    unique_blocks = df_history["block"].unique()

    # Define a color palette
    colors = px.colors.qualitative.Plotly

    for i, block in enumerate(unique_blocks):
        block_data = df_history[df_history["block"] == block].sort_values("timestamp")
        color = colors[i % len(colors)]

        # Main line
        fig.add_trace(
            go.Scatter(
                x=block_data["timestamp"],
                y=block_data[metric_to_plot],
                mode="lines+markers",
                name=f"Block {block}",
                line=dict(color=color),
                legendgroup=f"group_{block}",
            )
        )

        # Add CI band if error_y is present
        if "error_y" in block_data.columns and block_data["error_y"].notna().any():
            upper_bound = block_data[metric_to_plot] + block_data["error_y"]
            lower_bound = block_data[metric_to_plot] - block_data["error_y"]

            # Upper bound (transparent)
            fig.add_trace(
                go.Scatter(
                    x=block_data["timestamp"],
                    y=upper_bound,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=f"group_{block}",
                )
            )

            # Lower bound (fill to upper)
            fig.add_trace(
                go.Scatter(
                    x=block_data["timestamp"],
                    y=lower_bound,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i : i + 2], 16) for i in (0, 2, 4)) + [0.2])}",  # Hex to RGBA with opacity
                    name=f"95% CI (Block {block})",
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=f"group_{block}",
                )
            )

    fig.update_layout(
        title=f"Historical {metric_to_plot} by Block (with 95% CI)"
        if (is_performance_metric and "error_y" in df_history.columns)
        else f"Historical {metric_to_plot} by Block",
        xaxis_title="Timestamp",
        yaxis_title=metric_to_plot,
        hovermode="x unified",
    )
    _safe_plot(fig)


# =========================
# IPIP Model Tab
# =========================
def show_ipip_section(pipeline_name: str):
    """
    Inspects the persisted IPIP model structure (ensembles, base models).
    Loads from `model_structure.json` or fallback to loading the pickle file directly.
    """
    st.subheader("IPIP Model Inspector")
    project_root = get_pipelines_root()
    model_path = (
        project_root / "pipelines" / pipeline_name / "models" / f"{pipeline_name}.pkl"
    )
    model_structure_path = (
        project_root / "pipelines" / pipeline_name / "metrics" / "model_structure.json"
    )

    if not model_path.exists():
        st.info("No persisted model found.")
        return

    # Try to load model structure from JSON first
    model_struct_info = None
    if model_structure_path.exists():
        model_struct_info = _read_json_cached(
            str(model_structure_path), _mtime(model_structure_path)
        )
        if model_struct_info:
            st.info("Displaying model structure from `model_structure.json`")

    sizes = []
    if model_struct_info:
        sizes = model_struct_info.get("models_per_ensemble", [])
        num_ensembles = model_struct_info.get("num_ensembles", 0)
        timestamp = model_struct_info.get("timestamp", "N/A")
        st.write(f"**Last updated:** {timestamp}")
    else:
        # Fallback to loading the model directly if JSON not found
        st.info(
            "`model_structure.json` not found. Loading model directly for structure."
        )
        try:
            model = _load_joblib_cached(str(model_path), _mtime(model_path))
            if hasattr(model, "ensembles_"):
                sizes = [len(Ek) for Ek in getattr(model, "ensembles_", [])]
                num_ensembles = len(sizes)
            else:
                st.warning(
                    "The loaded model does not appear to be an IPIP model (missing `ensembles_` attribute)."
                )
                return
        except Exception as e:
            st.error(f"Could not load model to inspect its structure: {e}")
            return

    if sizes:
        cols = st.columns(3)
        cols[0].metric("Ensembles (p)", num_ensembles)
        cols[1].metric("Max Models/Ensemble (b)", max(sizes) if sizes else 0)
        cols[2].metric("Total Base Models", sum(sizes))

        df_ensembles = pd.DataFrame(
            {"ensemble_id": range(len(sizes)), "num_models": sizes}
        )
        _safe_table(df_ensembles)
        fig = px.bar(
            df_ensembles, x="ensemble_id", y="num_models", title="Models per Ensemble"
        )
        _safe_plot(fig)
    else:
        st.info("No ensemble information available.")


# =========================
# Custom Evolution Section
# =========================
def show_ipip_evolution_section(pipeline_name: str):
    """
    Displays the enhanced 'Evolution' tab for IPIP results.
    1. Shows general lifecycle decisions and approval history (generic).
    2. Shows IPIP internal parameter evolution (p, b, ensembles, replacement %).
    """
    # 1. Generic History (Approval, Decision Timeline, Eval Metrics)
    show_evolution_section(pipeline_name)

    st.markdown("---")
    st.subheader("⚙️ IPIP Parameter Evolution")

    project_root = get_pipelines_root()
    metrics_dir = project_root / "pipelines" / pipeline_name / "metrics"
    history_path = metrics_dir / "training_history.json"

    if history_path.exists():
        history_data = _read_json_cached(str(history_path), _mtime(str(history_path)))
        if history_data:
            df_hist = pd.DataFrame(history_data)

            # Ensure we have block information or index to plot against
            # Usually we use 'block' column if it exists, or just index
            x_axis = "block" if "block" in df_hist.columns else None

            # --- Plot 1: p and b ---
            if "p" in df_hist.columns and "b" in df_hist.columns:
                fig_params = px.line(
                    df_hist,
                    x=x_axis,
                    y=["p", "b"],
                    title="Evolution of Adaptation Parameters (p & b)",
                    markers=True,
                    labels={"value": "Parameter Value", "variable": "Parameter"},
                )
                _safe_plot(fig_params)

            # --- Plot 2: Ensembles ---
            if "num_ensembles" in df_hist.columns:
                fig_ensembles = px.line(
                    df_hist,
                    x=x_axis,
                    y="num_ensembles",
                    title="Evolution of Model Complexity (Number of Ensembles)",
                    markers=True,
                    labels={"num_ensembles": "Ensembles count"},
                )
                _safe_plot(fig_ensembles)

            # --- Plot 3: Replacement Percentage ---
            if "replacement_percentage" in df_hist.columns:
                # Fill N/A with 0 for initial train
                df_hist["replacement_percentage"] = df_hist[
                    "replacement_percentage"
                ].fillna(0)

                fig_repl = px.bar(
                    df_hist,
                    x=x_axis,
                    y="replacement_percentage",
                    title="Replacement Percentage per Block (Model Renewal Rate)",
                    labels={"replacement_percentage": "Replacement %"},
                    color="replacement_percentage",
                    color_continuous_scale="Viridis",
                )
                _safe_plot(fig_repl)

            with st.expander("Raw Parameter Data"):
                _safe_table(df_hist)
        else:
            st.info("Training history file is empty.")
    else:
        st.info(
            "No `training_history.json` found. Run the pipeline to generate history."
        )


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
    project_root = get_pipelines_root()
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
    general_log_text = clean_log_content(
        _read_text_cached(str(logs_path), _mtime(logs_path))
    )
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
    error_log_text = clean_log_content(
        _read_text_cached(str(error_logs_path), _mtime(error_logs_path))
    )
    if error_log_text and "No logs found" not in error_log_text:
        st.code(error_log_text, language="bash")
    else:
        st.success("No errors found.")

    # Warning Log
    st.markdown("### Warning Log")
    warning_log_text = clean_log_content(
        _read_text_cached(str(warning_logs_path), _mtime(warning_logs_path))
    )
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
except SystemExit:  # Handles Streamlit's internal argument passing
    st.error(
        "Please provide a pipeline name, e.g., `streamlit run ... -- --pipeline_name my_pipeline`"
    )
    st.stop()

st.title(f"Dashboard — IPIP Pipeline: `{pipeline_name}`")

project_root = get_pipelines_root()
# Auto-refresh logic
control_file = (
    project_root / "pipelines" / pipeline_name / "control" / "control_file.txt"
)
if "control_mtime" not in st.session_state:
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
        df, last_file = dashboard_data_loader(
            runner_cfg["data_dir"],
            project_root / "pipelines" / pipeline_name / "control",
        )

block_col = _detect_block_col(pipeline_name, df)

# Tabs
tab_names = [
    "Dataset",
    "Train/Retrain",
    "Evaluator",
    "Historical Performance",  # Explicitly separate from Evolution
    "Evolution",
    "IPIP Model",
    "Logs",
]
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
    # Use the new IPIP specific evolution section
    show_ipip_evolution_section(pipeline_name)

with tabs[5]:
    show_ipip_section(pipeline_name)

with tabs[6]:
    show_logs_section(pipeline_name)
