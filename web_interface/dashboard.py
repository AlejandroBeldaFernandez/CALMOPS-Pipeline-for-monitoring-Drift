import streamlit as st
import pandas as pd
import json
import os
import re
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import argparse
import numpy as np
from typing import Optional, Tuple, Dict, List
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from utils import _load_any_dataset, load_log, dashboard_data_loader, actualizar_registro
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')
# =========================
# Streamlit Configuration & Performance Optimization
# =========================
# High-performance dashboard for ML pipeline monitoring
# Features caching, optimized data loading, and statistical visualizations

st.set_page_config(page_title="Monitor ML Pipeline", layout="wide")
st.title("ML Pipeline Monitor")



def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def _load_json_cached(path: str, stamp: float) -> dict:
    """Load JSON file with Streamlit caching for performance optimization.
    
    Args:
        path: File path to JSON file
        stamp: Modification timestamp for cache invalidation
        
    Returns:
        Parsed JSON data as dictionary
    """
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def _load_csv_cached(path: str, stamp: float) -> pd.DataFrame:
    """Load CSV file with Streamlit caching for performance optimization.
    
    Args:
        path: File path to CSV file
        stamp: Modification timestamp for cache invalidation
        
    Returns:
        Pandas DataFrame with loaded data
    """
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def _load_any_cached(path: str, stamp: float) -> pd.DataFrame:
    """Load any supported dataset format with caching.
    
    Supports multiple formats: CSV, ARFF, JSON, Excel, Parquet, TXT
    Uses modification timestamp for efficient cache invalidation.
    
    Args:
        path: File path to dataset
        stamp: Modification timestamp for cache invalidation
        
    Returns:
        Pandas DataFrame with loaded dataset
    """
    return _load_any_dataset(path)

def _sample_series(values: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Efficiently sample large arrays for visualization performance.
    
    Reduces large datasets to manageable size for plotting while maintaining
    statistical representativeness through random sampling.
    
    Args:
        values: Input array to sample
        max_points: Maximum number of points to return
        seed: Random seed for reproducible sampling
        
    Returns:
        Sampled array with at most max_points elements
    """
    n = values.shape[0]
    if n <= max_points:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return values[idx]

def _ecdf_quantile_curve(values: np.ndarray, q_points: int = 512, logx: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generate empirical cumulative distribution function curve.
    
    Creates smooth ECDF curves for statistical visualization by computing
    quantile points. Supports logarithmic x-axis scaling for skewed distributions.
    
    Args:
        values: Input data array
        q_points: Number of quantile points to compute (higher = smoother)
        logx: Apply log transformation to x-axis (excludes non-positive values)
        
    Returns:
        Tuple of (x_values, cumulative_probabilities) for plotting
    """
    
    q = np.linspace(0.0, 1.0, q_points, endpoint=True)
    x = np.quantile(values, q)
    y = q
    if logx:
      
        x = np.where(x <= 0, np.nan, x)
        mask = ~np.isnan(x)
        return x[mask], y[mask]
    return x, y

def _paired_hist(prev: np.ndarray, curr: np.ndarray, bins: int, density: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired histograms for drift detection visualization.
    
    Computes histograms for two datasets using identical bins for direct comparison.
    Handles edge cases like empty arrays and identical min/max values gracefully.
    
    Args:
        prev: Previous/reference dataset
        curr: Current dataset to compare
        bins: Number of histogram bins
        density: Whether to normalize histograms to probability densities
        
    Returns:
        Tuple of (prev_histogram, curr_histogram, bin_centers)
    """

    if prev.size == 0 or curr.size == 0:
        return np.array([]), np.array([]), np.array([])
    vmin = float(min(np.nanmin(prev), np.nanmin(curr)))
    vmax = float(max(np.nanmax(prev), np.nanmax(curr)))
 
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
        eps = 1.0 if not np.isfinite(vmin) or not np.isfinite(vmax) else max(1e-9, abs(vmin) * 0.01 or 1.0)
        vmin, vmax = (0.0 - eps, 0.0 + eps) if not np.isfinite(vmin) or not np.isfinite(vmax) else (vmin - eps, vmax + eps)
    hist_prev, edges = np.histogram(prev, bins=bins, range=(vmin, vmax), density=density)
    hist_curr, _     = np.histogram(curr, bins=bins, range=(vmin, vmax), density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist_prev, hist_curr, centers

# =========================
# Dataset Section
# =========================
def show_dataset_section(data_dir, pipeline_name):
    """Displays dataset preview, info, stats, top categorical values, and a drift heatmap."""
    st.subheader("Dataset Information")
    control_dir = os.path.join("pipelines", pipeline_name, "control")

    df, last_file = dashboard_data_loader(data_dir, control_dir)

    if df.empty or not last_file:
        st.warning("No processed dataset found yet.")
        return

    st.write(f"**Last processed dataset:** `{last_file}`")

    # Preview (head)
    st.markdown("### Data Preview (head)")
    st.dataframe(df.head(10), use_container_width=True)

    # Info table (ligero y vectorizado)
    st.markdown("### Dataset Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": df.nunique(dropna=True).values,
        "Dtype": df.dtypes.astype(str).values
    })
    st.dataframe(info_df, use_container_width=True)

    # Show df.info()
    st.markdown("### Dataset Info (`df.info()`)")
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    # Summary
    st.markdown(f"""
    **Total Rows:** {df.shape[0]}  
    **Total Columns:** {df.shape[1]}  
    **Memory Usage:** {df.memory_usage().sum() / 1024**2:.2f} MB
    """)

    # Descriptive Statistics
    st.markdown("### Descriptive Statistics (`df.describe()`)")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    # Top valores categóricos (limita Nº columnas para evitar loops caros)
    st.markdown("### Top 5 Most Frequent Values in Categorical Columns")
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        max_cols = st.slider("Max. categorical columns to display", 1, min(20, len(cat_cols)), min(5, len(cat_cols)))
        show_cols = list(cat_cols[:max_cols])
        for col in show_cols:
            st.markdown(f"**{col}**")
            freq = df[col].value_counts().head(5).reset_index()
            freq.columns = [col, "Frequency"]
            fig = px.bar(
                freq, x=col, y="Frequency", text="Frequency",
                title=f"Top 5 in {col}", labels={col: "Values", "Frequency": "Frequency"}
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-30, height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns to show frequencies.")

    # Processed Files History
    st.markdown("### Processed Files History")
    control_file_path = os.path.join(control_dir, "control_file.txt")
    if os.path.exists(control_file_path):
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
            df_processed = pd.DataFrame(processed_files_data)
            df_processed = df_processed.sort_values(by="Processed At", ascending=True).reset_index(drop=True)
            st.dataframe(df_processed, use_container_width=True)
        else:
            st.info("No files recorded in control_file.txt yet.")
    else:
        st.info("control_file.txt not found. No processing history available.")

# =========================
# Evaluator Section
# =========================
def show_evaluator_section(pipeline_name):
    """Displays evaluation metrics for the approved model, including thresholds, circuit breaker status, and an overview of candidates."""
    st.subheader("Approved Model Evaluation Results")
    st.markdown("This section provides a detailed analysis of the currently approved **Champion** model's performance on the test set.")

    base_dir = os.path.join("pipelines", pipeline_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    candidates_dir = os.path.join(base_dir, "candidates")
    eval_path = os.path.join(metrics_dir, "eval_results.json")
    health_path = os.path.join(metrics_dir, "health.json")

    if not os.path.exists(eval_path):
        st.info("No evaluation results found yet.")
        return None  # ← importante

    results = _load_json_cached(eval_path, _mtime(eval_path))
    if not results:
        st.warning("Empty evaluation results.")
        return None  # ← importante

    # Approval flag
    if results.get("approved", False):
        _ = st.success("Model approved. Meets the established thresholds.")
    else:
        _ = st.error("Model NOT approved. Does not meet the established thresholds.")

    # Thresholds
    _ = st.markdown("## Used Thresholds")
    thresholds = results.get("thresholds", {})
    if thresholds:
        thresholds_df = pd.DataFrame(list(thresholds.items()), columns=["Metric", "Threshold"])
        _ = st.table(thresholds_df)  
    else:
        _ = st.info("No thresholds found in the evaluation results.")

    # Prediction examples
    if "predictions" in results and results["predictions"]:
        _ = st.markdown("## Prediction Examples")
        preds_df = pd.DataFrame(results["predictions"])
        _ = st.table(preds_df) 

    # Metrics
    _ = st.markdown("## Test Metrics")
    metrics = results.get("metrics", {})
    if "classification_report" in metrics:
        _ = st.write(f"**Accuracy:** {round(metrics.get('accuracy', 0), 4)}")
        _ = st.write(f"**Balanced Accuracy:** {round(metrics.get('balanced_accuracy', 0), 4)}")
        _ = st.write(f"**F1 (macro):** {round(metrics.get('f1', 0), 4)}")

        report_df = pd.DataFrame(metrics["classification_report"])
        if set(["precision", "recall", "f1-score", "support"]).issubset(report_df.index):
            _ = st.table(report_df.transpose())
        else:
            _ = st.table(report_df)
        
        # Confusion Matrix
        if "predictions" in results and results["predictions"]:
            st.markdown("### Confusion Matrix")
            preds_df = pd.DataFrame(results["predictions"])
            y_true = preds_df["y_true"]
            y_pred = preds_df["y_pred"]
            labels = sorted(list(set(y_true) | set(y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig = px.imshow(cm,
                            labels=dict(x="Predicted Label", y="True Label"),
                            x=labels,
                            y=labels,
                            text_auto=True)
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)

        if 'roc_auc' in metrics and 'roc_curve' in metrics:
            st.write(f"**ROC AUC:** {round(metrics['roc_auc'], 4)}")
            st.markdown("### ROC Curve")
            roc_data = metrics['roc_curve']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'],
                                mode='lines',
                                name=f"ROC (AUC = {metrics['roc_auc']:.4f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random Chance',
                                line=dict(dash='dash')))
            fig.update_layout(title='ROC Curve',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)

    elif "r2" in metrics:
        _ = st.write(f"**R²:** {round(metrics.get('r2', 0), 4)}")
        _ = st.write(f"**RMSE:** {round(metrics.get('rmse', 0), 4)}")
        _ = st.write(f"**MAE:** {round(metrics.get('mae', 0), 4)}")
        _ = st.write(f"**MSE:** {round(metrics.get('mse', 0), 4)}")

        # Residual Plot
        if "predictions" in results and results["predictions"]:
            st.markdown("### Residual Plot")
            preds_df = pd.DataFrame(results["predictions"])
            y_true = preds_df["y_true"]
            y_pred = preds_df["y_pred"]
            residuals = y_true - y_pred
            fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title='Residual Plot')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    else:
        _ = st.info("No evaluation metrics found.")

    # Circuit Breaker status
    _ = st.markdown("## Circuit Breaker Status")
    if os.path.exists(health_path):
        try:
            health = _load_json_cached(health_path, _mtime(health_path))
            consecutive = int(health.get("consecutive_failures", 0) or 0)
            paused_until = health.get("paused_until")
            last_failure_ts = health.get("last_failure_ts")

            cols = st.columns(3)
            cols[0].metric("Consecutive Failures", consecutive)

            def _fmt_ts(ts):
                try:
                    import datetime, time
                    return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return str(ts)

            cols[1].metric("Paused Until", _fmt_ts(paused_until) if paused_until else "—")
            cols[2].metric("Last Failure", _fmt_ts(last_failure_ts) if last_failure_ts else "—")

            import time as _t
            paused = bool(paused_until and _t.time() < float(paused_until))
            if paused:
                _ = st.warning("Retraining is currently **paused** by the circuit breaker.")
            else:
                _ = st.success("Retraining is **active** (not paused).")
        except Exception as e:
            _ = st.warning(f"Could not read health.json: {e}")
    else:
        _ = st.info("No circuit breaker state found yet (health.json).")

    # Candidates overview
    _ = st.markdown("## Candidates (Non-Approved Models)")
    if not os.path.exists(candidates_dir):
        _ = st.info("No candidates directory yet.")
        return None 

    candidates = []
    try:
        for entry in sorted(
            [os.path.join(candidates_dir, d) for d in os.listdir(candidates_dir) if os.path.isdir(os.path.join(candidates_dir, d))],
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )[:10]:
            meta_path = os.path.join(entry, "meta.json")
            eval_p = os.path.join(entry, "eval_results.json")
            row = {"path": entry, "timestamp": None, "file": None, "approved": False, "key_metric": None, "metric_value": None}
            try:
                if os.path.exists(meta_path):
                    meta = _load_json_cached(meta_path, _mtime(meta_path))
                    row["approved"] = bool(meta.get("approved", False))
                    row["file"] = meta.get("file")
                    row["timestamp"] = meta.get("timestamp")
                if os.path.exists(eval_p):
                    ev = _load_json_cached(eval_p, _mtime(eval_p))
                    m = ev.get("metrics", {})
                    if "accuracy" in m:
                        row["key_metric"] = "accuracy"; row["metric_value"] = m.get("accuracy")
                    elif "f1" in m:
                        row["key_metric"] = "f1"; row["metric_value"] = m.get("f1")
                    elif "r2" in m:
                        row["key_metric"] = "r2"; row["metric_value"] = m.get("r2")
            except Exception:
                pass
            candidates.append(row)

        if candidates:
            df_cand = pd.DataFrame(candidates)
            if "timestamp" in df_cand.columns:
                df_cand = df_cand.sort_values(by="timestamp", ascending=False, na_position="last")
            show_cols = ["timestamp", "file", "approved", "key_metric", "metric_value", "path"]
            _ = st.dataframe(df_cand[show_cols], use_container_width=True)  # ← captura el DeltaGenerator
            _ = st.caption("Showing up to 10 latest candidates. Each folder contains `model.pkl` and `eval_results.json`.")
        else:
            _ = st.info("No candidates have been saved yet.")
    except Exception as e:
        _ = st.warning(f"Could not enumerate candidates: {e}")

    return None  

def show_historical_performance_section(pipeline_name):
    """Plots the historical performance of Champion and Challenger models over time, including the final approved Champion."""
    st.subheader("Historical Model Performance")
    st.markdown("This chart shows the `balanced_accuracy` of models over time, based on drift detection runs and final evaluations.")

    base_dir = os.path.join("pipelines", pipeline_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    drift_history_dir = os.path.join(metrics_dir, "drift_history")

    history_data = []

    # 1. Load data from drift history
    if os.path.exists(drift_history_dir):
        history_files = sorted([f for f in os.listdir(drift_history_dir) if f.startswith("drift_results_") and f.endswith(".json")])
        for file_name in history_files:
            file_path = os.path.join(drift_history_dir, file_name)
            try:
                results = _load_json_cached(file_path, _mtime(file_path))
                timestamp_str = file_name.replace("drift_results_", "").replace(".json", "")
                timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
                tests = results.get("tests", {})

                # Extract Champion performance from drift check
                if "Performance_Previous" in tests and tests["Performance_Previous"].get("balanced_accuracy"):
                    champ_perf = tests["Performance_Previous"]["balanced_accuracy"].get("balanced_accuracy")
                    if champ_perf is not None:
                        history_data.append({"timestamp": timestamp, "model_type": "Champion", "balanced_accuracy": champ_perf})

                # Extract Challenger performance from drift check
                if "Performance_Current" in tests and tests["Performance_Current"].get("balanced_accuracy"):
                    chall_perf = tests["Performance_Current"]["balanced_accuracy"].get("balanced_accuracy")
                    if chall_perf is not None:
                        history_data.append({"timestamp": timestamp, "model_type": "Challenger", "balanced_accuracy": chall_perf})
            except Exception as e:
                st.warning(f"Could not read or parse {file_name}: {e}")

    # 2. Load data from the final evaluator result for the current Champion
    eval_path = os.path.join(metrics_dir, "eval_results.json")
    if os.path.exists(eval_path):
        eval_results = _load_json_cached(eval_path, _mtime(eval_path))
        metrics = eval_results.get("metrics", {})
        timestamp = eval_results.get("timestamp")
        if metrics.get("balanced_accuracy") and timestamp:
            history_data.append({
                "timestamp": pd.to_datetime(timestamp),
                "model_type": "Champion", # Consolidate to 'Champion'
                "balanced_accuracy": metrics["balanced_accuracy"]
            })

    # 3. Load data from historical evaluation results
    eval_history_dir = os.path.join(metrics_dir, "eval_history")
    if os.path.exists(eval_history_dir):
        eval_history_files = sorted([f for f in os.listdir(eval_history_dir) if f.startswith("eval_results_") and f.endswith(".json")])
        for file_name in eval_history_files:
            file_path = os.path.join(eval_history_dir, file_name)
            try:
                results = _load_json_cached(file_path, _mtime(file_path))
                timestamp_str = file_name.replace("eval_results_", "").replace(".json", "")
                timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
                metrics = results.get("metrics", {})
                if metrics.get("balanced_accuracy"):
                    history_data.append({"timestamp": timestamp, "model_type": "Champion", "balanced_accuracy": metrics["balanced_accuracy"]}) # Consolidate to 'Champion'
            except Exception as e:
                st.warning(f"Could not read or parse historical eval file {file_name}: {e}")

    if not history_data:
        st.info("No historical performance data with balanced_accuracy found.")
        return

    df_history = pd.DataFrame(history_data)
    # Clean up data: sort by timestamp, then drop duplicates keeping the last entry for each timestamp and model_type
    df_history = df_history.sort_values(by="timestamp").drop_duplicates(subset=['timestamp', 'model_type'], keep='last')

    # Date range selector
    if not df_history.empty:
        min_date = df_history["timestamp"].min().date()
        max_date = df_history["timestamp"].max().date()
        date_range = st.date_input("Select date range for performance history", (min_date, max_date), min_value=min_date, max_value=max_date)

        if len(date_range) == 2:
            start_date, end_date = date_range
            df_history = df_history[(df_history["timestamp"].dt.date >= start_date) & (df_history["timestamp"].dt.date <= end_date)]

    st.dataframe(df_history)

    # Plot performance over time
    if not df_history.empty:
        fig = px.line(df_history, x="timestamp", y="balanced_accuracy", markers=True,
                      title="Historical Balanced Accuracy")
        fig.update_traces(connectgaps=True)  # Connect points over gaps
        st.plotly_chart(fig, use_container_width=True)


def show_drift_trends_section(pipeline_name):
    """
    Displays historical drift metrics trends.
    """
    st.markdown("## Drift Metric Trends Over Time")
    
    base_dir = os.path.join("pipelines", pipeline_name)
    drift_history_dir = os.path.join(base_dir, "metrics", "drift_history")

    if not os.path.exists(drift_history_dir):
        st.info("No drift history found.")
        return

    history_files = sorted([f for f in os.listdir(drift_history_dir) if f.startswith("drift_results_") and f.endswith(".json")])

    if not history_files:
        st.info("No historical drift data found.")
        return

    all_results = []
    for file_name in history_files:
        file_path = os.path.join(drift_history_dir, file_name)
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
                # Extract timestamp from filename for sorting
                timestamp_str = file_name.replace("drift_results_", "").replace(".json", "")
                results["timestamp"] = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
                all_results.append(results)
        except Exception as e:
            st.warning(f"Could not read or parse {file_name}: {e}")
    
    if not all_results:
        st.warning("No valid historical data could be loaded.")
        return

    # Extract all features and metrics from the historical data
    all_features = set()
    all_metrics = set()
    for result in all_results:
        tests = result.get("tests", {})
        for test_name, test_results in tests.items():
            if isinstance(test_results, dict):
                for feature_name, metrics in test_results.items():
                    all_features.add(feature_name)
                    if isinstance(metrics, dict):
                        for metric_name in metrics.keys():
                            all_metrics.add(metric_name)
    
    if not all_features:
        st.info("No features found in historical drift data.")
        return

    # Date range selector
    min_date = min([res["timestamp"] for res in all_results])
    max_date = max([res["timestamp"] for res in all_results])
    date_range = st.date_input("Select date range", (min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        start_date, end_date = date_range
        all_results = [res for res in all_results if start_date <= res["timestamp"].date() <= end_date]

    selected_feature = st.selectbox("Select Feature for Trend Analysis", options=sorted(list(all_features)))
    
    if not selected_feature:
        return

    feature_data = []
    for result in all_results:
        timestamp = result["timestamp"]
        tests = result.get("tests", {})
        row = {"timestamp": timestamp}
        for test_name, test_results in tests.items():
            if isinstance(test_results, dict) and selected_feature in test_results:
                metrics = test_results[selected_feature]
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            row[f"{test_name}_{metric_name}"] = value
        feature_data.append(row)

    if not feature_data:
        st.info(f"No historical data found for feature: {selected_feature}")
        return

    df_trends = pd.DataFrame(feature_data)
    df_trends = df_trends.set_index("timestamp")
    
    if df_trends.empty:
        st.info(f"No plottable data for feature: {selected_feature}")
        return

    st.dataframe(df_trends)

   

# =========================
# Drift Section (lightweight)
# =========================
def show_drift_section(pipeline_name):
    """
    Shows a simplified summary of drift detection results, comparing the
    Champion (previous) vs. Challenger (current) models. It focuses on
    high-level drift flags and performance comparison rather than per-feature plots.
    """
    st.subheader("Drift Analysis: Champion vs. Challenger")
    st.markdown("This section analyzes for data and model drift, comparing the established **Champion** modelTrains and evaluates a new **Challenger** model on recent data.")

    base_dir = os.path.join("pipelines", pipeline_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    drift_path = os.path.join(metrics_dir, "drift_results.json")

    if not os.path.exists(drift_path):
        st.info("No drift results saved yet.")
        return
    results = _load_json_cached(drift_path, _mtime(drift_path))
    if not results:
        st.warning("Drift results are empty.")
        return

    tests = results.get("tests", {})
    drift_flags = results.get("drift", {})

    # Decision badge
    decision = results.get("decision")
    if decision:
        if decision == "no_drift":
            st.success("No drift detected – Champion model maintained")
        elif decision == "previous_promoted":
            reason = results.get("promotion_reason")
            st.warning("Champion model promoted due to better performance" + (f" (reason: `{reason}`)" if reason else ""))
        elif decision == "retrain":
            st.error("Retraining triggered due to drift, creating a new Challenger")
        elif decision == "train":
            st.info("First run: training initial Champion model")
        elif decision == "end_error":
            st.warning("Ended with error while loading current model")
        else:
            st.info(f"Decision: {decision}")

    # Summary flags
    if isinstance(drift_flags, dict) and drift_flags:
        st.markdown("## Drift Test Summary")
        st.markdown("Indicates if drift was detected in any variable for each statistical test.")
        promoted_key = "promoted_model" if "promoted_model" in results else None
        if promoted_key is not None:
            val = results.get(promoted_key)
            if val is True:
                st.success("The previous model was promoted to the current one.")
                reason = results.get("promotion_reason")
                if reason: st.info(f"**Promotion reason:** `{reason}`")
            elif val is False:
                st.info("The previous model was NOT promoted.")
            elif val == "error":
                st.warning("Error comparing with the previous model.")

        summary_data = []
        for k, v in drift_flags.items():
            # We only show the general test result, not per variable
            if "::" not in k:  # Avoid showing per-variable drift flags
                summary_data.append({"Test": k, "Result": "Drift detected" if bool(v) else "No drift detected"})
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            total = len(summary_df)
            detected = (summary_df["Result"].str.contains("Drift")).sum()
            st.success(f"Tests without drift: {total - detected} of {total}")
            st.error(f"Tests with drift: {detected} of {total}")

            def color_drift_results(val):
                if val == "Drift detected":
                    return 'background-color: #ffe6e6; color: red'  # Light red background, red text
                elif val == "No drift detected":
                    return 'background-color: #e6ffe6; color: green' # Light green background, green text
                return ''

            st.dataframe(summary_df.style.applymap(color_drift_results, subset=['Result']), use_container_width=True)
        else:
            st.info("No drift results found.")
            return

    # Performance checks (compact)
    st.markdown("## Performance Checks")

    def _render_perf_table(perf_dict: dict, title: str):
        if not isinstance(perf_dict, dict) or len(perf_dict) == 0:
            return
        rows = []
        for metric_name, payload in perf_dict.items():
            # Extract value more robustly, prioritizing specific metric name
            value = payload.get(metric_name) # Try exact metric name first
            if value is None: # Fallback to common names if exact not found
                value = (
                    payload.get(metric_name.lower())
                    or payload.get("accuracy")
                    or payload.get("balanced_accuracy")
                    or payload.get("F1")
                    or payload.get("RMSE")
                    or payload.get("R2")
                    or payload.get("MAE")
                    or payload.get("MSE")
                    or payload.get("value") # Generic 'value' key
                )
            rows.append({
                "Metric": metric_name,
                "Value": f"{value:.4f}" if isinstance(value, float) else value,
                "Threshold": payload.get("threshold"),
                "Drift": bool(payload.get("drift", False))
            })
        st.markdown(f"### {title}")
        st.table(pd.DataFrame(rows))

    if "Performance_Current" in tests:
        _render_perf_table(tests["Performance_Current"], "Challenger Model vs Thresholds")
    if "Performance_Previous" in tests:
        _render_perf_table(tests["Performance_Previous"], "Champion Model vs Thresholds")
    if "Performance_Comparison" in tests and isinstance(tests["Performance_Comparison"], dict):
        comp_rows = []
        for mname, payload in tests["Performance_Comparison"].items():
            metric = payload.get("metric", mname)
            prev_v = payload.get("prev"); curr_v = payload.get("current")
            thr = payload.get("threshold")
            change = payload.get("relative_drop", payload.get("relative_increase"))
            drift_key = f"comparison::{metric.split('_')[0]}"
            comp_rows.append({
                "Metric": metric,
                "Champion": f"{prev_v:.4f}" if isinstance(prev_v, float) else prev_v,
                "Challenger": f"{curr_v:.4f}" if isinstance(curr_v, float) else curr_v,
                "Relative Performance Change": f"{float(change)*100:.2f}%" if change is not None else "N/A",
                "Threshold": thr,
                "Drift": bool(drift_flags.get(drift_key, False))
            })
        if comp_rows:
            st.markdown("### Champion vs Challenger (Relative Performance)")
            st.table(pd.DataFrame(comp_rows))

            # Bar chart comparison
            df_comp = pd.DataFrame(comp_rows)
            if not df_comp.empty:
                # Ensure Champion and Challenger are numeric for plotting
                df_comp["Champion"] = pd.to_numeric(df_comp["Champion"], errors='coerce')
                df_comp["Challenger"] = pd.to_numeric(df_comp["Challenger"], errors='coerce')
                
                df_melted = df_comp.melt(id_vars=["Metric"], value_vars=["Champion", "Challenger"], var_name="Model", value_name="Score")
                fig = px.bar(df_melted, x="Metric", y="Score", color="Model", barmode="group", title="Champion vs Challenger Performance")
                st.plotly_chart(fig, use_container_width=True)

    # Heatmap (from drift results)
    st.markdown("## Drift Heatmap")
    # base_dir and metrics_dir are already defined at the beginning of show_drift_section
    drift_path = os.path.join(metrics_dir, "drift_results.json")

    if os.path.exists(drift_path):
        results = _load_json_cached(drift_path, _mtime(drift_path))
        tests = results.get("tests", {})

        # Dynamically identify available metric types from the keys
        available_metric_types = set()
        for k in tests.keys():
            if k.endswith(('_ks', '_mw', '_psi', '_chi2')):
                if k.endswith('_ks'): available_metric_types.add('KS p-value')
                elif k.endswith('_mw'): available_metric_types.add('MW p-value')
                elif k.endswith('_psi'): available_metric_types.add('PSI')
                elif k.endswith('_chi2'): available_metric_types.add('Chi2 p-value')
        
        metric_options = sorted(list(available_metric_types))

        if not metric_options:
            st.info("No per-feature drift metrics found to build a heatmap.")
        else:
            msel = st.selectbox("Heatmap metric", options=metric_options, index=0, key="drift_heatmap_metric")
            topk = st.slider("Top K features to display", min_value=10, max_value=200, value=50, step=10, key="drift_heatmap_topk")

            feat_vals: Dict[str, Optional[float]] = {}
            ordered: List[Tuple[str, float]] = []

            for k, payload in tests.items():
                feature_name = k.rsplit('_', 1)[0] # Extract feature name
                if msel == "KS p-value" and k.endswith('_ks'):
                    try:
                        v = payload.get("p_value")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None
                elif msel == "MW p-value" and k.endswith('_mw'):
                    try:
                        v = payload.get("p_value")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None
                elif msel == "PSI" and k.endswith('_psi'):
                    try:
                        v = payload.get("psi")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None
                elif msel == "Chi2 p-value" and k.endswith('_chi2'):
                    try:
                        v = payload.get("p_value")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None
            
            # Sort based on the selected metric type
            if msel in ("KS p-value", "MW p-value", "Chi2 p-value"):
                # For p-values, lower is more significant, so sort ascending
                ordered = sorted([(k, v) for k, v in feat_vals.items() if v is not None], key=lambda x: x[1])
            elif msel == "PSI":
                # For PSI, higher indicates more drift, so sort descending by absolute value
                ordered = sorted([(k, v) for k, v in feat_vals.items() if v is not None], key=lambda x: -abs(x[1]))

            if not ordered:
                st.info("No values available for the selected metric.")
            else:
                ordered = ordered[:topk]
                heat_df = pd.DataFrame({"feature": [k for k, _ in ordered], "value": [v for _, v in ordered]})
                if heat_df.empty:
                    st.info("No numeric values available for heatmap.")
                else:
                    # Define a custom colorscale for warmer colors and white for 0.0
                    
                    colorscale = [
                        [0.0, 'rgb(255, 255, 255)'],    # 1. White at 0.0
                        [0.001, 'rgb(255, 253, 208)'], # 2. Lightest yellow just above 0
                        [0.2, 'rgb(255, 187, 120)'],   # 3. Light Orange
                        [0.4, 'rgb(255, 140, 0)'],    # 4. Orange
                        [0.6, 'rgb(230, 84, 0)'],     # 5. Dark Orange/Red
                        [0.8, 'rgb(204, 41, 0)'],     # 6. Red
                        [1.0, 'rgb(139, 0, 0)']       # 7. Dark Red
                    ]

                    heat_fig = go.Figure(
                        data=go.Heatmap(
                            z=heat_df["value"].values.reshape(-1, 1), # Reshape for matrix-like view
                            x=[msel], # Single column for the metric
                            y=heat_df["feature"].tolist(), # Features as rows
                            colorscale=colorscale, # Apply custom colorscale
                            xgap=1, ygap=1 # Add gaps for better visual separation
                        )
                    )
                    heat_fig.update_layout(
                        title=f"Drift heatmap by feature – {msel} (Top {topk})",
                        xaxis=dict(title=msel, tickangle=0), # Metric name on x-axis
                        yaxis=dict(title="Feature"), # Features on y-axis
                        height=min(800, 50 * len(ordered)) # Adjust height dynamically
                    )
                    st.plotly_chart(heat_fig, use_container_width=True)

    else:
        st.info("Drift results file (`drift_results.json`) not found. Heatmap cannot be displayed.")

    # Show drift trends
    show_drift_trends_section(pipeline_name)

# =========================
# Train / Retrain Section
# =========================
def show_train_section(pipeline_name):
    """
    Displays training or retraining metrics, providing detailed explanations for the retraining methods used.
    Expects JSON at: pipelines/{pipeline_name}/metrics/train_results.json
    """
    st.subheader("Training / Retraining Results")
    train_path = os.path.join("pipelines", pipeline_name, "metrics", "train_results.json")

    if not os.path.exists(train_path):
        st.info("No training results found yet.")
        return

    results = _load_json_cached(train_path, _mtime(train_path))
    if not results:
        st.warning("Empty training results.")
        return

    MODE_LABELS = {
        0: "Full retraining",
        1: "Incremental (partial_fit)",
        2: "Windowed retraining (rolling)",
        3: "Ensemble old + new (stacking)",
        4: "Stacking old + cloned(old)",
        5: "Replay mix (prev + current)",
        6: "Recalibration (Platt/Isotonic)",
    }
    
    MODE_DESCRIPTIONS = {
        0: "The model is retrained from scratch on the new dataset, completely replacing the old model.",
        1: "The model is updated with the new data using `partial_fit`, suitable for models that support incremental learning.",
        2: "The model is retrained on a rolling window of the most recent data, discarding the oldest samples.",
        3: "A new model is trained on the new data, and an ensemble is created by stacking it with the old model.",
        4: "An ensemble is created by stacking the old model with a clone of itself that has been retrained on the new data.",
        5: "The model is retrained on a dataset composed of a mix of previous data and the new current data.",
        6: "The model’s probability outputs are recalibrated using methods like Platt Scaling or Isotonic Regression on the new data.",
    }

    st.markdown("## General Information")
    _type = results.get("type", "-")
    _file = results.get("file", "-")
    _date = results.get("timestamp", "-")
    _model = results.get("model", "-")

    cols = st.columns(2)
    with cols[0]:
        st.write(f"**Type:** {_type}")
        st.write(f"**File:** {_file}")
        st.write(f"**Date:** {_date}")
    with cols[1]:
        st.write(f"**Model:** {_model}")
        _strategy = results.get("strategy")
        if _strategy:
            st.write(f"**Strategy:** {_strategy}")

        if _type == "retrain":
            mode = results.get("mode", None)
            fallback = results.get("fallback", None)
            if mode is not None:
                label = MODE_LABELS.get(mode, f"Unknown ({mode})")
                st.write(f"**Mode:** {mode} — {label}")
                # Show the description for the used mode
                description = MODE_DESCRIPTIONS.get(mode)
                if description:
                    st.info(description)

            if fallback is not None:
                st.markdown("**Fallback:** " + ("Enabled (used)" if bool(fallback) else "Not used"))

            gs = results.get("gridsearch", None)
            if isinstance(gs, dict) and gs:
                st.markdown("**GridSearchCV:**")
                if "best_params" in gs:
                    bp = gs["best_params"]
                    if isinstance(bp, dict) and bp:
                        st.table(pd.DataFrame([bp]).transpose().rename(columns={0: "best_params"}))
                    else:
                        st.write(bp)
                if "cv" in gs and gs["cv"] is not None:
                    st.write(f"**CV folds:** {gs['cv']}")
            elif isinstance(gs, str):
                st.write(f"**GridSearchCV:** {gs}")

            extra_rows = []
            for k in ["window_size_used", "window_size"]:
                if k in results:
                    extra_rows.append(("Window size", results[k])); break
            if "calibration" in results:      extra_rows.append(("Calibration", results["calibration"]))
            if "replay_frac_old" in results:  extra_rows.append(("Replay frac (old)", results["replay_frac_old"]))
            if "meta" in results:             extra_rows.append(("Meta", results["meta"]))
            if "components" in results:       extra_rows.append(("Ensemble components", results["components"]))
            for k, nice in [("train_size", "Train size"), ("eval_size", "Eval size"),
                            ("epochs", "Epochs"), ("learning_rate", "Learning rate"),
                            ("early_stopping", "Early stopping")]:
                if k in results:
                    extra_rows.append((nice, results[k]))
            if extra_rows:
                st.markdown("**Extra retrain details:**")
                st.table(pd.DataFrame(extra_rows, columns=["Key", "Value"]))

    # Métrics
    st.markdown("## Metrics")
    if "classification_report" in results:
        st.markdown("### Classification Metrics")
        bal_acc = results.get("balanced_accuracy", results.get("balanced_accuracy_score"))
        if bal_acc is not None:
            try:
                st.write(f"**Balanced Accuracy:** {float(bal_acc):.4f}")
            except Exception:
                st.write(f"**Balanced Accuracy:** {bal_acc}")
        clf_report = results["classification_report"]
        clf_report_df = pd.DataFrame(clf_report)
        st.dataframe(clf_report_df.transpose() if set(["precision", "recall", "f1-score", "support"]).issubset(clf_report_df.index) else clf_report_df, use_container_width=True)
    elif all(k in results for k in ["r2", "rmse", "mae", "mse"]):
        st.markdown("### Regression Metrics")
        try:
            st.write(f"**R2:** {float(results['r2']):.4f}")
            st.write(f"**RMSE:** {float(results['rmse']):.4f}")
            st.write(f"**MAE:** {float(results['mae']):.4f}")
            st.write(f"**MSE:** {float(results['mse']):.4f}")
        except Exception:
            st.write(f"**R2:** {results['r2']}")
            st.write(f"**RMSE:** {results['rmse']}")
            st.write(f"**MAE:** {results['mae']}")
            st.write(f"**MSE:** {results['mse']}")
    else:
        st.warning("No valid metrics found in the training results.")

    if "error" in results:
        st.error(f"Error: {results['error']}")

    with st.expander("Raw training JSON"):
        st.json(results)

# =========================
# Logs Section
# =========================
def parse_logs_to_df(log_text: str) -> pd.DataFrame:
    """
    Parse log lines into a structured DataFrame with columns:
    date, pipeline, level, message.
    """
    log_pattern = re.compile(
        r'(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - '
        r'(?P<pipeline>[\w_]+) - (?P<level>\w+) - '
        r'(?P<message>.*)'
    )
    rows = []
    for line in log_text.splitlines():
        match = log_pattern.match(line)
        if match:
            rows.append(match.groupdict())
        else:
            rows.append({"date": None, "pipeline": None, "level": None, "message": line})
    return pd.DataFrame(rows)

def show_logs_section(pipeline_name):
    import os


    st.subheader("Pipeline Logs")
    logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline.log")
    error_logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline_errors.log")
    warning_logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline_warnings.log")

    # Function to read logs and delete DeltaGenerator messages
    def clean_log_content(log_text):

        # Ignore lines that start with DeltaGenerator
        lines = [line for line in log_text.splitlines() if not line.strip().startswith("DeltaGenerator")]
        return "\n".join(lines).strip()

    # General Log
    st.markdown("### General Log")
    general_log_text = clean_log_content(load_log(logs_path))
    df_general = parse_logs_to_df(general_log_text)
    if not df_general.empty:
        st.dataframe(
            df_general.style.map(
                lambda v: "color: red;" if v == "ERROR" else "color: orange;" if v == "WARNING" else "color: green;",
                subset=["level"]
            ),
            use_container_width=True
        )
    else:
        st.info("No general logs found.")

    # Error Log
    st.markdown("### Error Log")
    error_log_text = clean_log_content(load_log(error_logs_path))
    if error_log_text:
        st.code(error_log_text, language="bash")
    else:
        st.success("No errors found.")

    # Warning Log
    st.markdown("### Warning Log")
    warning_log_text = clean_log_content(load_log(warning_logs_path))
    if warning_log_text:
        st.code(warning_log_text, language="bash")
    else:
        st.success("No warnings found.")


# =========================
# Main App (Page Layout)
# =========================

# CLI args 
parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_name", type=str, required=True)
args, _ = parser.parse_known_args()
pipeline_name = args.pipeline_name

# Load config.json (pipeline/data_dir)
config_path = os.path.join("pipelines", pipeline_name, "config", "config.json")
if os.path.exists(config_path):
    config = _load_json_cached(config_path, _mtime(config_path))
    pipeline_name = config.get("pipeline_name", pipeline_name)
    data_dir = config.get("data_dir")
else:
    st.error("config.json not found. Please start the monitor with a valid pipeline.")
    st.stop()

if not data_dir:
    st.error("`data_dir` missing in config.json.")
    st.stop()

# Tabs
tabs = st.tabs(["Dataset", "Evaluator", "Drift", "Historical Performance", "Train/Retrain", "Logs"])

# =========================
# Auto-refresh on control_file change
# =========================
control_file = os.path.join("pipelines", pipeline_name, "control", "control_file.txt")


# Initialize timestamp in session_state in case that it doesn't exists
if "control_file_mtime" not in st.session_state:
    st.session_state.control_file_mtime = os.path.getmtime(control_file) if os.path.exists(control_file) else 0.0


# Check for changes
current_mtime = os.path.getmtime(control_file) if os.path.exists(control_file) else 0.0
if current_mtime != st.session_state.control_file_mtime:
    st.session_state.control_file_mtime = current_mtime
    st.experimental_rerun()

with tabs[0]:
    show_dataset_section(data_dir, pipeline_name)

with tabs[1]:
    show_evaluator_section(pipeline_name)

with tabs[2]:
    show_drift_section(pipeline_name)

with tabs[3]:
    show_historical_performance_section(pipeline_name)

with tabs[4]:
    show_train_section(pipeline_name)

with tabs[5]:
    show_logs_section(pipeline_name)
