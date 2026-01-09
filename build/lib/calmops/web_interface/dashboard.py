import os

# Suppress TensorFlow logs before importing it
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(# tf.compat.v1.logging.ERROR)
# tf.get_logger().setLevel("ERROR")

import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Optional, Tuple, Dict, List
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from calmops.utils import get_pipelines_root
from pathlib import Path
from utils import _load_any_dataset, load_log, dashboard_data_loader, update_record
from dashboard_common import (
    _mtime,
    _read_json_cached,
    _read_csv_cached,
    _load_any_dataset_cached,
    _sample_series,
    _ecdf_quantile_curve,
    _paired_hist,
)

# =========================
# Streamlit Configuration & Performance Optimization
# =========================
# High-performance dashboard for ML pipeline monitoring
# Features caching, optimized data loading, and statistical visualizations

st.set_page_config(page_title="Monitor ML Pipeline", layout="wide")
st.title("ML Pipeline Monitor")


# Helpers moved to dashboard_common.py


project_root = get_pipelines_root()


# =========================
# Dataset Section
# =========================
def show_dataset_section(data_dir, pipeline_name):
    """
    Displays the standard dataset information tab.
    Includes preview, lightweight info, descriptive stats, and categorical analysis.
    """
    """Displays dataset preview, info, stats, top categorical values, and a drift heatmap."""
    st.subheader("Dataset Information")
    control_dir = project_root / "pipelines" / pipeline_name / "control"

    df, last_file = dashboard_data_loader(data_dir, str(control_dir))

    if df.empty or not last_file:
        st.warning("No processed dataset found yet.")
        return

    st.write(f"**Last processed dataset:** `{last_file}`")

    # Preview (head)
    st.markdown("### Data Preview (head)")
    st.dataframe(df.head(10), use_container_width=True)

    # Info table (lightweight and vectorized)
    st.markdown("### Dataset Information")
    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Unique Values": df.nunique(dropna=True).values,
            "Dtype": df.dtypes.astype(str).values,
        }
    )
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

    # Categorical Variable Analysis
    st.markdown("### Categorical Variable Analysis")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        max_cols_cat = st.slider(
            "Max. categorical columns to display",
            1,
            min(20, len(cat_cols)),
            min(5, len(cat_cols)),
            key="cat_slider_dataset",
        )
        show_cols_cat = list(cat_cols[:max_cols_cat])

        for col in show_cols_cat:
            st.markdown(f"#### `{col}`")

            # Basic stats
            num_unique = df[col].nunique()
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Categories", num_unique)
            with col2:
                st.metric("Mode (Most Frequent)", str(mode_val))

            # Proportions
            proportions = df[col].value_counts(normalize=True)

            # Display as pie chart if few categories, otherwise as bar chart
            if num_unique <= 10:
                fig = px.pie(
                    proportions,
                    values=proportions.values,
                    names=proportions.index,
                    title=f"Proportions for `{col}`",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Category Proportions (Top 10)**")
                st.dataframe(proportions.head(10))
    else:
        st.info("No categorical columns to analyze.")

    # Processed Files History
    st.markdown("### Processed Files History")
    control_file_path = control_dir / "control_file.txt"
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
            df_processed = pd.DataFrame(processed_files_data)
            df_processed = df_processed.sort_values(
                by="Processed At", ascending=True
            ).reset_index(drop=True)
            st.dataframe(df_processed, use_container_width=True)
        else:
            st.info("No files recorded in control_file.txt yet.")
    else:
        st.info("control_file.txt not found. No processing history available.")


# =========================
# Evaluator Section
# =========================
def show_evaluator_section(pipeline_name):
    """
    Displays evaluation metrics for the approved model.
    Includes thresholds, metrics (Accuracy/F1/R2), confusion matrix, and ROC curves.
    """
    """Displays evaluation metrics for the approved model, including thresholds, circuit breaker status, and an overview of candidates."""
    st.subheader("Approved Model Evaluation Results")
    st.markdown(
        "This section provides a detailed analysis of the currently approved **Champion** model's performance on the test set."
    )

    base_dir = project_root / "pipelines" / pipeline_name
    metrics_dir = base_dir / "metrics"
    candidates_dir = base_dir / "candidates"
    eval_path = metrics_dir / "eval_results.json"
    health_path = metrics_dir / "health.json"

    if not eval_path.exists():
        st.info("No evaluation results found yet.")
        return None  # Important

    results = _read_json_cached(str(eval_path), _mtime(str(eval_path)))
    if not results:
        st.warning("Empty evaluation results.")
        return None  # Important

    # Approval flag
    if results.get("approved", False):
        _ = st.success("Model approved. Meets the established thresholds.")
    else:
        _ = st.error("Model NOT approved. Does not meet the established thresholds.")

    # Thresholds
    _ = st.markdown("## Used Thresholds")
    thresholds = results.get("thresholds", {})
    if thresholds:
        thresholds_df = pd.DataFrame(
            list(thresholds.items()), columns=["Metric", "Threshold"]
        )
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
        _ = st.write(
            f"**Balanced Accuracy:** {round(metrics.get('balanced_accuracy', 0), 4)}"
        )
        _ = st.write(f"**F1 (macro):** {round(metrics.get('f1', 0), 4)}")

        report_df = pd.DataFrame(metrics["classification_report"])
        if set(["precision", "recall", "f1-score", "support"]).issubset(
            report_df.index
        ):
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
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted Label", y="True Label"),
                x=labels,
                y=labels,
                text_auto=True,
            )
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        if "roc_auc" in metrics and "roc_curve" in metrics:
            st.write(f"**ROC AUC:** {round(metrics['roc_auc'], 4)}")
            st.markdown("### ROC Curve")
            roc_data = metrics["roc_curve"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode="lines",
                    name=f"ROC (AUC = {metrics['roc_auc']:.4f})",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Chance",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
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
            fig = px.scatter(
                x=y_pred,
                y=residuals,
                labels={"x": "Predicted Values", "y": "Residuals"},
                title="Residual Plot",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    else:
        _ = st.info("No evaluation metrics found.")

    # Candidates overview
    _ = st.markdown("## Candidates (Non-Approved Models)")
    if not candidates_dir.exists():
        _ = st.info("No candidates directory yet.")
        return None

    candidates = []
    try:
        for entry in sorted(
            [d for d in candidates_dir.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:10]:
            meta_path = entry / "meta.json"
            eval_p = entry / "eval_results.json"
            row = {
                "path": str(entry),
                "timestamp": None,
                "file": None,
                "approved": False,
                "key_metric": None,
                "metric_value": None,
            }
            try:
                if meta_path.exists():
                    meta = _read_json_cached(str(meta_path), _mtime(str(meta_path)))
                    row["approved"] = bool(meta.get("approved", False))
                    row["file"] = meta.get("file")
                    row["timestamp"] = meta.get("timestamp")
                if eval_p.exists():
                    ev = _read_json_cached(str(eval_p), _mtime(str(eval_p)))
                    m = ev.get("metrics", {})
                    if "accuracy" in m:
                        row["key_metric"] = "accuracy"
                        row["metric_value"] = m.get("accuracy")
                    elif "f1" in m:
                        row["key_metric"] = "f1"
                        row["metric_value"] = m.get("f1")
                    elif "r2" in m:
                        row["key_metric"] = "r2"
                        row["metric_value"] = m.get("r2")
            except Exception:
                pass
            candidates.append(row)

        if candidates:
            df_cand = pd.DataFrame(candidates)
            if "timestamp" in df_cand.columns:
                df_cand = df_cand.sort_values(
                    by="timestamp", ascending=False, na_position="last"
                )
            show_cols = [
                "timestamp",
                "file",
                "approved",
                "key_metric",
                "metric_value",
                "path",
            ]
            _ = st.dataframe(
                df_cand[show_cols], use_container_width=True
            )  # captures the DeltaGenerator
            _ = st.caption(
                "Showing up to 10 latest candidates. Each folder contains `model.pkl` and `eval_results.json`."
            )
        else:
            _ = st.info("No candidates have been saved yet.")
    except Exception as e:
        _ = st.warning(f"Could not enumerate candidates: {e}")

    return None


def show_historical_performance_section(pipeline_name):
    """
    Plots historical performance (Balanced Accuracy) of Champion vs Challenger models over time.
    """
    """
    Plots the historical performance of Champion and Challenger models over time
    by reading from a unified evaluation history directory.
    """
    st.subheader("Historical Model Performance")
    st.markdown(
        "This chart shows the `balanced_accuracy` of all model evaluations over time."
    )

    base_dir = project_root / "pipelines" / pipeline_name
    metrics_dir = base_dir / "metrics"
    eval_history_dir = metrics_dir / "eval_history"

    history_data = []

    # Unified data loading from eval_history
    if eval_history_dir.exists():
        eval_files = sorted(
            [
                f
                for f in eval_history_dir.iterdir()
                if f.name.startswith("eval_results_") and f.name.endswith(".json")
            ]
        )
        drift_files = sorted(
            [
                f
                for f in eval_history_dir.iterdir()
                if f.name.startswith("drift_results_") and f.name.endswith(".json")
            ]
        )
        history_files = eval_files + drift_files

        for file_path in history_files:
            try:
                results = _read_json_cached(str(file_path), _mtime(str(file_path)))

                # Safely extract required fields
                metrics = results.get("metrics", {})
                perf = metrics.get("balanced_accuracy")
                role = results.get(
                    "model_role", "Challenger"
                )  # Default to Challenger for old files

                # Timestamp can be in the file or from the filename
                timestamp = results.get("timestamp")
                if timestamp:
                    timestamp = pd.to_datetime(timestamp)
                else:
                    timestamp_str = file_path.stem.replace("eval_results_", "").replace(
                        "drift_results_", ""
                    )
                    timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")

                if perf is not None and timestamp is not None:
                    history_data.append(
                        {
                            "timestamp": timestamp,
                            "model_type": role,
                            "balanced_accuracy": float(perf),
                        }
                    )
            except Exception as e:
                st.warning(
                    f"Could not read or parse historical eval file {file_path.name}: {e}"
                )

    if not history_data:
        st.info("No historical performance data found in 'eval_history'.")
        return

    df_history = pd.DataFrame(history_data)

    st.markdown("### Raw Historical Data (Debug)")
    st.dataframe(df_history)

    # Clean up data: sort by timestamp, then drop duplicates keeping the last entry for each timestamp and model_type
    df_history = df_history.sort_values(by="timestamp").drop_duplicates(
        subset=["timestamp", "model_type"], keep="last"
    )

    # Date range selector
    if not df_history.empty:
        min_date = df_history["timestamp"].min().date()
        max_date = df_history["timestamp"].max().date()
        date_range = st.date_input(
            "Select date range for performance history",
            (min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            df_history = df_history[
                (df_history["timestamp"].dt.date >= start_date)
                & (df_history["timestamp"].dt.date <= end_date)
            ]

    # Display the filtered dataframe before plotting
    st.markdown("### Filtered Historical Data (Debug)")
    st.dataframe(df_history)

    # Plot performance over time
    if not df_history.empty:
        # Scatter plot for colored markers
        fig = px.scatter(
            df_history,
            x="timestamp",
            y="balanced_accuracy",
            color="model_type",
            title="Historical Balanced Accuracy (Champion vs. Challenger)",
            hover_data=["model_type"],
        )

        # Add a line connecting all points, without color differentiation
        fig.add_trace(
            go.Scatter(
                x=df_history["timestamp"],
                y=df_history["balanced_accuracy"],
                mode="lines",
                line=dict(color="grey"),  # A neutral color for the connecting line
                showlegend=False,
            )
        )  # Don't show this line in the legend

        # Ensure the markers are on top of the line
        fig.update_traces(marker=dict(size=10), selector=dict(mode="markers"))

        st.plotly_chart(fig, use_container_width=True)


def show_pca_drift_plot(df_previous, df_current):
    st.markdown("---")
    st.subheader("Drift Visualization with PCA")

    if df_previous.empty or df_current.empty:
        st.info("Both the reference and current dataset are needed for PCA comparison.")
        return

    # Select only numerical columns
    num_cols = df_previous.select_dtypes(include=np.number).columns.intersection(
        df_current.select_dtypes(include=np.number).columns
    )

    if len(num_cols) < 2:
        st.info("At least 2 common numerical features are needed for PCA analysis.")
        return

    df_prev_num = df_previous[num_cols].copy()
    df_curr_num = df_current[num_cols].copy()

    # Impute missing values with the mean
    df_prev_num.fillna(df_prev_num.mean(), inplace=True)
    df_curr_num.fillna(df_curr_num.mean(), inplace=True)

    # Scale data: fit on previous, transform both
    scaler = StandardScaler()
    scaled_prev = scaler.fit_transform(df_prev_num)
    scaled_curr = scaler.transform(df_curr_num)

    # Fit PCA on previous data and transform both
    pca = PCA(n_components=2)
    pca.fit(scaled_prev)

    principal_components_prev = pca.transform(scaled_prev)
    principal_components_curr = pca.transform(scaled_curr)

    # Create DataFrames for plotting
    pca_df_prev = pd.DataFrame(data=principal_components_prev, columns=["PC1", "PC2"])
    pca_df_prev["dataset"] = "Previous"

    pca_df_curr = pd.DataFrame(data=principal_components_curr, columns=["PC1", "PC2"])
    pca_df_curr["dataset"] = "Current"

    pca_df = pd.concat([pca_df_prev, pca_df_curr], ignore_index=True)

    # Plot
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="dataset",
        title="2D Dataset Comparison with PCA",
        labels={"color": "Dataset"},
        opacity=0.7,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    This chart projects the reference and current datasets into a two-dimensional space using Principal Component Analysis (PCA).
    - **Separate point clouds** suggest a significant change in the data distribution (drift).
    - **Overlapping point clouds** indicate that the datasets are structurally similar.
    """)


def show_pht_plot(config, pipeline_name):
    st.markdown("---")
    st.subheader("Concept Drift Analysis (PHT)")

    data_dir = config.get("data_dir")
    base_dir = get_pipelines_root() / "pipelines" / pipeline_name
    control_dir = base_dir / "control"
    control_file_path = control_dir / "control_file.txt"

    if not control_file_path.exists():
        st.info("`control_file.txt` not found. Cannot generate PHT plot.")
        return

    with open(control_file_path, "r") as f:
        processed_files = [line.strip().split(",")[0] for line in f.readlines()]

    if len(processed_files) < 2:
        st.info("At least 2 processed datasets are needed for PHT analysis.")
        return

    if len(processed_files) == 2:
        num_windows = 2
        st.info("Only 2 processed datasets available. Showing comparison between them.")
    else:
        num_windows = st.slider(
            "Number of time windows to compare",
            2,
            min(10, len(processed_files)),
            min(5, len(processed_files)),
            key="pht_windows",
        )

    # Get last N files
    files_to_load = processed_files[-num_windows:]

    datasets = {}
    for fname in files_to_load:
        fpath = Path(data_dir) / fname
        if fpath.exists():
            datasets[fname] = pd.read_csv(fpath)
        else:
            st.warning(f"Could not find dataset: {fpath}")

    if not datasets:
        st.error("Could not load datasets for PHT analysis.")
        return

    all_cols = next(iter(datasets.values())).columns
    num_cols = next(iter(datasets.values())).select_dtypes(include=np.number).columns

    selected_features = st.multiselect(
        "Select features to analyze",
        options=num_cols,
        default=list(num_cols[: min(3, len(num_cols))]),
    )

    if not selected_features:
        st.info("Please select at least one feature.")
        return

    for feature in selected_features:
        st.markdown(f"#### PHT Analysis for `{feature}`")

        fig = go.Figure()
        means = []
        window_names = list(datasets.keys())

        # Add violin plots for each window
        for i, (name, df) in enumerate(datasets.items()):
            if feature in df.columns:
                fig.add_trace(
                    go.Violin(
                        y=df[feature],
                        x0=i,  # Use numeric index for positioning
                        name=name,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
                means.append(df[feature].mean())
            else:
                means.append(None)

        # Add line connecting the means
        fig.add_trace(
            go.Scatter(
                x=list(range(len(window_names))),
                y=means,
                mode="lines+markers",
                name="Mean",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title=f"Distribution and Mean of '{feature}' over time",
            xaxis_title="Time Window (Dataset)",
            yaxis_title=f"Value of {feature}",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(window_names))),
                ticktext=[
                    name[:15] + "..." if len(name) > 15 else name
                    for name in window_names
                ],  # Shorten long names
            ),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# Drift Section (lightweight)
# =========================
def show_drift_section(pipeline_name, config):
    """
    Shows a simplified summary of drift detection results, comparing the
    Champion (previous) vs. Challenger (current) models. It focuses on
    high-level drift flags and performance comparison rather than per-feature plots.
    """
    st.subheader("Drift Analysis: Champion vs. Challenger")
    st.markdown(
        "This section analyzes for data and model drift, comparing the established **Champion** modelTrains and evaluates a new **Challenger** model on recent data."
    )

    base_dir = project_root / "pipelines" / pipeline_name
    metrics_dir = base_dir / "metrics"
    drift_path = metrics_dir / "drift_results.json"

    if not drift_path.exists():
        st.info("No drift results saved yet.")
        return
    results = _read_json_cached(str(drift_path), _mtime(str(drift_path)))
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
            st.warning(
                "Champion model promoted due to better performance"
                + (f" (reason: `{reason}`)" if reason else "")
            )
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
        st.markdown(
            "Indicates if drift was detected in any variable for each statistical test."
        )
        promoted_key = "promoted_model" if "promoted_model" in results else None
        if promoted_key is not None:
            val = results.get(promoted_key)
            if val is True:
                st.success("The previous model was promoted to the current one.")
                reason = results.get("promotion_reason")
                if reason:
                    st.info(f"**Promotion reason:** `{reason}`")
            elif val is False:
                st.info("The previous model was NOT promoted.")
            elif val == "error":
                st.warning("Error comparing with the previous model.")

        summary_data = []
        for k, v in drift_flags.items():
            # We only show the general test result, not per variable
            if "::" not in k:  # Avoid showing per-variable drift flags
                summary_data.append(
                    {
                        "Test": k,
                        "Result": "Drift detected" if bool(v) else "No drift detected",
                    }
                )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            total = len(summary_df)
            detected = (summary_df["Result"].str.contains("Drift")).sum()
            st.success(f"Tests without drift: {total - detected} of {total}")
            st.error(f"Tests with drift: {detected} of {total}")

            def color_drift_results(val):
                if val == "Drift detected":
                    return "background-color: #ffe6e6; color: red"  # Light red background, red text
                elif val == "No drift detected":
                    return "background-color: #e6ffe6; color: green"  # Light green background, green text
                return ""

            st.dataframe(
                summary_df.style.map(color_drift_results, subset=["Result"]),
                use_container_width=True,
            )
        else:
            st.info("No drift results found.")

    # Performance checks (compact)
    st.markdown("## Performance Checks")

    def _render_perf_table(perf_dict: dict, title: str):
        if not isinstance(perf_dict, dict) or len(perf_dict) == 0:
            return
        rows = []
        for metric_name, payload in perf_dict.items():
            # Extract value more robustly, prioritizing specific metric name
            value = payload.get(metric_name)  # Try exact metric name first
            if value is None:  # Fallback to common names if exact not found
                value = (
                    payload.get(metric_name.lower())
                    or payload.get("accuracy")
                    or payload.get("balanced_accuracy")
                    or payload.get("F1")
                    or payload.get("RMSE")
                    or payload.get("R2")
                    or payload.get("MAE")
                    or payload.get("MSE")
                    or payload.get("value")  # Generic 'value' key
                )
            rows.append(
                {
                    "Metric": metric_name,
                    "Value": f"{value:.4f}" if isinstance(value, float) else value,
                    "Threshold": payload.get("threshold"),
                    "Drift": bool(payload.get("drift", False)),
                }
            )
        st.markdown(f"### {title}")
        st.table(pd.DataFrame(rows))

    if "Performance_Current" in tests:
        _render_perf_table(
            tests["Performance_Current"], "Challenger Model vs Thresholds"
        )
    if "Performance_Previous" in tests:
        _render_perf_table(
            tests["Performance_Previous"], "Champion Model vs Thresholds"
        )
    if "Performance_Comparison" in tests and isinstance(
        tests["Performance_Comparison"], dict
    ):
        comp_rows = []
        for mname, payload in tests["Performance_Comparison"].items():
            metric = payload.get("metric", mname)
            prev_v = payload.get("prev")
            curr_v = payload.get("current")
            thr = payload.get("threshold")
            change = payload.get("relative_drop", payload.get("relative_increase"))
            drift_key = f"comparison::{metric.split('_')[0]}"
            comp_rows.append(
                {
                    "Metric": metric,
                    "Champion": f"{prev_v:.4f}"
                    if isinstance(prev_v, float)
                    else prev_v,
                    "Challenger": f"{curr_v:.4f}"
                    if isinstance(curr_v, float)
                    else curr_v,
                    "Relative Performance Change": f"{float(change) * 100:.2f}%"
                    if change is not None
                    else "N/A",
                    "Threshold": thr,
                    "Drift": bool(drift_flags.get(drift_key, False)),
                }
            )
        if comp_rows:
            st.markdown("### Champion vs Challenger (Relative Performance)")
            st.table(pd.DataFrame(comp_rows))

            # Bar chart comparison
            df_comp = pd.DataFrame(comp_rows)
            if not df_comp.empty:
                # Ensure Champion and Challenger are numeric for plotting
                df_comp["Champion"] = pd.to_numeric(
                    df_comp["Champion"], errors="coerce"
                )
                df_comp["Challenger"] = pd.to_numeric(
                    df_comp["Challenger"], errors="coerce"
                )

                df_melted = df_comp.melt(
                    id_vars=["Metric"],
                    value_vars=["Champion", "Challenger"],
                    var_name="Model",
                    value_name="Score",
                )
                fig = px.bar(
                    df_melted,
                    x="Metric",
                    y="Score",
                    color="Model",
                    barmode="group",
                    title="Champion vs Challenger Performance",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Detailed Feature Drift Comparison
    # ==================================
    st.markdown("---")
    st.subheader("Detailed Feature Drift Comparison")

    # Load datasets for comparison
    try:
        files_compared = results.get("files_compared")
        if (
            files_compared
            and "reference_file" in files_compared
            and "current_file" in files_compared
        ):
            st.info(
                "Loading datasets from `drift_results.json` for precise comparison."
            )
            ref_path = files_compared["reference_file"]
            cur_path = files_compared["current_file"]

            if not os.path.exists(ref_path):
                st.warning(f"Reference file not found: `{ref_path}`")
                return
            if not os.path.exists(cur_path):
                st.warning(f"Current file not found: `{cur_path}`")
                return

            df_previous = pd.read_csv(ref_path)
            df_current = pd.read_csv(cur_path)
            st.write(
                f"Comparing `{os.path.basename(ref_path)}` (Reference) vs. `{os.path.basename(cur_path)}` (Current)"
            )

        else:
            st.info(
                "`files_compared` not in drift results. Falling back to default data loading."
            )
            # Load previous data
            previous_data_path = base_dir / "control" / "previous_data.csv"
            if not previous_data_path.exists():
                st.info(
                    "`previous_data.csv` not found, cannot perform detailed drift comparison."
                )
                return
            df_previous = pd.read_csv(previous_data_path)

            # Load current data (latest processed file)
            data_dir = config.get("data_dir")
            control_dir = base_dir / "control"
            df_current, last_file = dashboard_data_loader(data_dir, str(control_dir))

            if df_current.empty:
                st.info("No current data found to compare against.")
                return
            st.write(
                f"Comparing `previous_data.csv` (Reference) vs. `{last_file}` (Current)"
            )

    except Exception as e:
        st.error(f"Error loading data for drift comparison: {e}")
        return

    # Align columns - crucial for comparison
    common_cols = df_previous.columns.intersection(df_current.columns)
    df_previous = df_previous[common_cols]
    df_current = df_current[common_cols]

    # Numerical feature comparison
    st.markdown("### Numerical Feature Statistics")
    num_cols = df_current.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        selected_num_col = st.selectbox("Select Numerical Feature", options=num_cols)
        if selected_num_col:
            prev_stats = df_previous[selected_num_col].describe()
            curr_stats = df_current[selected_num_col].describe()

            stats_df = pd.DataFrame(
                {
                    "Metric": prev_stats.index,
                    "Previous": prev_stats.values,
                    "Current": curr_stats.values,
                }
            )
            stats_df["Difference"] = stats_df["Current"] - stats_df["Previous"]

            st.dataframe(stats_df)
    else:
        st.info("No numerical columns to compare.")

    # Categorical feature comparison
    st.markdown("### Categorical Feature Statistics")
    cat_cols = df_current.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        selected_cat_col = st.selectbox("Select Categorical Feature", options=cat_cols)
        if selected_cat_col:
            st.markdown(f"#### Comparison for `{selected_cat_col}`")

            # Basic stats
            prev_nunique = df_previous[selected_cat_col].nunique()
            curr_nunique = df_current[selected_cat_col].nunique()
            prev_mode = (
                df_previous[selected_cat_col].mode().iloc[0]
                if not df_previous[selected_cat_col].mode().empty
                else "N/A"
            )
            curr_mode = (
                df_current[selected_cat_col].mode().iloc[0]
                if not df_current[selected_cat_col].mode().empty
                else "N/A"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Categories (Prev)", prev_nunique)
                st.metric(
                    "Unique Categories (Curr)",
                    curr_nunique,
                    delta=int(curr_nunique - prev_nunique),
                )
            with col2:
                st.metric("Mode (Prev)", str(prev_mode))
                st.metric("Mode (Curr)", str(curr_mode))

            # Proportions
            with st.expander(f"Proportions for `{selected_cat_col}`"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Previous Proportions**")
                    prev_props = df_previous[selected_cat_col].value_counts(
                        normalize=True
                    )
                    if prev_nunique <= 10:
                        fig_prev = px.pie(
                            prev_props,
                            values=prev_props.values,
                            names=prev_props.index,
                            title="Previous",
                        )
                        st.plotly_chart(fig_prev, use_container_width=True)
                    else:
                        st.dataframe(prev_props)
                with c2:
                    st.write("**Current Proportions**")
                    curr_props = df_current[selected_cat_col].value_counts(
                        normalize=True
                    )
                    if curr_nunique <= 10:
                        fig_curr = px.pie(
                            curr_props,
                            values=curr_props.values,
                            names=curr_props.index,
                            title="Current",
                        )
                        st.plotly_chart(fig_curr, use_container_width=True)
                    else:
                        st.dataframe(curr_props)
    else:
        st.info("No categorical columns to compare.")

    # PCA Drift Visualization
    show_pca_drift_plot(df_previous, df_current)

    # PHT Plot
    show_pht_plot(config, pipeline_name)

    # Heatmap (from drift results)
    st.markdown("## Drift Heatmap")
    # base_dir and metrics_dir are already defined at the beginning of show_drift_section
    drift_path = metrics_dir / "drift_results.json"

    if drift_path.exists():
        results = _read_json_cached(str(drift_path), _mtime(str(drift_path)))
        tests = results.get("tests", {})

        # Dynamically identify available metric types from the keys
        available_metric_types = set()
        for k in tests.keys():
            if k.endswith(("_ks", "_mw", "_psi", "_chi2")):
                if k.endswith("_ks"):
                    available_metric_types.add("KS p-value")
                elif k.endswith("_mw"):
                    available_metric_types.add("MW p-value")
                elif k.endswith("_psi"):
                    available_metric_types.add("PSI")
                elif k.endswith("_chi2"):
                    available_metric_types.add("Chi2 p-value")

        metric_options = sorted(list(available_metric_types))

        if not metric_options:
            st.info("No per-feature drift metrics found to build a heatmap.")
        else:
            msel = st.selectbox(
                "Heatmap metric",
                options=metric_options,
                index=0,
                key="drift_heatmap_metric",
            )
            topk = st.slider(
                "Top K features to display",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key="drift_heatmap_topk",
            )

            feat_vals: Dict[str, Optional[float]] = {}
            ordered: List[Tuple[str, float]] = []

            for k, payload in tests.items():
                feature_name = k.rsplit("_", 1)[0]  # Extract feature name
                if msel == "KS p-value" and k.endswith("_ks"):
                    try:
                        v = payload.get("p_value")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None
                elif msel == "MW p-value" and k.endswith("_mw"):
                    try:
                        v = payload.get("p_value")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None
                elif msel == "PSI" and k.endswith("_psi"):
                    try:
                        v = payload.get("psi")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None
                elif msel == "Chi2 p-value" and k.endswith("_chi2"):
                    try:
                        v = payload.get("p_value")
                        feat_vals[feature_name] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        feat_vals[feature_name] = None

            # Sort based on the selected metric type
            if msel in ("KS p-value", "MW p-value", "Chi2 p-value"):
                # For p-values, lower is more significant, so sort ascending
                ordered = sorted(
                    [(k, v) for k, v in feat_vals.items() if v is not None],
                    key=lambda x: x[1],
                )
            elif msel == "PSI":
                # For PSI, higher indicates more drift, so sort descending by absolute value
                ordered = sorted(
                    [(k, v) for k, v in feat_vals.items() if v is not None],
                    key=lambda x: -abs(x[1]),
                )

            if not ordered:
                st.info("No values available for the selected metric.")
            else:
                ordered = ordered[:topk]
                heat_df = pd.DataFrame(
                    {
                        "feature": [k for k, _ in ordered],
                        "value": [v for _, v in ordered],
                    }
                )
                if heat_df.empty:
                    st.info("No numeric values available for heatmap.")
                else:
                    # Define a custom colorscale for warmer colors and white for 0.0

                    colorscale = [
                        [0.0, "rgb(255, 255, 255)"],  # 1. White at 0.0
                        [
                            0.001,
                            "rgb(255, 253, 208)",
                        ],  # 2. Lightest yellow just above 0
                        [0.2, "rgb(255, 187, 120)"],  # 3. Light Orange
                        [0.4, "rgb(255, 140, 0)"],  # 4. Orange
                        [0.6, "rgb(230, 84, 0)"],  # 5. Dark Orange/Red
                        [0.8, "rgb(204, 41, 0)"],  # 6. Red
                        [1.0, "rgb(139, 0, 0)"],  # 7. Dark Red
                    ]

                    heat_fig = go.Figure(
                        data=go.Heatmap(
                            z=heat_df["value"].values.reshape(
                                -1, 1
                            ),  # Reshape for matrix-like view
                            x=[msel],  # Single column for the metric
                            y=heat_df["feature"].tolist(),  # Features as rows
                            colorscale=colorscale,  # Apply custom colorscale
                            xgap=1,
                            ygap=1,  # Add gaps for better visual separation
                        )
                    )
                    heat_fig.update_layout(
                        title=f"Drift heatmap by feature – {msel} (Top {topk})",
                        xaxis=dict(title=msel, tickangle=0),  # Metric name on x-axis
                        yaxis=dict(title="Feature"),  # Features on y-axis
                        height=min(800, 50 * len(ordered)),  # Adjust height dynamically
                    )
                    st.plotly_chart(heat_fig, use_container_width=True)

    else:
        st.info(
            "Drift results file (`drift_results.json`) not found. Heatmap cannot be displayed."
        )


# =========================
# Train / Retrain Section
# =========================
def show_train_section(pipeline_name):
    """
    Displays training or retraining metrics, providing detailed explanations for the retraining methods used.
    Expects JSON at: pipelines/{pipeline_name}/metrics/train_results.json
    """
    st.subheader("Training / Retraining Results")
    train_path = (
        project_root / "pipelines" / pipeline_name / "metrics" / "train_results.json"
    )

    if not train_path.exists():
        st.info("No training results found yet.")
        return

    results = _read_json_cached(str(train_path), _mtime(str(train_path)))
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
                st.markdown(
                    "**Fallback:** "
                    + ("Enabled (used)" if bool(fallback) else "Not used")
                )

            gs = results.get("gridsearch", None)
            if isinstance(gs, dict) and gs:
                st.markdown("**GridSearchCV:**")
                if "best_params" in gs:
                    bp = gs["best_params"]
                    if isinstance(bp, dict) and bp:
                        st.table(
                            pd.DataFrame([bp])
                            .transpose()
                            .rename(columns={0: "best_params"})
                        )
                    else:
                        st.write(bp)
                if "cv" in gs and gs["cv"] is not None:
                    st.write(f"**CV folds:** {gs['cv']}")
            elif isinstance(gs, str):
                st.write(f"**GridSearchCV:** {gs}")

            extra_rows = []
            for k in ["window_size_used", "window_size"]:
                if k in results:
                    extra_rows.append(("Window size", results[k]))
                    break
            if "calibration" in results:
                extra_rows.append(("Calibration", results["calibration"]))
            if "replay_frac_old" in results:
                extra_rows.append(("Replay frac (old)", results["replay_frac_old"]))
            if "meta" in results:
                extra_rows.append(("Meta", results["meta"]))
            if "components" in results:
                extra_rows.append(("Ensemble components", results["components"]))
            for k, nice in [
                ("train_size", "Train size"),
                ("eval_size", "Eval size"),
                ("epochs", "Epochs"),
                ("learning_rate", "Learning rate"),
                ("early_stopping", "Early stopping"),
            ]:
                if k in results:
                    extra_rows.append((nice, results[k]))
            if extra_rows:
                st.markdown("**Extra retrain details:**")
                st.table(pd.DataFrame(extra_rows, columns=["Key", "Value"]))

    # Metrics
    st.markdown("## Metrics")
    if "classification_report" in results:
        st.markdown("### Classification Metrics")
        bal_acc = results.get(
            "balanced_accuracy", results.get("balanced_accuracy_score")
        )
        if bal_acc is not None:
            try:
                st.write(f"**Balanced Accuracy:** {float(bal_acc):.4f}")
            except Exception:
                st.write(f"**Balanced Accuracy:** {bal_acc}")
        clf_report = results["classification_report"]
        clf_report_df = pd.DataFrame(clf_report)
        st.dataframe(
            clf_report_df.transpose()
            if set(["precision", "recall", "f1-score", "support"]).issubset(
                clf_report_df.index
            )
            else clf_report_df,
            use_container_width=True,
        )
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


def show_logs_section(pipeline_name):
    import os

    st.subheader("Pipeline Logs")
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
    general_log_text = clean_log_content(load_log(str(logs_path)))
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
    error_log_text = clean_log_content(load_log(str(error_logs_path)))
    if error_log_text:
        st.code(error_log_text, language="bash")
    else:
        st.success("No errors found.")

    # Warning Log
    st.markdown("### Warning Log")
    warning_log_text = clean_log_content(load_log(str(warning_logs_path)))
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

# Load runner_config.json to get all pipeline details
config_path = (
    project_root / "pipelines" / pipeline_name / "config" / "runner_config.json"
)
if config_path.exists():
    config = _read_json_cached(str(config_path), _mtime(str(config_path)))
    pipeline_name = config.get("pipeline_name", pipeline_name)
    data_dir = config.get("data_dir")
    monitor_type = config.get("monitor_type", "unknown")
    schedule_info = config.get("schedule")
else:
    st.error(
        f"runner_config.json not found for pipeline '{pipeline_name}'. Please ensure the monitor has been started correctly."
    )
    st.stop()

if not data_dir:
    st.error("`data_dir` missing in runner_config.json.")
    st.stop()

# Display schedule information if available
if "schedule" in monitor_type and schedule_info:
    with st.expander("Schedule Information", expanded=True):
        st.write(f"**Monitor Type:** `{monitor_type}`")
        st.write("**Schedule Details:**")
        st.json(schedule_info)

# Tabs
tabs = st.tabs(
    ["Dataset", "Evaluator", "Drift", "Historical Performance", "Train/Retrain", "Logs"]
)

# =========================
# Auto-refresh on control_file change
# =========================
control_file = (
    project_root / "pipelines" / pipeline_name / "control" / "control_file.txt"
)


# Initialize timestamp in session_state in case that it doesn't exists
if "control_file_mtime" not in st.session_state:
    st.session_state.control_file_mtime = (
        _mtime(control_file) if control_file.exists() else 0.0
    )


# Check for changes
current_mtime = _mtime(control_file) if control_file.exists() else 0.0
if current_mtime != st.session_state.control_file_mtime:
    st.session_state.control_file_mtime = current_mtime
    st.experimental_rerun()

with tabs[0]:
    show_dataset_section(data_dir, pipeline_name)

with tabs[1]:
    show_evaluator_section(pipeline_name)

with tabs[2]:
    show_drift_section(pipeline_name, config)

with tabs[3]:
    show_historical_performance_section(pipeline_name)

with tabs[4]:
    show_train_section(pipeline_name)

with tabs[5]:
    show_logs_section(pipeline_name)
