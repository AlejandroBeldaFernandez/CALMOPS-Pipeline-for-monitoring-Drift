# monitor_dashboard.py
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from calmops.utils import get_pipelines_root
from utils import _load_any_dataset, dashboard_data_loader, show_evolution_section
from dashboard_common import (
    _mtime,
    _read_json_cached,
    _read_text_cached,
    _read_csv_cached,
    _load_any_dataset_cached,
    _sanitize_text,
    _sanitize_df,
    _sanitize_figure,
    _safe_table,
    _safe_table_static,
    _safe_markdown,
    _safe_write,
    _safe_caption,
    _safe_plot,
    _safe_json_display,
    _sorted_blocks,
    _get_pipeline_base_dir,
    _detect_block_col,
)


import os

# Suppress TensorFlow logs before importing it
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(# tf.compat.v1.logging.ERROR)
# tf.get_logger().setLevel("ERROR")


# Anti-delta sanitizers and safe rendering wrappers have been moved to dashboard_common.py


project_root = get_pipelines_root()


# Performance helpers moved to dashboard_common.py


# =========================
# Block helpers
# =========================
# _detect_block_col moved to dashboard_common.py


# _sorted_blocks moved to dashboard_common.py


# =========================
# Dataset Tab
# =========================
def show_dataset_section(
    df: pd.DataFrame, last_file: str, pipeline_name: str, block_col: Optional[str]
):
    """
    Displays the Dataset tab for block-based pipelines.
    Includes block distribution, preview, statistics, and missingness analysis.
    """
    st.subheader("Dataset Inspector")

    if df.empty or not last_file:
        st.warning("⚠ No processed dataset found yet.")
        return


# =========================
# Evaluator Tab
# =========================
def show_evaluator_section(
    pipeline_name: str, split_within_blocks: bool, train_percentage: float
):
    """
    Displays evaluation metrics for block-based pipelines.
    Handles 'Split Within Blocks' vs 'By Block' logic and visualizes per-block performance.
    """
    _ = st.subheader("Evaluation Results")

    base_dir = _get_pipeline_base_dir(pipeline_name)
    metrics_dir = base_dir / "metrics"

    # Display run configuration
    st.markdown("### Run Configuration")
    if split_within_blocks:
        st.info(
            f"Split Mode: **Split Within Blocks**. Each block was split into {train_percentage * 100:.0f}% for training and {100 - (train_percentage * 100):.0f}% for evaluation."
        )
    else:
        eval_path = metrics_dir / "eval_results.json"
        results = _read_json_cached(eval_path, _mtime(eval_path)) or {}
        eval_blocks = (results.get("blocks", {}) or {}).get("evaluated_blocks", [])
        st.info(
            f"Split Mode: **By Block**. Evaluation was performed on dedicated blocks: `{eval_blocks}`"
        )

    # Model Structure
    st.markdown("### Model Structure")
    train_path = metrics_dir / "train_results.json"
    train_results = _read_json_cached(train_path, _mtime(train_path)) or {}
    fallback_strategy = train_results.get("fallback_strategy", "global")

    if fallback_strategy == "ensemble":
        st.info(
            "Model type: **Ensemble**. Uses an ensemble of per-block models for predictions on unseen blocks."
        )
    else:
        st.info(
            "Model type: **Global**. Uses a single global model for predictions on unseen blocks."
        )

    candidates_dir = base_dir / "candidates"
    eval_path = metrics_dir / "eval_results.json"
    health_path = metrics_dir / "health.json"

    if not eval_path.exists():
        _safe_markdown("No evaluation results found yet.")
        return None

    results = _read_json_cached(eval_path, _mtime(eval_path)) or {}
    if not results:
        _safe_markdown("Empty evaluation results.")
        return None

    if results.get("approved", False):
        st.success(_sanitize_text("Model approved."))
    else:
        st.error(_sanitize_text("Model NOT approved."))

    # Download eval JSON
    try:
        st.download_button(
            "⬇ Download eval_results.json",
            data=json.dumps(results, indent=2),
            file_name="eval_results.json",
            mime="application/json",
        )
    except Exception:
        pass

    blocks_info = results.get("blocks", {}) or {}
    per_blk_full = blocks_info.get("per_block_metrics_full") or {}
    evaluated_blocks = [str(b) for b in (blocks_info.get("evaluated_blocks") or [])]

    if blocks_info:
        _safe_markdown("## 🧱 Blocks (from evaluator)")
        cols = st.columns(3)
        cols[0].metric(
            "Block column", _sanitize_text(str(blocks_info.get("block_col")))
        )
        cols[1].metric("Evaluated blocks", len(evaluated_blocks))
        cols[2].metric(
            "Reference (train) blocks", len(blocks_info.get("reference_blocks", []))
        )

        new_blocks = blocks_info.get("new_blocks_detected", []) or []
        missing_blocks = blocks_info.get("missing_reference_blocks", []) or []
        k1, k2 = st.columns(2)
        k1.metric("New blocks (not in train)", len(new_blocks))
        k2.metric("Missing reference blocks", len(missing_blocks))
        if new_blocks:
            _safe_markdown(
                "**New blocks detected:** " + ", ".join(map(_sanitize_text, new_blocks))
            )
        if missing_blocks:
            _safe_markdown(
                "**Reference blocks missing in eval:** "
                + ", ".join(map(_sanitize_text, missing_blocks))
            )

        if evaluated_blocks:
            sel_eval = st.selectbox(
                "Filter block (Evaluator)",
                options=["(All)"] + evaluated_blocks,
                index=0,
                key="eval_block_selector",
            )
        else:
            sel_eval = "(All)"

        if per_blk_full:
            _safe_markdown("### Per-block test metrics")
            df_blk = (
                pd.DataFrame.from_dict(per_blk_full, orient="index")
                .reset_index()
                .rename(columns={"index": "block"})
            )
            if sel_eval != "(All)":
                df_blk = df_blk[df_blk["block"].astype(str) == sel_eval]
            _safe_table(df_blk)

            candidate_metrics = [
                m
                for m in [
                    "accuracy",
                    "f1",
                    "balanced_accuracy",
                    "r2",
                    "rmse",
                    "mae",
                    "mse",
                ]
                if m in df_blk.columns
            ]
            if candidate_metrics:
                metric_to_plot = st.selectbox(
                    "Metric to plot",
                    options=candidate_metrics,
                    index=0,
                    key="eval_metric_sel",
                )

                df_line = _sanitize_df(df_blk.copy())
                try:
                    order = _sorted_blocks(df_line["block"])
                    df_line["block"] = pd.Categorical(
                        df_line["block"].astype(str),
                        categories=[str(o) for o in order],
                        ordered=True,
                    )
                    df_line = df_line.sort_values("block")
                except Exception:
                    df_line["block"] = df_line["block"].astype(str)

                df_line["block_str"] = df_line["block"].astype(str)

                title = f"{metric_to_plot} by Block"
                if split_within_blocks:
                    title += " (on Eval Split)"

                fig = px.line(
                    df_line,
                    x="block_str",
                    y=metric_to_plot,
                    markers=True,
                    title=_sanitize_text(title),
                    labels={
                        "block_str": _sanitize_text("Block"),
                        metric_to_plot: _sanitize_text(metric_to_plot),
                    },
                )

                fig.update_xaxes(type="category")

                thr = results.get("thresholds", {}).get(metric_to_plot)
                if thr is not None:
                    fig.add_hline(
                        y=float(thr),
                        line_dash="dash",
                        annotation_text=_sanitize_text(f"threshold = {thr}"),
                        annotation_position="top left",
                    )
                _safe_plot(fig)

            cm_dict = blocks_info.get("per_block_confusion_matrix")
            if cm_dict:
                _safe_markdown("### 🧩 Confusion matrix (per block)")
                blocks_with_cm = [b for b, cm in cm_dict.items() if cm]
                if blocks_with_cm:
                    bsel_cm = st.selectbox(
                        "Block (confusion matrix)",
                        options=blocks_with_cm,
                        index=0,
                        key="cm_block",
                    )
                    try:
                        cm = np.array(cm_dict[bsel_cm], dtype=float)
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            aspect="auto",
                            labels=dict(x="Predicted", y="True", color="Count"),
                            title=_sanitize_text(f"Confusion matrix – {bsel_cm}"),
                        )
                        _safe_plot(fig_cm)
                    except Exception as e:
                        _safe_markdown(
                            _sanitize_text(f"Could not render confusion matrix: {e}")
                        )

    _safe_markdown("## Used Thresholds")
    thresholds = results.get("thresholds", {})
    if thresholds:
        # Convert all threshold values to string to avoid mixed types
        threshold_data = [
            {"Metric": k, "Threshold": str(v)} for k, v in thresholds.items()
        ]
        _safe_table_static(pd.DataFrame(threshold_data))
    else:
        _safe_markdown("No thresholds found in the evaluation results.")

    if "predictions" in results and results["predictions"]:
        _safe_markdown("## Prediction Examples")
        preds_df = pd.DataFrame(results["predictions"])
        if "eval_block_selector" in st.session_state:
            bsel = st.session_state["eval_block_selector"]
            if bsel != "(All)" and "block" in preds_df.columns:
                preds_df = preds_df[preds_df["block"].astype(str) == bsel]
        _safe_table_static(preds_df)

    _safe_markdown("## Test Metrics (global)")
    metrics = results.get("metrics", {})
    if "classification_report" in metrics:
        _safe_write(f"*Accuracy:* {round(metrics.get('accuracy', 0), 4)}")
        _safe_write(
            f"*Balanced Accuracy:* {round(metrics.get('balanced_accuracy', 0), 4)}"
        )
        _safe_write(f"*F1 (macro):* {round(metrics.get('f1', 0), 4)}")
        report_df = pd.DataFrame(metrics["classification_report"])
        if set(["precision", "recall", "f1-score", "support"]).issubset(
            report_df.index
        ):
            _safe_table_static(report_df.transpose())
        else:
            _safe_table_static(report_df)
    elif "r2" in metrics or "rmse" in metrics:
        if "r2" in metrics:
            _safe_write(f"*R²:* {round(metrics.get('r2', 0), 4)}")
        if "rmse" in metrics:
            _safe_write(f"*RMSE:* {round(metrics.get('rmse', 0), 4)}")
        if "mae" in metrics:
            _safe_write(f"*MAE:* {round(metrics.get('mae', 0), 4)}")
        if "mse" in metrics:
            _safe_write(f"*MSE:* {round(metrics.get('mse', 0), 4)}")
    else:
        _safe_markdown("No evaluation metrics found.")

    _safe_markdown("## 🛑 Circuit Breaker Status")
    if os.path.exists(health_path):
        try:
            health = _read_json_cached(health_path, _mtime(health_path)) or {}
            consecutive = int(health.get("consecutive_failures", 0) or 0)
            paused_until = health.get("paused_until")
            last_failure_ts = health.get("last_failure_ts")
            cols = st.columns(3)
            cols[0].metric("Consecutive Failures", consecutive)

            def _fmt_ts(ts):
                try:
                    import datetime as _dt

                    return _dt.datetime.fromtimestamp(float(ts)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except Exception:
                    return str(ts)

            cols[1].metric(
                "Paused Until",
                _sanitize_text(_fmt_ts(paused_until) if paused_until else "—"),
            )
            cols[2].metric(
                "Last Failure",
                _sanitize_text(_fmt_ts(last_failure_ts) if last_failure_ts else "—"),
            )

            import time as _t

            paused = bool(paused_until and _t.time() < float(paused_until))
            if paused:
                st.warning(
                    _sanitize_text(
                        "⏸ Retraining is currently paused by the circuit breaker."
                    )
                )
            else:
                st.success(_sanitize_text("▶ Retraining is active (not paused)."))
        except Exception as e:
            st.warning(_sanitize_text(f"Could not read health.json: {e}"))
    else:
        _safe_markdown("No circuit breaker state found yet (health.json).")

    _safe_markdown("## 🕵️ Candidates (Non-Approved Models)")
    if not os.path.exists(candidates_dir):
        _safe_markdown("No candidates directory yet.")
        return None

    candidates = []
    try:
        for entry in sorted(
            [
                candidates_dir / d
                for d in os.listdir(candidates_dir)
                if (candidates_dir / d).is_dir()
            ],
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
                    meta = _read_json_cached(meta_path, _mtime(meta_path))
                    row["approved"] = bool(meta.get("approved", False))
                    row["file"] = meta.get("file")
                    row["timestamp"] = meta.get("timestamp")
                if os.path.exists(eval_p):
                    ev = _read_json_cached(eval_p, _mtime(eval_p))
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
            _safe_table(df_cand[show_cols])
            _safe_caption(
                "Showing up to 10 latest candidates. Each folder contains `model.pkl` and `eval_results.json`."
            )
        else:
            _safe_markdown("No candidates have been saved yet.")
    except Exception as e:
        st.warning(f"Could not enumerate candidates: {e}")

    return None


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
    base_dir = _get_pipeline_base_dir(pipeline_name)
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


def show_drift_section(pipeline_name, config):
    st.subheader("🌊 Drift Results (block_wise, pairwise)")

    base_dir = _get_pipeline_base_dir(pipeline_name)
    metrics_dir = base_dir / "metrics"
    drift_path = metrics_dir / "drift_results.json"

    if not os.path.exists(drift_path):
        _safe_markdown("No drift results saved yet.")
        return

    results = _read_json_cached(drift_path, _mtime(drift_path)) or {}
    if not results:
        _safe_markdown("Drift results are empty.")
        return

    # --- Decision banner
    decision = results.get("decision")
    if decision == "no_drift":
        st.success(_sanitize_text("🟢 No drift detected – current model maintained"))
    elif decision == "retrain":
        st.error(_sanitize_text("Retraining triggered due to drift"))
    elif decision == "train":
        st.info(_sanitize_text("🆕 First run: training from scratch"))
    elif decision == "end_error":
        st.warning(_sanitize_text("⚠ Ended with error while checking drift"))

    if results.get("promoted_model"):
        st.info(
            _sanitize_text(
                f"⤴ Previous model was promoted (reason: {results.get('promotion_reason', '-')})."
            )
        )

    # --- Basic payload
    blk = results.get("blockwise", {}) or {}
    blocks_all = [str(b) for b in (blk.get("blocks", []) or [])]
    block_col = blk.get("block_col")
    pairwise = blk.get("pairwise", {}) or {}
    by_test = results.get("drift", {}).get("by_test", {}) or {}
    by_block_stat_drift = blk.get("by_block_stat_drift", {}) or {}

    # --- Header KPIs
    total_pairs = len(blocks_all) * (len(blocks_all) - 1) // 2
    pairs_with_drift = 0
    if pairwise:
        # Sum of drift flags over all tests
        all_pairs = set()
        for test_name, test_results in pairwise.items():
            for pair, result in test_results.items():
                if result.get("drift"):
                    # Normalize pair to avoid double counting (e.g., "1|2" and "2|1")
                    sorted_pair = "|".join(sorted(pair.split("|")))
                    all_pairs.add(sorted_pair)
        pairs_with_drift = len(all_pairs)

    blocks_with_drift = sum(1 for v in by_block_stat_drift.values() if v)

    cols = st.columns(5)
    cols[0].metric("Block column", _sanitize_text(str(block_col)))
    cols[1].metric("Num blocks", len(blocks_all))
    cols[2].metric("Tests with drift", sum(1 for v in by_test.values() if v))
    cols[3].metric("Pairs with drift", f"{pairs_with_drift} / {total_pairs}")
    cols[4].metric("Blocks with drift", f"{blocks_with_drift} / {len(blocks_all)}")

    # --- Block selector (like Train)
    if blocks_all:
        block_filter = st.selectbox(
            "Filter block (Drift)",
            options=["(All)"] + blocks_all,
            index=0,
            key="drift_block_selector",
        )
    else:
        block_filter = "(All)"

    # Scope blocks (ordering tries numeric/datetime first)
    def _sorted_blocks_natural(vals: List[str]) -> List[str]:
        try:
            nums = [float(v) for v in vals]
            return [x for _, x in sorted(zip(nums, vals))]
        except Exception:
            try:
                dt = pd.to_datetime(vals, errors="raise")
                return [x for _, x in sorted(zip(dt, vals))]
            except Exception:
                return sorted(vals, key=lambda x: str(x))

    scope_blocks = (
        blocks_all
        if block_filter == "(All)"
        else [block_filter] + [b for b in blocks_all if b != block_filter]
    )
    scope_blocks = _sorted_blocks_natural(scope_blocks)

    # Summary box
    _safe_markdown("## Summary")
    if by_test:
        df_sum = pd.DataFrame(
            [
                {"Test": k, "Drift": "YES" if bool(v) else "NO"}
                for k, v in by_test.items()
            ]
        )
        _safe_table_static(df_sum)
    else:
        _safe_markdown("No statistical tests found.")

    # ===== Helpers for matrices =====
    def _value_key_and_label(test_name: str) -> tuple[str | None, str, str]:
        if test_name in ("KS", "MWU", "CVM"):
            return "p_min", "p-value (min) — lower is worse", "Viridis_r"
        if test_name == "PSI":
            return "psi_max", "PSI (max) — higher is worse", "Inferno"
        if test_name in ("Hellinger", "EMD"):
            return "distance_max", "Distance (max) — higher is worse", "Inferno"
        return None, "value", "Plasma"

    def _make_matrix_for_test(
        test_name: str, scope: List[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (values matrix M, flags matrix F) for a given test and block scope."""
        value_key, _, _ = _value_key_and_label(test_name)
        matrix = pairwise.get(test_name, {}) or {}
        blocks = [b for b in scope if b in blocks_all]
        M = pd.DataFrame(index=blocks, columns=blocks, dtype=float)
        F = pd.DataFrame(index=blocks, columns=blocks, dtype=bool)
        for i in range(len(blocks)):
            M.iloc[i, i] = np.nan
            F.iloc[i, i] = False
        for k, payload in matrix.items():
            try:
                bi, bj = k.split("|", 1)
            except Exception:
                continue
            if bi not in blocks or bj not in blocks:
                continue
            # value
            val = None
            if value_key is None:
                if "p_value" in payload and payload["p_value"] is not None:
                    val = float(payload["p_value"])
                elif "distance" in payload and payload["distance"] is not None:
                    val = float(payload["distance"])
            else:
                v = payload.get(value_key)
                if v is not None:
                    val = float(v)
            if val is not None:
                M.loc[bi, bj] = val
                M.loc[bj, bi] = val
            # flag
            fl = bool(payload.get("drift", False))
            F.loc[bi, bj] = fl
            F.loc[bj, bi] = fl

        # If a specific block was chosen, blank out pairs that don't involve it (for clarity)
        if block_filter != "(All)":
            for r in list(M.index):
                for c in list(M.columns):
                    if r != block_filter and c != block_filter:
                        M.loc[r, c] = np.nan
                        F.loc[r, c] = False
        return M, F

    # ===== Per-test bar: #pairs with drift (in scope) =====
    if pairwise:
        rows = []
        for t in pairwise.keys():
            _, F_t = _make_matrix_for_test(t, scope_blocks)
            if F_t is not None and not F_t.empty:
                # count upper triangle True
                n = len(F_t.index)
                num = int(np.triu(F_t.values, k=1).sum()) if n >= 2 else 0
                tot = n * (n - 1) // 2
                rows.append({"test": t, "pairs_with_drift": num, "pairs_total": tot})
        if rows:
            _safe_markdown("### 🧮 Drift pairs per test (in scope)")
            df_counts = pd.DataFrame(rows)
            df_counts["ratio"] = df_counts.apply(
                lambda r: 0.0
                if r["pairs_total"] == 0
                else r["pairs_with_drift"] / r["pairs_total"],
                axis=1,
            )
            _safe_table_static(df_counts)

            fig_cnt = px.bar(
                df_counts,
                x="test",
                y="pairs_with_drift",
                title=_sanitize_text("Pairs with drift by test"),
                labels={"test": "Test", "pairs_with_drift": "Pairs with drift"},
            )
            _safe_plot(fig_cnt)

    # ===== Pairwise heatmaps and details for a chosen test =====
    if pairwise and scope_blocks:
        _safe_markdown("## 🗺 Pairwise matrices")
        tests_available = list(pairwise.keys())
        tsel = st.selectbox(
            "Select test", options=tests_available, index=0, key="drift_test_select"
        )
        show_only_drift = st.checkbox(
            "Show only pairs with drift", value=False, key="drift_only_pairs"
        )

        value_key, label, colorscale = _value_key_and_label(tsel)
        M, F = _make_matrix_for_test(tsel, scope_blocks)

        # Optional: show only drift pairs
        if show_only_drift and not F.empty and not M.empty:
            mask = F.values.copy()
            for i in range(mask.shape[0]):
                mask[i, i] = False
            M = M.where(pd.DataFrame(mask, index=M.index, columns=M.columns))

        if M is not None and not M.empty:
            # KPIs for this view
            n_sc = len([b for b in scope_blocks if b in M.index])
            tot_pairs = n_sc * (n_sc - 1) // 2
            flagged_pairs = int(np.triu(F.values, k=1).sum()) if n_sc >= 2 else 0
            k1, k2 = st.columns(2)
            k1.metric("Pairs in scope", tot_pairs)
            k2.metric("Pairs with drift (selected test)", flagged_pairs)

            heat = go.Heatmap(
                z=M.values,
                x=[_sanitize_text(str(c)) for c in M.columns],
                y=[_sanitize_text(str(r)) for r in M.index],
                coloraxis="coloraxis",
                hoverongaps=False,
            )
            heat_fig = go.Figure(data=[heat])
            heat_fig.update_layout(
                title=_sanitize_text(f"{tsel} pairwise matrix – {label}"),
                xaxis=dict(title="Block"),
                yaxis=dict(title="Block"),
                coloraxis=dict(
                    colorscale=colorscale, colorbar=dict(title=_sanitize_text(label))
                ),
                height=520,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            # mark drift cells
            anns = []
            for i, r in enumerate(M.index):
                for j, c in enumerate(M.columns):
                    if i == j:
                        continue
                    try:
                        if bool(F.loc[r, c]):
                            anns.append(
                                dict(
                                    x=_sanitize_text(str(c)),
                                    y=_sanitize_text(str(r)),
                                    text="⚠",
                                    showarrow=False,
                                    font=dict(size=14),
                                    xanchor="center",
                                    yanchor="middle",
                                )
                            )
                    except Exception:
                        pass
            if anns:
                heat_fig.update_layout(annotations=anns)
            _safe_plot(heat_fig)

            # Drift partners per block (selected test)
            _safe_markdown("### 🔢 Drift partners per block (selected test)")
            drift_counts = {
                b: int(F.loc[b].sum(skipna=True)) if b in F.index else 0
                for b in M.index
            }
            df_dc = pd.DataFrame(
                {
                    "block": list(drift_counts.keys()),
                    "pairs_with_drift": list(drift_counts.values()),
                }
            )
            fig_dc = px.bar(
                df_dc,
                x=[str(b) for b in df_dc["block"]],
                y="pairs_with_drift",
                title=_sanitize_text(f"Blocks with drift pairs – {tsel}"),
                labels={
                    "x": _sanitize_text("Block"),
                    "pairs_with_drift": _sanitize_text("Pairs with drift"),
                },
            )
            _safe_plot(fig_dc)

            # Distribution of pairwise values
            vals = []
            for i, r in enumerate(M.index):
                for j, c in enumerate(M.columns):
                    if j <= i:  # upper triangle only
                        continue
                    v = M.loc[r, c]
                    if v is not None and not np.isnan(v):
                        vals.append(float(v))
            if vals:
                _safe_markdown("### Pairwise values distribution")
                df_vals = pd.DataFrame({label.split("—")[0].strip(): vals})
                fig_hist = px.histogram(
                    df_vals,
                    x=df_vals.columns[0],
                    nbins=30,
                    title=_sanitize_text(
                        f"Distribution – {tsel} ({label.split('—')[0].strip()})"
                    ),
                    histnorm="probability density",
                )
                _safe_plot(fig_hist)

            # Top worst pairs table
            top_rows = []

            def _worse_key(v: float) -> float:
                if tsel in ("KS", "MWU", "CVM"):
                    return v if v is not None else 1.0  # smaller p worse (asc)
                else:
                    return -(
                        v if v is not None else 0.0
                    )  # larger distance worse (desc)

            for i, r in enumerate(M.index):
                for j, c in enumerate(M.columns):
                    if j <= i:
                        continue
                    v = M.loc[r, c]
                    if v is None or np.isnan(v):
                        continue
                    top_rows.append((r, c, float(v), bool(F.loc[r, c])))

            if top_rows:
                top_sorted = sorted(top_rows, key=lambda x: _worse_key(x[2]))[:20]
                df_top = pd.DataFrame(
                    top_sorted, columns=["block_A", "block_B", "value", "drift"]
                )
                df_top["drift"] = df_top["drift"].map(lambda z: "YES" if z else "NO")
                _safe_markdown("### 🏁 Top pairs (worst first)")
                _safe_table_static(df_top)

            # Thresholds / alpha used (if present in payloads)
            _safe_markdown("### 🎚 Thresholds / α used (selected test)")
            # Try to infer thresholds from any payload in the matrix
            thr_alpha, thr_generic = None, None
            for payload in (pairwise.get(tsel, {}) or {}).values():
                if "alpha" in payload and payload["alpha"] is not None:
                    thr_alpha = payload["alpha"]
                    break
            for payload in (pairwise.get(tsel, {}) or {}).values():
                if "threshold" in payload and payload["threshold"] is not None:
                    thr_generic = payload["threshold"]
                    break
            rows_thr = []
            if thr_alpha is not None:
                rows_thr.append({"key": "alpha", "value": thr_alpha})
            if thr_generic is not None:
                rows_thr.append({"key": "threshold", "value": thr_generic})
            if rows_thr:
                _safe_table_static(pd.DataFrame(rows_thr))
            else:
                _safe_markdown("No thresholds found in payloads for this test.")
        else:
            _safe_markdown("No values available for this test.")

    # ---- Any-test flags per block
    by_block_stat = blk.get("by_block_stat_drift", {}) or {}
    if by_block_stat:
        _safe_markdown("## 🚩 Stat-drift flags by block (any test)")
        # Respect block filter
        rows = [
            {"block": b, "flag": int(bool(v))}
            for b, v in by_block_stat.items()
            if (block_filter == "(All)" or str(b) == block_filter)
        ]
        if rows:
            df_flags = pd.DataFrame(rows)
            fig_flags = px.bar(
                df_flags,
                x=[str(b) for b in df_flags["block"]],
                y="flag",
                title=_sanitize_text("Blocks flagged in any pairwise test"),
                labels={"x": _sanitize_text("Block"), "flag": _sanitize_text("Flag")},
            )
            _safe_plot(fig_flags)

    # ---- Feature Explorer
    _safe_markdown("## 🔎 Feature Explorer (Block vs Block)")
    cfg_path = _get_pipeline_base_dir(pipeline_name) / "config" / "config.json"
    data_dir = None
    if cfg_path.exists():
        cfg = _read_json_cached(cfg_path, _mtime(cfg_path)) or {}
        data_dir = cfg.get("data_dir")

    control_file = (
        _get_pipeline_base_dir(pipeline_name) / "control" / "control_file.txt"
    )
    if not (data_dir and control_file.exists()):
        _safe_markdown("Dataset not available for feature explorer.")
        return
    with open(control_file, "r") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    if not lines:
        _safe_markdown("Dataset not available for feature explorer.")
        return
    last_file = lines[-1].split(",")[0]
    curr_path = Path(data_dir) / last_file
    if not curr_path.exists():
        _safe_markdown("Dataset not available for feature explorer.")
        return
    df = _load_any_dataset_cached(curr_path, _mtime(curr_path))
    if df is None or df.empty:
        _safe_markdown("Dataset not available for feature explorer.")
        return
    bc = _detect_block_col(pipeline_name, df)
    if bc is None or bc not in df.columns:
        _safe_markdown("No block column in dataset.")
        return

    blocks = [str(b) for b in _sorted_blocks(df[bc])]
    if len(blocks) < 2:
        _safe_markdown("Need at least two blocks to compare.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        b1 = st.selectbox("Block A", options=blocks, index=0, key="fx_b1")
    with c2:
        b2 = st.selectbox(
            "Block B", options=[b for b in blocks if b != b1], index=0, key="fx_b2"
        )
    with c3:
        num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != bc]
        if not num_cols:
            _safe_markdown("No numeric columns.")
            return
        col = st.selectbox("Variable", options=num_cols, index=0, key="fx_col")

    bins = st.slider(
        "Bins", min_value=10, max_value=100, value=40, step=5, key="fx_bins"
    )
    norm = st.checkbox("Normalize to density", value=True, key="fx_norm")
    mode = st.radio(
        "Plot", ["Histogram", "ECDF", "Violin/Box"], horizontal=True, key="fx_plot_mode"
    )

    s1 = (
        pd.to_numeric(df.loc[df[bc].astype(str) == b1, col], errors="coerce")
        .dropna()
        .to_numpy()
    )
    s2 = (
        pd.to_numeric(df.loc[df[bc].astype(str) == b2, col], errors="coerce")
        .dropna()
        .to_numpy()
    )

    def _stats(arr):
        if arr.size == 0:
            return {
                "n": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
            }
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    _safe_markdown("### Summary stats")
    _safe_table_static(
        pd.DataFrame(
            [
                {"block": b1, **_stats(s1)},
                {"block": b2, **_stats(s2)},
            ]
        )
    )

    if mode == "Histogram":
        histnorm = "probability density" if norm else None
        fig = go.Figure()
        if s1.size:
            fig.add_trace(
                go.Histogram(
                    x=s1,
                    name=_sanitize_text(f"B{b1}"),
                    nbinsx=bins,
                    histnorm=histnorm,
                    opacity=0.55,
                )
            )
        if s2.size:
            fig.add_trace(
                go.Histogram(
                    x=s2,
                    name=_sanitize_text(f"B{b2}"),
                    nbinsx=bins,
                    histnorm=histnorm,
                    opacity=0.55,
                )
            )
        fig.update_layout(
            barmode="overlay",
            title=_sanitize_text(f"Histogram – {col}"),
            xaxis_title=_sanitize_text(col),
            yaxis_title=_sanitize_text("Density" if norm else "Count"),
        )
        _safe_plot(fig)
    elif mode == "ECDF":
        df_ecdf = pd.DataFrame(
            {
                col: np.concatenate([s1, s2]),
                "block": [f"B{b1}"] * len(s1) + [f"B{b2}"] * len(s2),
            }
        )
        fig = px.ecdf(
            df_ecdf, x=col, color="block", title=_sanitize_text(f"ECDF – {col}")
        )
        _safe_plot(fig)
    else:
        df_v = pd.DataFrame(
            {
                col: np.concatenate([s1, s2]),
                "block": [f"B{b1}"] * len(s1) + [f"B{b2}"] * len(s2),
            }
        )
        fig = go.Figure()
        for b in [f"B{b1}", f"B{b2}"]:
            fig.add_trace(
                go.Violin(
                    y=df_v[df_v["block"] == b][col],
                    x=[b] * len(df_v[df_v["block"] == b]),
                    name=_sanitize_text(b),
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig.update_layout(
            title=_sanitize_text(f"Violin/Box – {col}"),
            xaxis_title="Block",
            yaxis_title=_sanitize_text(col),
        )
        _safe_plot(fig)

    # PCA and PHT plots
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
                "`files_compared` not in drift results. Cannot generate PCA/PHT plots."
            )
            df_previous = pd.DataFrame()
            df_current = pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading data for drift comparison: {e}")
        return

    if not df_previous.empty and not df_current.empty:
        show_pca_drift_plot(df_previous, df_current)

    show_pht_plot(config, pipeline_name)


def show_historical_performance_section(pipeline_name: str):
    st.subheader("Historical Performance (block-wise)")

    base_dir = _get_pipeline_base_dir(pipeline_name)
    metrics_dir = base_dir / "metrics"
    eval_history_dir = metrics_dir / "eval_history"

    if not eval_history_dir.exists():
        _safe_markdown("No evaluation history found.")
        return

    history_files = sorted(
        [
            f
            for f in os.listdir(eval_history_dir)
            if f.startswith("eval_results_") and f.endswith(".json")
        ]
    )

    if not history_files:
        _safe_markdown("No historical evaluation data found.")
        return

    all_results = []
    for file_name in history_files:
        file_path = os.path.join(eval_history_dir, file_name)
        try:
            results = _read_json_cached(file_path, _mtime(file_path))
            if not results:
                continue

            timestamp_str = file_name.replace("eval_results_", "").replace(".json", "")
            timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
            results["timestamp"] = timestamp
            all_results.append(results)
        except Exception as e:
            st.warning(f"Could not read or parse {file_name}: {e}")

    if not all_results:
        st.warning("No valid historical evaluation data could be loaded.")
        return

    # Extract per-block performance data
    block_performance_data = []
    for result in all_results:
        timestamp = result["timestamp"]
        blocks_info = result.get("blocks", {})
        per_block_metrics = blocks_info.get("per_block_metrics_full", {})
        if per_block_metrics:
            for block, metrics in per_block_metrics.items():
                row = {"timestamp": timestamp, "block": block}
                row.update(metrics)
                block_performance_data.append(row)

    if not block_performance_data:
        _safe_markdown("No per-block performance data found in history.")
        return

    df_history = pd.DataFrame(block_performance_data)
    df_history = df_history.sort_values(by=["timestamp", "block"]).reset_index(
        drop=True
    )

    st.dataframe(df_history)

    # Plot performance over time
    available_metrics = [
        col for col in df_history.columns if col not in ["timestamp", "block"]
    ]
    if not available_metrics:
        _safe_markdown("No metrics available to plot.")
        return

    metric_to_plot = st.selectbox("Select Metric to Plot", options=available_metrics)

    if metric_to_plot:
        fig = px.line(
            df_history,
            x="timestamp",
            y=metric_to_plot,
            color="block",
            markers=True,
            title=f"Historical {metric_to_plot} by Block",
        )
        fig.update_traces(connectgaps=True)  # Connect points over gaps
        _safe_plot(fig)


def show_train_section(
    pipeline_name: str, split_within_blocks: bool, train_percentage: float
):
    st.subheader("🤖 Training / Retraining Results (block_wise)")
    metrics_dir = _get_pipeline_base_dir(pipeline_name) / "metrics"
    train_path = os.path.join(metrics_dir, "train_results.json")

    if not os.path.exists(train_path):
        _safe_markdown("No training results found yet.")
        return

    results = _read_json_cached(train_path, _mtime(train_path)) or {}
    if not results:
        _safe_markdown("Empty training results.")
        return

    st.markdown("### Run Configuration")
    if split_within_blocks:
        st.info(
            f"Split Mode: **Split Within Blocks**. Each block was split into {train_percentage * 100:.0f}% for training and {100 - (train_percentage * 100):.0f}% for evaluation."
        )
    else:
        train_blocks = results.get("train_blocks", [])
        st.info(
            f"Split Mode: **By Block**. Training was performed on dedicated blocks: `{train_blocks}`"
        )

    try:
        st.download_button(
            "⬇ Download train_results.json",
            data=json.dumps(results, indent=2),
            file_name="train_results.json",
            mime="application/json",
        )
    except Exception:
        pass

    _safe_markdown("## General Information")
    _type = str(results.get("type", results.get("type", "-"))).lower()
    _date = results.get("timestamp", "-")
    _model = results.get("model", results.get("model", "-"))
    _fallback_strategy = results.get("fallback_strategy", "global")

    _safe_write(f"*Type:* {_sanitize_text(_type)}")
    _safe_write(f"*Date:* {_sanitize_text(_date)}")
    _safe_write(f"*Model:* {_sanitize_text(_model)}")
    _safe_write(
        f"*Fallback Strategy:* **{_sanitize_text(_fallback_strategy.capitalize())}**"
    )

    train_blocks = [str(b) for b in (results.get("train_blocks") or [])]
    retrained_blocks = [str(b) for b in (results.get("retrained_blocks") or [])]
    is_retrain = _type == "retrain" and len(retrained_blocks) > 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Train size", int(results.get("train_size", 0)))
    k2.metric("Eval size", int(results.get("eval_size", 0)))
    k3.metric("Retrained blocks", len(retrained_blocks) if is_retrain else 0)

    show_only_retrained = False
    if is_retrain:
        show_only_retrained = st.checkbox(
            "Show only retrained blocks",
            value=True,
            help="When enabled, the selector and tables will only include the blocks retrained in this run.",
            key="show_only_retrained_blocks",
        )

    selector_pool = (
        retrained_blocks if (is_retrain and show_only_retrained) else train_blocks
    )
    if selector_pool:
        sel_label = (
            "Filter block (Retrained only)"
            if (is_retrain and show_only_retrained)
            else "Filter block (Train blocks only)"
        )
        sel_train = st.selectbox(
            sel_label,
            options=["(All)"] + selector_pool,
            index=0,
            key="train_block_selector",
        )
    else:
        sel_train = "(All)"

    per_blk = results.get("per_block_train", {}) or {}
    if per_blk:
        _safe_markdown("### 🧱 Per-block (in-sample)")
        df_blk = (
            pd.DataFrame.from_dict(per_blk, orient="index")
            .reset_index()
            .rename(columns={"index": "block"})
        )

        if sel_train != "(All)":
            df_blk = df_blk[df_blk["block"].astype(str) == sel_train]
        elif is_retrain and show_only_retrained and retrained_blocks:
            df_blk = df_blk[df_blk["block"].astype(str).isin(retrained_blocks)]

        if df_blk.empty:
            _safe_markdown("No per-block metrics to display for the selected filter.")
        else:
            _safe_table(df_blk)

            metric_for_line = None
            for m in ["accuracy", "balanced_accuracy", "f1", "r2", "rmse"]:
                if m in df_blk.columns:
                    metric_for_line = m
                    break
            if metric_for_line:
                df_line = _sanitize_df(df_blk.copy())
                try:
                    order = _sorted_blocks(df_line["block"])
                    df_line["block"] = pd.Categorical(
                        df_line["block"].astype(str),
                        categories=[str(o) for o in order],
                        ordered=True,
                    )
                    df_line = df_line.sort_values("block")
                except Exception:
                    df_line["block"] = df_line["block"].astype(str)

                df_line["block_str"] = df_line["block"].astype(str)

                title = f"{metric_for_line} by Block (Train Subset)"

                fig = px.line(
                    df_line,
                    x="block_str",
                    y=metric_for_line,
                    markers=True,
                    title=_sanitize_text(title),
                    labels={
                        "block_str": _sanitize_text("Block"),
                        metric_for_line: _sanitize_text(metric_for_line),
                    },
                )
                fig.update_xaxes(type="category")
                _safe_plot(fig)
    else:
        _safe_markdown("No per-block training metrics found in the report.")

    with st.expander(_sanitize_text("🔎 Raw training JSON")):
        _safe_json_display(results)


# =========================
# Logs Tab
# =========================
def parse_logs_to_df(log_text: str) -> pd.DataFrame:
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
    st.subheader("📜 Pipeline Logs")
    logs_dir = _get_pipeline_base_dir(pipeline_name) / "logs"
    logs_path = logs_dir / "pipeline.log"
    error_logs_path = logs_dir / "pipeline_errors.log"
    warning_logs_path = logs_dir / "pipeline_warnings.log"

    _safe_markdown("### 📘 General Log")
    general_log_text = _read_text_cached(logs_path, _mtime(logs_path))
    if general_log_text and general_log_text.strip():
        df_general = parse_logs_to_df(general_log_text)
        if not df_general.empty:
            df_general["message"] = df_general["message"].map(_sanitize_text)

            # Filters: level + search query
            lcol, qcol = st.columns([1, 3])
            level_sel = lcol.selectbox(
                "Level",
                options=["ALL", "INFO", "WARNING", "ERROR"],
                index=0,
                key="log_level",
            )
            query = qcol.text_input("Search text", value="", key="log_query")

            df_f = df_general.copy()
            if level_sel != "ALL":
                df_f = df_f[df_f["level"] == level_sel]
            if query:
                df_f = df_f[df_f["message"].str.contains(query, case=False, na=False)]

            st.dataframe(
                df_f.style.map(
                    lambda v: "color: red;"
                    if v == "ERROR"
                    else ("color: orange;" if v == "WARNING" else "color: green;"),
                    subset=["level"],
                ),
                use_container_width=True,
            )
        else:
            _safe_markdown("No general logs found.")
    else:
        _safe_markdown("No general logs found.")

    _safe_markdown("### Error Log")
    error_log_text = _read_text_cached(error_logs_path, _mtime(error_logs_path))
    if error_log_text and error_log_text.strip():
        st.code(_sanitize_text(error_log_text), language="bash")
    else:
        st.success(_sanitize_text("No errors found."))

    _safe_markdown("### ⚠ Warning Log")
    warning_log_text = _read_text_cached(warning_logs_path, _mtime(warning_logs_path))
    if warning_log_text and warning_log_text.strip():
        st.code(_sanitize_text(warning_log_text), language="bash")
    else:
        st.success(_sanitize_text("No warnings found."))


# =========================
# Main App
# =========================
st.set_page_config(page_title="Monitor ML Pipeline (Block-wise)", layout="wide")
st.title("Monitor (Block-wise)")

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_name", type=str, required=True)
args, _ = parser.parse_known_args()
pipeline_name = args.pipeline_name

# Load runner_config to get all settings
runner_config_path = (
    _get_pipeline_base_dir(pipeline_name) / "config" / "runner_config.json"
)
if os.path.exists(runner_config_path):
    runner_config = (
        _read_json_cached(runner_config_path, _mtime(runner_config_path)) or {}
    )
    pipeline_name = runner_config.get("pipeline_name", pipeline_name)
    data_dir = runner_config.get("data_dir")
    split_within_blocks = runner_config.get("split_within_blocks", False)
    train_percentage = runner_config.get("train_percentage", 0.8)
else:
    # Fallback to old config.json for compatibility
    config_path = _get_pipeline_base_dir(pipeline_name) / "config" / "config.json"
    if os.path.exists(config_path):
        config = _read_json_cached(config_path, _mtime(config_path)) or {}
        pipeline_name = config.get("pipeline_name", pipeline_name)
        data_dir = config.get("data_dir")
        split_within_blocks = False
        train_percentage = 0.8
    else:
        st.error(
            _sanitize_text(
                f"runner_config.json or config.json not found for pipeline '{pipeline_name}'. Please start the monitor correctly."
            )
        )
        st.stop()

if not data_dir:
    st.error(_sanitize_text("`data_dir` missing in configuration."))
    st.stop()

# =========================
# Global controls & data loading
# =========================
control_dir = _get_pipeline_base_dir(pipeline_name) / "control"
df, last_file = dashboard_data_loader(data_dir, control_dir)
block_col = _detect_block_col(pipeline_name, df)

tabs = st.tabs(["Dataset", "Evaluator", "Drift", "Evolution", "Train/Retrain", "Logs"])

# Auto-refresh on control_file change
control_file = _get_pipeline_base_dir(pipeline_name) / "control" / "control_file.txt"
if "control_file_mtime" not in st.session_state:
    st.session_state.control_file_mtime = _mtime(control_file)

current_mtime = _mtime(control_file)
if current_mtime > st.session_state.control_file_mtime:
    st.session_state.control_file_mtime = current_mtime
    st.experimental_rerun()

with tabs[0]:
    show_dataset_section(df, last_file, pipeline_name, block_col)

with tabs[1]:
    show_evaluator_section(pipeline_name, split_within_blocks, train_percentage)

with tabs[2]:
    cfg = runner_config if "runner_config" in locals() else config
    show_drift_section(pipeline_name, cfg)

with tabs[3]:
    show_evolution_section(pipeline_name)

with tabs[4]:
    show_train_section(pipeline_name, split_within_blocks, train_percentage)

with tabs[5]:
    show_logs_section(pipeline_name)
