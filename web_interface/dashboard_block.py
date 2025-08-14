import os
import sys
import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    _load_any_dataset,
    leer_metrics,
    load_csv,
    load_json,
    load_log,
    leer_registros,
    dashboard_data_loader,
    actualizar_registro,
)

# =========================
# Helpers (Bloques)
# =========================

def _detect_block_col(pipeline_name: str, df: pd.DataFrame, default: str = "block_id") -> str | None:
    """
    Detección de columna de bloque:
      1) pipelines/<pipeline>/config/config.json -> block_col
      2) pipelines/<pipeline>/control/blocks_snapshot.json -> última entrada de control_file
      3) Heurística: 'block_id' o columnas que contengan 'block'
    """
    # 1) config.json
    cfg_path = os.path.join("pipelines", pipeline_name, "config", "config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if cfg.get("block_col") and cfg["block_col"] in df.columns:
                return cfg["block_col"]
    except Exception:
        pass

    # 2) blocks_snapshot.json
    snap_path = os.path.join("pipelines", pipeline_name, "control", "blocks_snapshot.json")
    try:
        if os.path.exists(snap_path):
            with open(snap_path, "r") as f:
                snap = json.load(f)
            control_file = os.path.join("pipelines", pipeline_name, "control", "control_file.txt")
            file_key = None
            if os.path.exists(control_file):
                with open(control_file, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    file_key = lines[-1].split(",")[0]
            if file_key and snap.get(file_key, {}).get("block_col") in df.columns:
                return snap[file_key]["block_col"]
    except Exception:
        pass

    # 3) heurística
    if default in df.columns:
        return default
    for c in df.columns:
        if "block" in c.lower():
            return c
    return None


def _sorted_blocks(series: pd.Series):
    """Orden estable: intenta numérico, luego datetime, luego lexicográfico."""
    vals = series.dropna().unique().tolist()
    # numeric
    try:
        nums = [float(v) for v in vals]
        return [x for _, x in sorted(zip(nums, vals))]
    except Exception:
        pass
    # datetime
    try:
        dt = pd.to_datetime(vals, errors="raise")
        return [x for _, x in sorted(zip(dt, vals))]
    except Exception:
        pass
    # lexicographic
    return sorted(vals, key=lambda x: str(x))


# =========================
# Dataset Section
# =========================

def show_dataset_section(data_dir, pipeline_name):
    """Displays dataset preview, info, stats, and top categorical values — ahora por bloque."""
    st.subheader("📊 Dataset Information")
    control_dir = os.path.join("pipelines", pipeline_name, "control")
    df, last_file = dashboard_data_loader(data_dir, control_dir)

    if df.empty or not last_file:
        st.warning("⚠️ No processed dataset found yet.")
        return

    st.write(f"**Last processed dataset:** `{last_file}`")

    # >>> BLOQUES: detectar columna de bloque y resumir
    block_col = _detect_block_col(pipeline_name, df)
    if block_col and block_col in df.columns:
        st.info(f"🔢 Block column detected: **`{block_col}`**")
        blocks = _sorted_blocks(df[block_col])
        counts = df[block_col].value_counts(dropna=False)
        # Reindexar counts con el orden de blocks
        counts = counts.reindex(blocks, fill_value=0)

        st.markdown("### 🧱 Blocks overview")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(pd.DataFrame({"block": [str(b) for b in counts.index], "rows": counts.values}))
        with c2:
            fig = px.bar(
                x=[str(b) for b in counts.index],
                y=counts.values,
                labels={"x": "Block", "y": "Rows"},
                title="Rows per block",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Selector de bloque (incluye opción 'All')
        sel = st.selectbox("Focus block", options=["(All)"] + [str(b) for b in blocks], index=0)
        if sel != "(All)":
            df_view = df[df[block_col].astype(str) == sel]
        else:
            df_view = df
    else:
        st.info("No block column detected. Showing global info.")
        df_view = df
        block_col = None

    # Preview
    st.markdown("### 👀 Preview (head)")
    st.dataframe(df_view.head(10))

    # Dataset info table
    st.markdown("### 🗂️ Dataset Info")
    info_dict = {
        "Column": df_view.columns,
        "Non-Null Count": [df_view[col].notnull().sum() for col in df_view.columns],
        "Unique Values": [df_view[col].nunique(dropna=True) for col in df_view.columns],
        "Dtype": df_view.dtypes.values,
    }
    info_df = pd.DataFrame(info_dict)
    st.dataframe(info_df)

    # General summary
    st.markdown(
        f"""
    **Total Rows (view):** {df_view.shape[0]}  
    **Total Columns:** {df_view.shape[1]}  
    **Memory Usage (view):** {df_view.memory_usage().sum() / 1024**2:.2f} MB
    """
    )

    # Descriptive stats
    st.markdown("### 📈 Descriptive Statistics (view)")
    st.dataframe(df_view.describe(include="all").transpose())

    # Top categorical value counts (view)
    st.markdown("### 🔎 Top 5 Most Frequent Values in Categorical Columns (view)")
    cat_cols = df_view.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            st.markdown(f"**{col}**")
            freq = df_view[col].value_counts().head(5).reset_index()
            freq.columns = [col, "Frequency"]
            fig = px.bar(
                freq,
                x=col,
                y="Frequency",
                text="Frequency",
                title=f"Top 5 in {col}",
                labels={col: "Values", "Frequency": "Frequency"},
            )
            fig.update_traces(textposition="outside", marker=dict(color="skyblue", line=dict(color="black", width=1)))
            fig.update_layout(xaxis_tickangle=-30, height=400, width=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns to show frequencies (in current view).")


# =========================
# Evaluator Section
# =========================

def show_evaluator_section(pipeline_name):
    """Displays evaluation metrics, thresholds, circuit breaker status, and candidates overview (ahora block-aware)."""
    st.subheader("🧪 Evaluation Results")

    base_dir = os.path.join("pipelines", pipeline_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    candidates_dir = os.path.join(base_dir, "candidates")
    eval_path = os.path.join(metrics_dir, "eval_results.json")
    health_path = os.path.join(metrics_dir, "health.json")

    if not os.path.exists(eval_path):
        st.info("No evaluation results found yet.")
        return

    with open(eval_path, "r") as f:
        results = json.load(f)

    if not results:
        st.warning("Empty evaluation results.")
        return

    # Approval flag
    if results.get("approved", False):
        st.success("✅ Model approved. Meets the established thresholds.")
    else:
        st.error("❌ Model NOT approved. Does not meet the established thresholds.")

    # >>> BLOQUES: sección de bloques si existe
    blocks_info = results.get("blocks", {}) or {}
    if blocks_info:
        st.markdown("## 🧱 Blocks (from evaluator)")
        cols = st.columns(3)
        cols[0].metric("Block column", str(blocks_info.get("block_col")))
        cols[1].metric("Evaluated block", str(blocks_info.get("evaluated_block_id")))
        ref_blocks = blocks_info.get("reference_blocks") or []
        cols[2].metric("Reference blocks", len(ref_blocks))
        if ref_blocks:
            st.caption(f"Reference blocks: {', '.join(map(str, ref_blocks))}")

        per_blk = blocks_info.get("per_block_metrics") or {}
        if per_blk:
            st.markdown("### 📊 Per-block metrics (test)")
            df_blk = pd.DataFrame.from_dict(per_blk, orient="index").reset_index().rename(columns={"index": "block"})
            st.dataframe(df_blk)
            # quick chart: primer métrico disponible
            for m in ["accuracy", "f1", "balanced_accuracy", "r2", "rmse"]:
                if m in df_blk.columns:
                    st.plotly_chart(px.bar(df_blk, x="block", y=m, title=f"{m} by block"), use_container_width=True)
                    break

    # Thresholds
    st.markdown("## 📏 Used Thresholds")
    thresholds = results.get("thresholds", {})
    if thresholds:
        thresholds_df = pd.DataFrame(list(thresholds.items()), columns=["Metric", "Threshold"])
        st.table(thresholds_df)
    else:
        st.info("No thresholds found in the evaluation results.")

    # Prediction examples
    if "predictions" in results and results["predictions"]:
        st.markdown("## 🔍 Prediction Examples")
        preds_df = pd.DataFrame(results["predictions"])
        st.table(preds_df)

    # Metrics
    st.markdown("## 📊 Test Metrics")
    metrics = results.get("metrics", {})
    if "classification_report" in metrics:
        st.write(f"**Accuracy:** {round(metrics.get('accuracy', 0), 4)}")
        st.write(f"**Balanced Accuracy:** {round(metrics.get('balanced_accuracy', 0), 4)}")
        st.write(f"**F1 (macro):** {round(metrics.get('f1', 0), 4)}")

        report_df = pd.DataFrame(metrics["classification_report"])
        if set(["precision", "recall", "f1-score", "support"]).issubset(report_df.index):
            st.table(report_df.transpose())
        else:
            st.table(report_df)
    elif "r2" in metrics:
        st.write(f"**R²:** {round(metrics.get('r2', 0), 4)}")
        st.write(f"**RMSE:** {round(metrics.get('rmse', 0), 4)}")
        st.write(f"**MAE:** {round(metrics.get('mae', 0), 4)}")
        st.write(f"**MSE:** {round(metrics.get('mse', 0), 4)}")
    else:
        st.info("No evaluation metrics found.")

    # Circuit Breaker
    st.markdown("## 🛑 Circuit Breaker Status")
    if os.path.exists(health_path):
        try:
            with open(health_path, "r") as f:
                health = json.load(f)

            consecutive = int(health.get("consecutive_failures", 0) or 0)
            paused_until = health.get("paused_until")
            last_failure_ts = health.get("last_failure_ts")

            cols = st.columns(3)
            cols[0].metric("Consecutive Failures", consecutive)

            def _fmt_ts(ts):
                try:
                    import time as _t, datetime as _dt
                    return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return str(ts)

            cols[1].metric("Paused Until", _fmt_ts(paused_until) if paused_until else "—")
            cols[2].metric("Last Failure", _fmt_ts(last_failure_ts) if last_failure_ts else "—")

            import time as _t
            paused = bool(paused_until and _t.time() < float(paused_until))
            st.warning("⏸️ Retraining is currently **paused** by the circuit breaker.") if paused else st.success("▶️ Retraining is **active** (not paused).")
        except Exception as e:
            st.warning(f"Could not read health.json: {e}")
    else:
        st.info("No circuit breaker state found yet (health.json).")

    # Candidates overview
    st.markdown("## 🗂️ Candidates (Non-Approved Models)")
    if not os.path.exists(candidates_dir):
        st.info("No candidates directory yet.")
        return

    candidates = []
    try:
        for entry in sorted(
            [os.path.join(candidates_dir, d) for d in os.listdir(candidates_dir) if os.path.isdir(os.path.join(candidates_dir, d))],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )[:10]:
            meta_path = os.path.join(entry, "meta.json")
            eval_p = os.path.join(entry, "eval_results.json")
            row = {"path": entry, "timestamp": None, "file": None, "approved": False, "key_metric": None, "metric_value": None}
            try:
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    row["approved"] = bool(meta.get("approved", False))
                    row["file"] = meta.get("file")
                    row["timestamp"] = meta.get("timestamp")
                if os.path.exists(eval_p):
                    with open(eval_p, "r") as f:
                        ev = json.load(f)
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
                df_cand = df_cand.sort_values(by="timestamp", ascending=False, na_position="last")
            show_cols = ["timestamp", "file", "approved", "key_metric", "metric_value", "path"]
            st.dataframe(df_cand[show_cols])
            st.caption("Showing up to 10 latest candidates. Each folder contains `model.pkl` and `eval_results.json`.")
        else:
            st.info("No candidates have been saved yet.")
    except Exception as e:
        st.warning(f"Could not enumerate candidates: {e}")


# =========================
# Drift Section (block-aware)
# =========================

def show_drift_section(pipeline_name):
    """
    Visual drift dashboard — con selector de bloque:
      - Decision badge + resumen
      - Performance checks (current/previous/comparison)
      - Multivariate tests
      - Feature explorer: reference (previous_data.csv) vs bloque seleccionado del último dataset
      - Drift heatmap por feature
      - Timeline de decisiones por bloque (si existe)
    """
    st.subheader("🌊 Drift Results")
    base_dir = os.path.join("pipelines", pipeline_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    control_dir = os.path.join(base_dir, "control")
    config_path = os.path.join(base_dir, "config", "config.json")

    # Load config for data_dir
    data_dir = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        data_dir = cfg.get("data_dir")

    # Blocks timeline (if exists)
    blocks_report_path = os.path.join(metrics_dir, "blocks_training_report.json")
    if os.path.exists(blocks_report_path):
        try:
            with open(blocks_report_path, "r") as f:
                rpt = json.load(f)
            st.markdown("## 🧭 Blocks decisions timeline")
            if isinstance(rpt.get("decisions"), list) and rpt["decisions"]:
                df_dec = pd.DataFrame(rpt["decisions"])
                st.dataframe(df_dec)
                cnt = df_dec["decision"].value_counts()
                st.plotly_chart(px.bar(x=cnt.index, y=cnt.values, title="Decisions count"), use_container_width=True)
        except Exception as e:
            st.info(f"(optional) Could not load blocks_training_report.json: {e}")

    # Load drift JSON
    drift_path = os.path.join(metrics_dir, "drift_results.json")
    if not os.path.exists(drift_path):
        st.info("No drift results saved yet.")
        return

    with open(drift_path, "r") as f:
        results = json.load(f)
    if not results:
        st.warning("Drift results are empty.")
        return

    tests = results.get("tests", {})
    drift_flags = results.get("drift", {})

    # Decision Badge
    decision = results.get("decision")
    if decision:
        if decision == "no_drift":
            st.success("🟢 No drift detected – current model maintained")
        elif decision == "previous_promoted":
            reason = results.get("promotion_reason")
            msg = "🔄 Previous model promoted due to better performance"
            if reason:
                msg += f" (reason: `{reason}`)"
            st.warning(msg)
        elif decision == "retrain":
            st.error("🛠 Retraining triggered due to drift")
        elif decision == "train":
            st.info("🆕 First run: training from scratch")
        elif decision == "end_error":
            st.warning("⚠️ Ended with error while loading current model")
        else:
            st.info(f"ℹ️ Decision: {decision}")

    # Summary
    has_drift_section = isinstance(drift_flags, dict) and len(drift_flags) > 0
    if has_drift_section:
        st.markdown("## 📌 Drift Summary")
        promoted_key = "promoted_model" if "promoted_model" in results else ("modelo_promovido" if "modelo_promovido" in results else None)
        if promoted_key is not None:
            val = results.get(promoted_key)
            if val is True:
                st.success("📌 The previous model was promoted to the current one.")
                reason = results.get("promotion_reason")
                if reason:
                    st.info(f"**Promotion reason:** `{reason}`")
            elif val is False:
                st.info("ℹ️ The previous model was NOT promoted.")
            elif val == "error":
                st.warning("⚠️ Error comparing with the previous model.")

        summary_data = [
            {"Test": k, "Result": "⚠️ Drift detected" if bool(v) else "✅ No drift detected"} for k, v in drift_flags.items()
        ]
        summary_df = pd.DataFrame(summary_data)
        total = len(summary_df)
        detected = (summary_df["Result"] == "⚠️ Drift detected").sum()
        st.success(f"✅ Tests without drift: {total - detected} of {total}")
        st.error(f"⚠️ Tests with drift: {detected} of {total}")

        def _color_rows(val: str):
            return (
                "background-color: #ff4d4d; color: white; font-weight: bold;"
                if "Drift detected" in val
                else "background-color: #33cc33; color: white; font-weight: bold;"
            )

        st.dataframe(summary_df.style.map(_color_rows, subset=["Result"]))
    else:
        st.info("No drift results found.")
        return

    # Performance tables
    st.markdown("## 🧪 Performance Checks")

    def _style_bool(val: bool):
        return "background-color: #ff4d4d; color: white; font-weight: 600;" if val else "background-color: #33cc33; color: white; font-weight: 600;"

    def _render_perf_table(perf_dict: dict, title: str):
        if not isinstance(perf_dict, dict) or len(perf_dict) == 0:
            return
        rows = []
        for metric_name, payload in perf_dict.items():
            value = (
                payload.get(metric_name.lower())
                or payload.get("accuracy")
                or payload.get("balanced_accuracy")
                or payload.get("F1")
                or payload.get("RMSE")
                or payload.get("R2")
                or payload.get("MAE")
                or payload.get("MSE")
                or payload.get("value")
            )
            rows.append(
                {
                    "Metric": metric_name,
                    "Value": value,
                    "Threshold": payload.get("threshold"),
                    "Drift": bool(payload.get("drift", False)),
                }
            )
        df = pd.DataFrame(rows)
        st.markdown(f"### {title}")
        st.dataframe(df.style.map(lambda v: _style_bool(v) if isinstance(v, bool) else "", subset=["Drift"]))

    if "Performance_Current" in tests:
        _render_perf_table(tests["Performance_Current"], "Current Model vs thresholds")
    if "Performance_Previous" in tests:
        _render_perf_table(tests["Performance_Previous"], "Previous Model vs thresholds")
    if "Performance_Comparison" in tests and isinstance(tests["Performance_Comparison"], dict):
        comp_rows = []
        for mname, payload in tests["Performance_Comparison"].items():
            metric = payload.get("metric", mname)
            prev_v = payload.get("prev")
            curr_v = payload.get("current")
            thr = payload.get("threshold")
            change = payload.get("relative_drop", payload.get("relative_increase"))
            # drift key: prueba exacta y prefijo como fallback
            dk_exact = f"comparison::{metric}"
            dk_prefix = f"comparison::{metric.split('_')[0]}"
            comp_rows.append(
                {
                    "Metric": metric,
                    "Previous": prev_v,
                    "Current": curr_v,
                    "Relative change": change if change is None else f"{float(change) * 100:.2f}%",
                    "Threshold": thr,
                    "Drift": bool(drift_flags.get(dk_exact, drift_flags.get(dk_prefix, False))),
                }
            )
        if comp_rows:
            st.markdown("### 🔁 Previous vs Current (Relative change)")
            st.dataframe(pd.DataFrame(comp_rows).style.map(lambda v: _style_bool(v) if isinstance(v, bool) else "", subset=["Drift"]))

    # Multivariate tests
    if ("MMD" in tests) or ("Energy Distance" in tests):
        st.markdown("## 🧭 Multivariate Tests")
        cols = st.columns(2)
        if "MMD" in tests:
            m = tests["MMD"]
            with cols[0]:
                st.markdown("**MMD**")
                st.write(f"Statistic: `{m.get('statistic')}`")
                pv = m.get("p_value", m.get("pvalue", m.get("p")))
                if pv is not None:
                    st.write(f"p-value: `{pv}`")
                if m.get("kernel"):
                    st.write(f"Kernel: `{m.get('kernel')}`")
                if m.get("bandwidth") is not None:
                    st.write(f"Bandwidth: `{m.get('bandwidth')}`")
                st.markdown("Drift: " + ("**⚠️ YES**" if bool(m.get("drift")) else "**✅ NO**"))
        if "Energy Distance" in tests:
            e = tests["Energy Distance"]
            with cols[1]:
                st.markdown("**Energy Distance**")
                dist = e.get("distance", e.get("emd_distance", e.get("emd_norm")))
                st.write(f"Distance: `{dist}`")
                pv = e.get("p_value", e.get("pvalue", e.get("p")))
                if pv is not None:
                    st.write(f"p-value: `{pv}`")
                st.markdown("Drift: " + ("**⚠️ YES**" if bool(e.get("drift")) else "**✅ NO**"))

    # Load reference and current dataset
    prev_path = os.path.join(control_dir, "previous_data.csv")
    df_prev = pd.read_csv(prev_path) if os.path.exists(prev_path) else None

    df_current = None
    last_file = None
    control_file = os.path.join(control_dir, "control_file.txt")
    if os.path.exists(control_file) and data_dir:
        try:
            with open(control_file, "r") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            if lines:
                last_file = lines[-1].split(",")[0]
                cand = os.path.join(data_dir, last_file)
                if os.path.exists(cand):
                    df_current = _load_any_dataset(cand)
        except Exception as e:
            st.error(f"Error loading latest dataset: {e}")
            df_current = None

    st.markdown("## 🔎 Feature Explorer (Reference vs Block)")
    if df_prev is None or df_current is None:
        st.info("I need both: `previous_data.csv` and the latest dataset. Skipping plots.")
        return

    # >>> BLOQUES: selector de bloque del dataset actual
    block_col = _detect_block_col(pipeline_name, df_current)
    if block_col and block_col in df_current.columns:
        block_options = _sorted_blocks(df_current[block_col])
        sel_block = st.selectbox("Select block to compare against reference window", options=[str(b) for b in block_options])
        df_curr_view = df_current[df_current[block_col].astype(str) == sel_block]
    else:
        st.info("No block column detected in current dataset — using full dataset as 'current'.")
        df_curr_view = df_current

    # Keep only common numeric columns (quita target típicos si están)
    common_cols = [c for c in df_prev.columns if c in df_curr_view.columns]
    num_cols = [
        c
        for c in common_cols
        if pd.api.types.is_numeric_dtype(df_prev[c]) and pd.api.types.is_numeric_dtype(df_curr_view[c])
    ]
    for cand in ("class", "target", "y"):
        if cand in num_cols:
            num_cols.remove(cand)

    if not num_cols:
        st.info("No common numeric columns found for distribution plots.")
        return

    left, right = st.columns([2, 1])
    with right:
        col = st.selectbox("Variable", options=num_cols, index=0)
        bins = st.slider("Bins", min_value=10, max_value=100, value=40, step=5)
        show_logx = st.checkbox("Log X-axis", value=False)
        show_norm = st.checkbox("Normalize histograms", value=True)

    s_prev = df_prev[col].dropna().astype(float)
    s_curr = df_curr_view[col].dropna().astype(float)

    with left:
        st.markdown(f"### {col}")

    # Histogram overlay
    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=s_prev, name="Reference", opacity=0.55, nbinsx=bins, histnorm="probability" if show_norm else ""
        )
    )
    hist_fig.add_trace(
        go.Histogram(
            x=s_curr,
            name="Selected block" if (block_col and block_col in df_current.columns) else "Current",
            opacity=0.55,
            nbinsx=bins,
            histnorm="probability" if show_norm else "",
        )
    )
    hist_fig.update_layout(
        barmode="overlay", title=f"Histogram – {col}", xaxis_title=col, yaxis_title="Density" if show_norm else "Count"
    )
    if show_logx:
        hist_fig.update_xaxes(type="log")
    st.plotly_chart(hist_fig, use_container_width=True)

    # ECDF overlay
    def _ecdf(x):
        x_sorted = np.sort(x)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    x1, y1 = _ecdf(s_prev.values)
    x2, y2 = _ecdf(s_curr.values)
    ecdf_fig = go.Figure()
    ecdf_fig.add_trace(go.Scatter(x=x1, y=y1, mode="lines", name="Reference"))
    ecdf_fig.add_trace(
        go.Scatter(
            x=x2, y=y2, mode="lines", name="Selected block" if (block_col and block_col in df_current.columns) else "Current"
        )
    )
    ecdf_fig.update_layout(title=f"ECDF – {col}", xaxis_title=col, yaxis_title="Cumulative probability")
    if show_logx:
        ecdf_fig.update_xaxes(type="log")
    st.plotly_chart(ecdf_fig, use_container_width=True)

    # Q–Q plot
    q = np.linspace(0.01, 0.99, 99)
    q_prev = np.quantile(s_prev, q)
    q_curr = np.quantile(s_curr, q)
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(x=q_prev, y=q_curr, mode="markers", name="Q–Q"))
    minv = float(np.nanmin([q_prev.min(), q_curr.min()]))
    maxv = float(np.nanmax([q_prev.max(), q_curr.max()]))
    qq_fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="y=x", line=dict(dash="dash")))
    qq_fig.update_layout(title=f"Q–Q Plot – {col}", xaxis_title="Reference quantiles", yaxis_title="Block quantiles")
    if show_logx:
        qq_fig.update_xaxes(type="log")
        qq_fig.update_yaxes(type="log")
    st.plotly_chart(qq_fig, use_container_width=True)

    # Violin + Box
    vb_fig = go.Figure()
    vb_fig.add_trace(go.Violin(y=s_prev, name="Reference", box_visible=True, meanline_visible=True, points="all", jitter=0.1))
    vb_fig.add_trace(
        go.Violin(
            y=s_curr,
            name="Selected block" if (block_col and block_col in df_current.columns) else "Current",
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.1,
        )
    )
    vb_fig.update_layout(title=f"Violin + Box – {col}", yaxis_title=col)
    st.plotly_chart(vb_fig, use_container_width=True)

    # Drift Heatmap
    st.markdown("## 🗺️ Drift Heatmap")
    metric_options = []
    if "Kolmogorov-Smirnov" in tests:
        metric_options.append("KS p-value")
    if "PSI" in tests:
        metric_options.append("PSI")
    if "Hellinger Distance" in tests:
        metric_options.append("Hellinger")
    if "Earth Mover's Distance" in tests:
        metric_options.append("EMD")
    if not metric_options:
        st.info("No per-feature drift metrics found to build a heatmap.")
        return

    msel = st.selectbox("Heatmap metric", options=metric_options, index=0)
    feat_vals = {}

    if msel == "KS p-value" and "Kolmogorov-Smirnov" in tests:
        for feat, payload in tests["Kolmogorov-Smirnov"].items():
            val = payload.get("p_value", payload.get("pvalue", payload.get("p")))
            feat_vals[feat] = float(val) if val is not None else None
    elif msel == "PSI" and "PSI" in tests:
        for feat, payload in tests["PSI"].items():
            if "psi" in payload and payload["psi"] is not None:
                feat_vals[feat] = float(payload["psi"])
    elif msel == "Hellinger" and "Hellinger Distance" in tests:
        for feat, payload in tests["Hellinger Distance"].items():
            v = payload.get("hellinger") or payload.get("hellinger_distance")
            feat_vals[feat] = float(v) if v is not None else None
    elif msel == "EMD" and "Earth Mover's Distance" in tests:
        for feat, payload in tests["Earth Mover's Distance"].items():
            v = payload.get("emd_norm") or payload.get("emd_distance") or payload.get("emd") or payload.get("distance")
            feat_vals[feat] = float(v) if v is not None else None

    if not feat_vals:
        st.info("No values available for the selected metric.")
        return

    heat_df = pd.DataFrame({"feature": list(feat_vals.keys()), "value": list(feat_vals.values())}).dropna()
    if heat_df.empty:
        st.info("No numeric values available for heatmap.")
        return

    heat_fig = go.Figure(
        data=go.Heatmap(
            z=heat_df["value"].values.reshape(1, -1),
            x=heat_df["feature"].tolist(),
            y=[msel],
            coloraxis="coloraxis",
        )
    )
    heat_fig.update_layout(
        title=f"Drift heatmap by feature – {msel}",
        xaxis=dict(title="Feature", tickangle=-45),
        yaxis=dict(title=""),
        coloraxis=dict(colorbar=dict(title=msel)),
    )
    st.plotly_chart(heat_fig, use_container_width=True)


# =========================
# Train / Retrain Section
# =========================

def show_train_section(pipeline_name):
    """Displays training or retraining metrics (ahora muestra bloques si hay)."""
    st.subheader("🤖 Training / Retraining Results")
    train_path = os.path.join("pipelines", pipeline_name, "metrics", "train_results.json")

    if not os.path.exists(train_path):
        st.info("No training results found yet.")
        return

    with open(train_path, "r") as f:
        results = json.load(f)

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

    st.markdown("## 📌 General Information")
    _type = results.get("tipo", results.get("type", "-"))
    _file = results.get("archivo", results.get("file", "-"))
    _date = results.get("timestamp", "-")
    _model = results.get("modelo", results.get("model", "-"))

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
            if fallback is not None:
                st.markdown("**Fallback:** " + ("🟠 Enabled (used)" if bool(fallback) else "🟢 Not used"))

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

    st.markdown("## 📊 Metrics")
    if "classification_report" in results:
        st.markdown("### 📈 Classification Metrics")
        bal_acc = results.get("balanced_accuracy", results.get("balanced_accuracy_score"))
        if bal_acc is not None:
            try:
                st.write(f"**Balanced Accuracy:** {float(bal_acc):.4f}")
            except Exception:
                st.write(f"**Balanced Accuracy:** {bal_acc}")
        clf_report = results["classification_report"]
        st.markdown("#### 📋 Classification Report")
        clf_report_df = pd.DataFrame(clf_report)
        if set(["precision", "recall", "f1-score", "support"]).issubset(clf_report_df.index):
            st.dataframe(clf_report_df.transpose())
        else:
            st.dataframe(clf_report_df)
    elif all(k in results for k in ["r2", "rmse", "mae", "mse"]):
        st.markdown("### 📈 Regression Metrics")
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

    with st.expander("🔎 Raw training JSON"):
        st.json(results)


# =========================
# Logs Section
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
            rows.append({"date": None, "pipeline": None, "level": None, "message": line})
    return pd.DataFrame(rows)


def show_logs_section(pipeline_name):
    st.subheader("📜 Pipeline Logs")
    logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline.log")
    error_logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline_errors.log")
    warning_logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline_warnings.log")

    st.markdown("### 📘 General Log")
    general_log_text = load_log(logs_path)
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
            )
        )
    else:
        st.info("No general logs found.")

    st.markdown("### ❌ Error Log")
    error_log_text = load_log(error_logs_path)
    if error_log_text.strip():
        st.code(error_log_text, language="bash")
    else:
        st.success("✅ No errors found.")

    st.markdown("### ⚠️ Warning Log")
    warning_log_text = load_log(warning_logs_path)
    if warning_log_text.strip():
        st.code(warning_log_text, language="bash")
    else:
        st.success("✅ No warnings found.")


# =========================
# Main App (Page Layout)
# =========================

st.set_page_config(page_title="Monitor ML Pipeline (Blocks)", layout="wide")
st.title("📌 Monitor (Blocks)")

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_name", type=str, required=True)
args, _ = parser.parse_known_args()
pipeline_name = args.pipeline_name

config_path = os.path.join("pipelines", pipeline_name, "config", "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    pipeline_name = config.get("pipeline_name", pipeline_name)
    data_dir = config.get("data_dir")
else:
    st.error("config.json not found. Please start the monitor with a valid pipeline.")
    st.stop()

if not data_dir:
    st.error("`data_dir` missing in config.json.")
    st.stop()

tabs = st.tabs(["📊 Dataset", "📈 Evaluator", "🔍 Drift", "🤖 Train/Retrain", "📜 Logs"])

with tabs[0]:
    show_dataset_section(data_dir, pipeline_name)

with tabs[1]:
    show_evaluator_section(pipeline_name)

with tabs[2]:
    show_drift_section(pipeline_name)

with tabs[3]:
    show_train_section(pipeline_name)

with tabs[4]:
    show_logs_section(pipeline_name)
