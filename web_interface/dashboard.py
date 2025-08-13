
import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from scipy.io import arff
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import argparse
import numpy as np
from utils import _load_any_dataset, leer_metrics, load_csv, load_json, load_log, leer_registros, dashboard_data_loader, actualizar_registro


# =========================
# Dataset Section
# =========================
def show_dataset_section(data_dir, pipeline_name):
    """Displays dataset preview, info, stats, and top categorical values."""
    st.subheader("📊 Dataset Information")
    control_dir = os.path.join("pipelines", pipeline_name, "control")
    df, last_file = dashboard_data_loader(data_dir, control_dir)

    if df.empty or not last_file:
        st.warning("⚠️ No processed dataset found yet.")
        return

    st.write(f"**Last processed dataset:** `{last_file}`")

    # Preview of first rows
    st.markdown("### 👀 Preview (head)")
    st.dataframe(df.head(10))

    # Dataset info table
    st.markdown("### 🗂️ Dataset Info")
    info_dict = {
        "Column": df.columns,
        "Non-Null Count": [df[col].notnull().sum() for col in df.columns],
        "Unique Values": [df[col].nunique(dropna=True) for col in df.columns],
        "Dtype": df.dtypes.values
    }
    info_df = pd.DataFrame(info_dict)
    st.dataframe(info_df)

    # General dataset summary
    st.markdown(f"""
    **Total Rows:** {df.shape[0]}  
    **Total Columns:** {df.shape[1]}  
    **Memory Usage:** {df.memory_usage().sum() / 1024**2:.2f} MB
    """)

    # Descriptive statistics for numeric and categorical columns
    st.markdown("### 📈 Descriptive Statistics")
    st.dataframe(df.describe(include="all").transpose())

    # Top categorical value counts
    st.markdown("### 🔎 Top 5 Most Frequent Values in Categorical Columns")
    cat_cols = df.select_dtypes(include="object").columns
    if not cat_cols.empty:
        for col in cat_cols:
            st.markdown(f"**{col}**")
            freq = df[col].value_counts().head(5).reset_index()
            freq.columns = [col, "Frequency"]

            fig = px.bar(
                freq,
                x=col,
                y="Frequency",
                text="Frequency",
                title=f"Top 5 in {col}",
                labels={col: "Values", "Frequency": "Frequency"}
            )
            fig.update_traces(textposition="outside", marker=dict(color="skyblue", line=dict(color="black", width=1)))
            fig.update_layout(xaxis_tickangle=-30, height=400, width=500)

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns to show frequencies.")

# =========================
# Evaluator Section
# =========================
def show_evaluator_section(pipeline_name):
    """Displays evaluation metrics, thresholds, circuit breaker status, and candidates overview."""
    st.subheader("🧪 Evaluation Results")

    base_dir = os.path.join("pipelines", pipeline_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    candidates_dir = os.path.join(base_dir, "candidates")
    eval_path = os.path.join(metrics_dir, "eval_results.json")
    health_path = os.path.join(metrics_dir, "health.json")

    # ------------------------
    # Current eval results
    # ------------------------
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
        # Normalize orientation if needed
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

    # ------------------------
    # Circuit Breaker status
    # ------------------------
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

            # Pretty-print timestamps if present
            def _fmt_ts(ts):
                try:
                    import time, datetime
                    return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return str(ts)

            if paused_until:
                cols[1].metric("Paused Until", _fmt_ts(paused_until))
            else:
                cols[1].metric("Paused Until", "—")

            if last_failure_ts:
                cols[2].metric("Last Failure", _fmt_ts(last_failure_ts))
            else:
                cols[2].metric("Last Failure", "—")

            # Show a hint if currently paused
            import time as _t
            paused = bool(paused_until and _t.time() < float(paused_until))
            if paused:
                st.warning("⏸️ Retraining is currently **paused** by the circuit breaker.")
            else:
                st.success("▶️ Retraining is **active** (not paused).")
        except Exception as e:
            st.warning(f"Could not read health.json: {e}")
    else:
        st.info("No circuit breaker state found yet (health.json).")

    # ------------------------
    # Candidates overview
    # ------------------------
    st.markdown("## 🗂️ Candidates (Non-Approved Models)")
    if not os.path.exists(candidates_dir):
        st.info("No candidates directory yet.")
        return

    # Gather last N candidates (by folder name / mtime)
    candidates = []
    try:
        for entry in sorted(
            [os.path.join(candidates_dir, d) for d in os.listdir(candidates_dir) if os.path.isdir(os.path.join(candidates_dir, d))],
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )[:10]:  # show up to 10 latest
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
                # Try to extract a key metric for quick sorting/reading
                if os.path.exists(eval_p):
                    with open(eval_p, "r") as f:
                        ev = json.load(f)
                    m = ev.get("metrics", {})
                    # Prefer classification metrics; else regression
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
            # Order for readability: timestamp desc
            if "timestamp" in df_cand.columns:
                df_cand = df_cand.sort_values(by="timestamp", ascending=False, na_position="last")
            # Pretty print
            show_cols = ["timestamp", "file", "approved", "key_metric", "metric_value", "path"]
            st.dataframe(df_cand[show_cols])
            st.caption("Showing up to 10 latest candidates. Each folder contains `model.pkl` and `eval_results.json`.")
        else:
            st.info("No candidates have been saved yet.")
    except Exception as e:
        st.warning(f"Could not enumerate candidates: {e}")

# =========================
# Drift Section (revamped, Frouros-compatible)
# =========================
def show_drift_section(pipeline_name):
    """
    Visual, interactive drift dashboard:
    - Decision badge + summary
    - Performance checks (current / previous / comparison)
    - Multivariate tests (MMD, Energy Distance) summary
    - Feature explorer with multiple plots:
        - Overlaid histograms
        - ECDF overlay (KS-style view)
        - Q-Q plot
        - Violin + Box
    - Drift heatmap by feature (KS p-value, PSI, Hellinger, EMD)
    Falls back gracefully if artifacts are missing.
    """
    # --- local loader to support many formats (csv/txt/arff/json/xls/xlsx/parquet)
    def _load_any_dataset(path_str: str):
        from pathlib import Path
        from scipy.io import arff as _arff
        p = Path(path_str)
        ext = p.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(p)
        if ext == ".txt":
            return pd.read_csv(p, sep=None, engine="python")  # autodetect delimiter
        if ext == ".arff":
            data, meta = _arff.loadarff(p)
            df_ = pd.DataFrame(data)
            # decode byte strings if present
            for c in df_.select_dtypes(include=["object"]).columns:
                if df_[c].apply(lambda v: isinstance(v, (bytes, bytearray))).any():
                    df_[c] = df_[c].apply(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
            return df_
        if ext == ".json":
            return pd.read_json(p)
        if ext in (".xls", ".xlsx"):
            return pd.read_excel(p)
        if ext == ".parquet":
            return pd.read_parquet(p)
        raise ValueError(f"Unsupported format: {ext} for {p}")

    st.subheader("🌊 Drift Results")
    base_dir = os.path.join("pipelines", pipeline_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    control_dir = os.path.join(base_dir, "control")
    config_path = os.path.join(base_dir, "config", "config.json")

    # Load config (to find data_dir for loading latest dataset)
    data_dir = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        data_dir = cfg.get("data_dir")

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

    # =========================
    # Decision Badge
    # =========================
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

    # =========================
    # Summary (flags across tests)
    # =========================
    has_drift_section = isinstance(drift_flags, dict) and len(drift_flags) > 0
    if has_drift_section:
        st.markdown("## 📌 Drift Summary")

        # Promotion info (if present)
        promoted_key = "promoted_model" if "promoted_model" in results else (
            "modelo_promovido" if "modelo_promovido" in results else None
        )
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

        # Tabular summary
        summary_data = [{"Test": k, "Result": "⚠️ Drift detected" if bool(v) else "✅ No drift detected"}
                        for k, v in drift_flags.items()]
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

    # =========================
    # Performance checks (compact tables)
    # =========================
    st.markdown("## 🧪 Performance Checks")

    def _style_bool(val: bool):
        return "background-color: #ff4d4d; color: white; font-weight: 600;" if val else \
               "background-color: #33cc33; color: white; font-weight: 600;"

    def _render_perf_table(perf_dict: dict, title: str):
        """Render a compact table for performance checks dict."""
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
            rows.append({
                "Metric": metric_name,
                "Value": value,
                "Threshold": payload.get("threshold"),
                "Drift": bool(payload.get("drift", False))
            })
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
            prev_v = payload.get("prev"); curr_v = payload.get("current")
            thr = payload.get("threshold")
            change = payload.get("relative_drop", payload.get("relative_increase"))
            drift_key = f"comparison::{metric.split('_')[0]}"
            comp_rows.append({
                "Metric": metric,
                "Previous": prev_v,
                "Current": curr_v,
                "Relative change": change if change is None else f"{float(change)*100:.2f}%",
                "Threshold": thr,
                "Drift": bool(drift_flags.get(drift_key, False))
            })
        if comp_rows:
            st.markdown("### 🔁 Previous vs Current (Relative change)")
            st.dataframe(pd.DataFrame(comp_rows).style.map(lambda v: _style_bool(v) if isinstance(v, bool) else "", subset=["Drift"]))

    # =========================
    # Multivariate tests (summary cards)
    # =========================
    if ("MMD" in tests) or ("Energy Distance" in tests):
        st.markdown("## 🧭 Multivariate Tests")
        cols = st.columns(2)
        if "MMD" in tests:
            m = tests["MMD"]
            with cols[0]:
                st.markdown("**MMD**")
                st.write(f"Statistic: `{m.get('statistic')}`")
                if m.get("p_value") is not None:
                    st.write(f"p-value: `{m.get('p_value')}`  (α={m.get('alpha')})")
                st.write(f"Kernel: `{m.get('kernel')}`")
                if m.get("bandwidth") is not None:
                    st.write(f"Bandwidth: `{m.get('bandwidth')}`")
                st.markdown("Drift: " + ("**⚠️ YES**" if bool(m.get("drift")) else "**✅ NO**"))
        if "Energy Distance" in tests:
            e = tests["Energy Distance"]
            with cols[1]:
                st.markdown("**Energy Distance**")
                st.write(f"Distance: `{e.get('distance')}`")
                if e.get("p_value") is not None:
                    st.write(f"p-value: `{e.get('p_value')}`  (α={e.get('alpha')})")
                st.markdown("Drift: " + ("**⚠️ YES**" if bool(e.get("drift")) else "**✅ NO**"))

    # =========================
    # Load data to plot distributions (if available)
    # =========================
    last_file = None
    df_current = None
    df_prev = None

    # previous_data.csv (reference used by tests)
    prev_path = os.path.join(control_dir, "previous_data.csv")
    if os.path.exists(prev_path):
        try:
            df_prev = pd.read_csv(prev_path)
        except Exception:
            df_prev = None

    # last processed dataset (current) - now with multi-format support
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

    # =========================
    # Feature Explorer (distributions & shapes)
    # =========================
    st.markdown("## 🔎 Feature Explorer")

    if df_prev is None or df_current is None:
        st.info("To render the distribution plots, I need both: `previous_data.csv` and the latest dataset. "
                "At least one is missing, so skipping interactive plots.")
        return

    # Keep only common numeric columns
    common_cols = [c for c in df_prev.columns if c in df_current.columns]
    num_cols = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(df_prev[c]) and pd.api.types.is_numeric_dtype(df_current[c])
    ]
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
    s_curr = df_current[col].dropna().astype(float)

    with left:
        st.markdown(f"### {col}")

    # 1) Histogram overlay
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=s_prev, name="Previous", opacity=0.55, nbinsx=bins,
                                    histnorm="probability" if show_norm else ""))
    hist_fig.add_trace(go.Histogram(x=s_curr, name="Current", opacity=0.55, nbinsx=bins,
                                    histnorm="probability" if show_norm else ""))
    hist_fig.update_layout(barmode="overlay", title=f"Histogram – {col}",
                           xaxis_title=col, yaxis_title="Density" if show_norm else "Count")
    if show_logx:
        hist_fig.update_xaxes(type="log")
    st.plotly_chart(hist_fig, use_container_width=True)

    # 2) ECDF overlay (KS-style visual)
    def _ecdf(x):
        x_sorted = np.sort(x)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    x1, y1 = _ecdf(s_prev.values)
    x2, y2 = _ecdf(s_curr.values)
    ecdf_fig = go.Figure()
    ecdf_fig.add_trace(go.Scatter(x=x1, y=y1, mode="lines", name="Previous"))
    ecdf_fig.add_trace(go.Scatter(x=x2, y=y2, mode="lines", name="Current"))
    ecdf_fig.update_layout(title=f"ECDF – {col}", xaxis_title=col, yaxis_title="Cumulative probability")
    if show_logx:
        ecdf_fig.update_xaxes(type="log")
    st.plotly_chart(ecdf_fig, use_container_width=True)

    # 3) Q–Q plot
    q = np.linspace(0.01, 0.99, 99)
    q_prev = np.quantile(s_prev, q)
    q_curr = np.quantile(s_curr, q)
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(x=q_prev, y=q_curr, mode="markers", name="Q–Q"))
    # y = x reference
    minv = float(np.nanmin([q_prev.min(), q_curr.min()]))
    maxv = float(np.nanmax([q_prev.max(), q_curr.max()]))
    qq_fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="y=x", line=dict(dash="dash")))
    qq_fig.update_layout(title=f"Q–Q Plot – {col}", xaxis_title="Previous quantiles", yaxis_title="Current quantiles")
    if show_logx:
        qq_fig.update_xaxes(type="log"); qq_fig.update_yaxes(type="log")
    st.plotly_chart(qq_fig, use_container_width=True)

    # 4) Violin + Box (side-by-side)
    vb_fig = go.Figure()
    vb_fig.add_trace(go.Violin(y=s_prev, name="Previous", box_visible=True, meanline_visible=True, points="all", jitter=0.1))
    vb_fig.add_trace(go.Violin(y=s_curr, name="Current", box_visible=True, meanline_visible=True, points="all", jitter=0.1))
    vb_fig.update_layout(title=f"Violin + Box – {col}", yaxis_title=col)
    st.plotly_chart(vb_fig, use_container_width=True)

    # =========================
    # Drift Heatmap (by feature)
    # =========================
    st.markdown("## 🗺️ Drift Heatmap")

    metric_options = []
    if "Kolmogorov-Smirnov" in tests: metric_options.append("KS p-value")
    if "PSI" in tests: metric_options.append("PSI")
    if "Hellinger Distance" in tests: metric_options.append("Hellinger")
    if "Earth Mover's Distance" in tests: metric_options.append("EMD")

    if not metric_options:
        st.info("No per-feature drift metrics found to build a heatmap.")
        return

    msel = st.selectbox("Heatmap metric", options=metric_options, index=0)

    feat_vals = {}
    if msel == "KS p-value" and "Kolmogorov-Smirnov" in tests:
        for feat, payload in tests["Kolmogorov-Smirnov"].items():
            val = payload.get("p_value")
            try:
                feat_vals[feat] = float(val) if val is not None else None
            except Exception:
                feat_vals[feat] = None

    elif msel == "PSI" and "PSI" in tests:
        for feat, payload in tests["PSI"].items():
            if "psi" in payload and payload["psi"] is not None:
                try:
                    feat_vals[feat] = float(payload["psi"])
                except Exception:
                    feat_vals[feat] = None
            else:
                psi_value = payload.get("psi_value")
                if isinstance(psi_value, dict):
                    exp = psi_value.get("expected"); act = psi_value.get("actual")
                    if isinstance(exp, list) and isinstance(act, list) and len(exp) == len(act) and len(exp) > 0:
                        try:
                            exp_arr = np.array(exp, dtype=float)
                            act_arr = np.array(act, dtype=float)
                            exp_arr = np.where(exp_arr == 0, 1e-8, exp_arr)
                            act_arr = np.where(act_arr == 0, 1e-8, act_arr)
                            psi_val = np.sum((act_arr - exp_arr) * np.log(act_arr / exp_arr))
                            feat_vals[feat] = float(psi_val)
                        except Exception:
                            feat_vals[feat] = None
                    else:
                        feat_vals[feat] = None
                else:
                    feat_vals[feat] = None

    elif msel == "Hellinger" and "Hellinger Distance" in tests:
        for feat, payload in tests["Hellinger Distance"].items():
            v = payload.get("hellinger_distance")
            feat_vals[feat] = float(v) if v is not None else None

    elif msel == "EMD" and "Earth Mover's Distance" in tests:
        for feat, payload in tests["Earth Mover's Distance"].items():
            v = payload.get("emd_distance")
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
            coloraxis="coloraxis"
        )
    )
    heat_fig.update_layout(
        title=f"Drift heatmap by feature – {msel}",
        xaxis=dict(title="Feature", tickangle=-45),
        yaxis=dict(title=""),
        coloraxis=dict(colorbar=dict(title=msel))
    )
    st.plotly_chart(heat_fig, use_container_width=True)


# =========================
# Train / Retrain Section
# =========================
def show_train_section(pipeline_name):
    """
    Displays training or retraining metrics.
    Expects JSON at: pipelines/{pipeline_name}/metrics/train_results.json
    """
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

    # ---- General info
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
        # Show strategy always if present (even if type was "train" por ejecuciones antiguas)
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

            # GridSearch details
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

            # Extra retrain details (shown only if present)
            extra_rows = []

            # Window size
            for k in ["window_size_used", "window_size"]:
                if k in results:
                    extra_rows.append(("Window size", results[k])); break

            # Calibration
            if "calibration" in results:
                extra_rows.append(("Calibration", results["calibration"]))

            # Replay mix
            if "replay_frac_old" in results:
                extra_rows.append(("Replay frac (old)", results["replay_frac_old"]))

            # Ensemble/meta info
            if "meta" in results:
                extra_rows.append(("Meta", results["meta"]))
            if "components" in results:
                extra_rows.append(("Ensemble components", results["components"]))

            # Misc training extras
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

    # ---- Metrics
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

    # ---- Errors / Raw JSON
    if "error" in results:
        st.error(f"Error: {results['error']}")

    with st.expander("🔎 Raw training JSON"):
        st.json(results)



# =========================
# Logs Section
# =========================
def parse_logs_to_df(log_text: str) -> pd.DataFrame:
    """
    Parse log lines into a structured DataFrame with columns:
    date, pipeline, level, message.
    If a line does not match the expected pattern, it is kept with message only.
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
    """
    Displays general, error, and warning logs for the given pipeline.
    Expects log files under: pipelines/{pipeline_name}/logs/
    """
    st.subheader("📜 Pipeline Logs")

    logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline.log")
    error_logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline_errors.log")
    warning_logs_path = os.path.join("pipelines", pipeline_name, "logs", "pipeline_warnings.log")

    # General Log
    st.markdown("### 📘 General Log")
    general_log_text = load_log(logs_path)
    df_general = parse_logs_to_df(general_log_text)
    if not df_general.empty:
        # Color the 'level' column to enhance readability
        st.dataframe(
            df_general.style.map(
                lambda v: "color: red;" if v == "ERROR" else "color: orange;" if v == "WARNING" else "color: green;",
                subset=["level"]
            )
        )
    else:
        st.info("No general logs found.")

    # Error Log
    st.markdown("### ❌ Error Log")
    error_log_text = load_log(error_logs_path)
    if error_log_text.strip():
        st.code(error_log_text, language="bash")
    else:
        st.success("✅ No errors found.")

    # Warning Log
    st.markdown("### ⚠️ Warning Log")
    warning_log_text = load_log(warning_logs_path)
    if warning_log_text.strip():
        st.code(warning_log_text, language="bash")
    else:
        st.success("✅ No warnings found.")


# =========================
# Main App (Page Layout)
# =========================

# Important: set page config as early as possible in Streamlit
st.set_page_config(page_title="Monitor ML Pipeline", layout="wide")
st.title("📌 Monitor")

# Support CLI args (streamlit passes its own args; we parse known ones)
parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_name", type=str, required=True)
args, _ = parser.parse_known_args()
pipeline_name = args.pipeline_name

# Load config.json for pipeline settings (name, data directory, etc.)
config_path = os.path.join("pipelines", pipeline_name, "config", "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    # Prefer config values if present
    pipeline_name = config.get("pipeline_name", pipeline_name)
    data_dir = config.get("data_dir")
else:
    st.error("config.json not found. Please start the monitor with a valid pipeline.")
    st.stop()

if not data_dir:
    st.error("`data_dir` missing in config.json.")
    st.stop()

# Tabbed interface
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
