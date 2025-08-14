# web_interface/dashboard_ipip.py
# -*- coding: utf-8 -*-
import os
import json
import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# =========================================================
# Helpers (I/O básicos — auto-contenido)
# =========================================================
SUPPORTED_EXT = (".csv", ".arff", ".json", ".xls", ".xlsx", ".parquet", ".txt")

def _load_any_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".json":
        return pd.read_json(path)
    elif suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".arff":
        from scipy.io import arff
        data, meta = arff.loadarff(path)
        return pd.DataFrame(data)
    elif suffix == ".txt":
        # intenta csv-like
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.read_fwf(path)
    else:
        raise ValueError(f"Formato no soportado: {suffix}")

def _read_json(path: str | Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _load_log_text(path: str | Path) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def _find_pipeline_dirs(pipeline_name: str) -> dict:
    base = Path("pipelines") / pipeline_name
    # cope with "models" / "modelos", "metrics" / "resultados"
    models_dir = base / "models"
    if not models_dir.exists():
        models_dir = base / "modelos"
    metrics_dir = base / "metrics"
    if not metrics_dir.exists():
        metrics_dir = base / "resultados"
    return {
        "base": base,
        "models": models_dir,
        "metrics": metrics_dir,
        "control": base / "control",
        "logs": base / "logs",
        "config": base / "config",
        "candidates": base / "candidates",
    }

def _dashboard_data_loader(data_dir: str, control_dir: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Devuelve (df, last_file) usando control/control_file.txt si existe,
    si no, toma el archivo más reciente por mtime.
    """
    last_file = None
    control_file = control_dir / "control_file.txt"
    if control_file.exists():
        try:
            with open(control_file, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if lines:
                last_file = lines[-1].split(",")[0]
        except Exception:
            last_file = None

    if last_file:
        path = Path(data_dir) / last_file
        if path.exists():
            return _load_any_dataset(path), last_file

    # fallback: más reciente
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return pd.DataFrame(), None
    cand = []
    for f in data_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
            try:
                cand.append((f, f.stat().st_mtime))
            except Exception:
                pass
    if not cand:
        return pd.DataFrame(), None
    cand.sort(key=lambda x: x[1], reverse=True)
    path = cand[0][0]
    try:
        return _load_any_dataset(path), path.name
    except Exception:
        return pd.DataFrame(), None

# =========================================================
# Helpers (Bloques)
# =========================================================
def _detect_block_col(pipeline_name: str, df: pd.DataFrame, default: str = "block_id") -> Optional[str]:
    # 1) config.json
    cfg_path = Path("pipelines") / pipeline_name / "config" / "config.json"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            col = cfg.get("block_col")
            if col and col in df.columns:
                return col
    except Exception:
        pass
    # 2) blocks_snapshot.json
    snap_path = Path("pipelines") / pipeline_name / "control" / "blocks_snapshot.json"
    try:
        if snap_path.exists():
            with open(snap_path, "r") as f:
                snap = json.load(f)
            control_file = Path("pipelines") / pipeline_name / "control" / "control_file.txt"
            file_key = None
            if control_file.exists():
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

def _sorted_blocks(series: pd.Series) -> List:
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

# =========================================================
# IPIP helpers (inspección del modelo)
# =========================================================
def _load_current_model(models_dir: Path, pipeline_name: str) -> Optional[object]:
    # el nombre normalmente es <pipeline>.pkl
    cand = models_dir / f"{pipeline_name}.pkl"
    if cand.exists():
        try:
            return joblib.load(cand)
        except Exception:
            return None
    # si no, busca el primer .pkl
    for f in models_dir.glob("*.pkl"):
        try:
            return joblib.load(f)
        except Exception:
            continue
    return None

def _summarize_ipip(model, feature_names: Optional[List[str]] = None) -> dict:
    """
    Devuelve:
      - p (ensembles), tamaños por ensemble, total modelos base
      - meta (prop_majoritaria, val_size, p,b,np_target)
      - base_estimator name (si es posible)
      - importancias agregadas (si base tiene feature_importances_)
    """
    summary = {
        "is_ipip": False,
        "p": None,
        "b": None,
        "ensemble_sizes": None,
        "total_base_models": None,
        "meta": {},
        "base_estimator": None,
        "feature_importances": None
    }
    if not hasattr(model, "ensembles_"):
        return summary

    ensembles = getattr(model, "ensembles_", None)
    if not isinstance(ensembles, list) or len(ensembles) == 0:
        return summary

    summary["is_ipip"] = True
    sizes = [len(Ek) for Ek in ensembles]
    summary["ensemble_sizes"] = sizes
    summary["p"] = len(ensembles)
    summary["b"] = max(sizes) if sizes else None
    summary["total_base_models"] = int(sum(sizes))

    meta = getattr(model, "meta_", {}) or {}
    summary["meta"] = meta

    # tratar de identificar el tipo de base_estimator
    try:
        base_name = None
        for Ek in ensembles:
            if Ek:
                base_name = Ek[0].__class__.__name__
                break
        summary["base_estimator"] = base_name
    except Exception:
        pass

    # importancias agregadas
    try:
        imps = []
        for Ek in ensembles:
            for m in Ek:
                if hasattr(m, "feature_importances_"):
                    imp = np.asarray(m.feature_importances_)
                    imps.append(imp)
        if imps and feature_names is not None:
            arr = np.vstack(imps)
            mean_imp = np.nanmean(arr, axis=0)
            df_imp = pd.DataFrame({"feature": feature_names, "importance": mean_imp})
            df_imp = df_imp.sort_values("importance", ascending=False)
            summary["feature_importances"] = df_imp
    except Exception:
        pass

    return summary

# =========================================================
# Secciones del dashboard
# =========================================================
def show_dataset_section(data_dir: str, pipeline_name: str):
    st.subheader("📊 Dataset Information")
    dirs = _find_pipeline_dirs(pipeline_name)
    df, last_file = _dashboard_data_loader(data_dir, dirs["control"])

    if df.empty or not last_file:
        st.warning("⚠️ No se encontró dataset procesado todavía.")
        return

    st.write(f"**Último dataset procesado:** `{last_file}`")

    block_col = _detect_block_col(pipeline_name, df)
    if block_col and block_col in df.columns:
        st.info(f"🔢 Columna de bloque detectada: **`{block_col}`**")
        counts = df[block_col].value_counts(dropna=False).sort_index()
        blocks = _sorted_blocks(df[block_col])
        st.markdown("### 🧱 Resumen de bloques")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(pd.DataFrame({"block": counts.index.astype(str), "rows": counts.values}))
        with c2:
            fig = px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": "Block", "y": "Rows"}, title="Filas por bloque")
            st.plotly_chart(fig, use_container_width=True)

        sel = st.selectbox("Bloque a visualizar", options=["(All)"] + [str(b) for b in blocks], index=0)
        if sel != "(All)":
            df_view = df[df[block_col].astype(str) == sel]
        else:
            df_view = df
    else:
        st.info("No se detectó columna de bloque. Mostrando información global.")
        df_view = df
        block_col = None

    st.markdown("### 👀 Preview (head)")
    st.dataframe(df_view.head(10))

    st.markdown("### 🗂️ Info")
    info_dict = {
        "Column": df_view.columns,
        "Non-Null Count": [df_view[col].notnull().sum() for col in df_view.columns],
        "Unique Values": [df_view[col].nunique(dropna=True) for col in df_view.columns],
        "Dtype": df_view.dtypes.values
    }
    st.dataframe(pd.DataFrame(info_dict))

    st.markdown(f"**Filas (vista):** {df_view.shape[0]} — **Columnas:** {df_view.shape[1]}")
    st.markdown("### 📈 Descriptivas (vista)")
    st.dataframe(df_view.describe(include="all").transpose())

    st.markdown("### 🔎 Top 5 en categóricas (vista)")
    cat_cols = df_view.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            st.markdown(f"**{col}**")
            freq = df_view[col].value_counts().head(5).reset_index()
            freq.columns = [col, "Frequency"]
            fig = px.bar(freq, x=col, y="Frequency", text="Frequency", title=f"Top 5 en {col}")
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-30, height=360)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin columnas categóricas en la vista actual.")

def show_evaluator_section(pipeline_name: str):
    st.subheader("🧪 Evaluación")
    dirs = _find_pipeline_dirs(pipeline_name)
    eval_path = dirs["metrics"] / "eval_results.json"
    health_path = dirs["metrics"] / "health.json"

    if not eval_path.exists():
        st.info("Aún no hay resultados de evaluación.")
        return

    results = _read_json(eval_path)
    if not results:
        st.warning("eval_results.json vacío o ilegible.")
        return

    st.markdown("## ✅ Estado")
    if results.get("approved", False):
        st.success("Modelo **APROBADO**: cumple umbrales.")
    else:
        st.error("Modelo **NO APROBADO**: no cumple umbrales.")

    blocks_info = results.get("blocks", {}) or {}
    if blocks_info:
        st.markdown("## 🧱 Bloques (evaluator)")
        cols = st.columns(3)
        cols[0].metric("Block column", str(blocks_info.get("block_col")))
        cols[1].metric("Bloque evaluado", str(blocks_info.get("evaluated_block_id")))
        ref_blocks = blocks_info.get("reference_blocks") or []
        cols[2].metric("Bloques referencia", len(ref_blocks))
        if ref_blocks:
            st.caption(f"Referencia: {', '.join(map(str, ref_blocks))}")

        per_blk = blocks_info.get("per_block_metrics") or {}
        if per_blk:
            st.markdown("### 📊 Métricas por bloque (test)")
            df_blk = pd.DataFrame.from_dict(per_blk, orient="index").reset_index().rename(columns={"index": "block"})
            st.dataframe(df_blk)
            # chart rápido
            for m in ["accuracy", "f1", "balanced_accuracy", "r2", "rmse"]:
                if m in df_blk.columns:
                    st.plotly_chart(px.bar(df_blk, x="block", y=m, title=f"{m} por bloque"), use_container_width=True)
                    break

    st.markdown("## 📏 Umbrales usados")
    thr = results.get("thresholds", {})
    if thr:
        st.table(pd.DataFrame(list(thr.items()), columns=["Métrica", "Umbral"]))
    else:
        st.info("Sin umbrales en resultados.")

    st.markdown("## 📊 Métricas (test)")
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
        st.info("No se encontraron métricas en los resultados.")

    st.markdown("## 🛑 Circuit Breaker")
    if health_path.exists():
        try:
            h = _read_json(health_path) or {}
            consecutive = int(h.get("consecutive_failures", 0) or 0)
            paused_until = h.get("paused_until")
            last_failure_ts = h.get("last_failure_ts")
            cols = st.columns(3)
            cols[0].metric("Consecutive Failures", consecutive)

            def _fmt(ts):
                try:
                    import datetime, time
                    return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return str(ts)

            cols[1].metric("Paused Until", _fmt(paused_until) if paused_until else "—")
            cols[2].metric("Last Failure", _fmt(last_failure_ts) if last_failure_ts else "—")

            import time as _t
            paused = bool(paused_until and _t.time() < float(paused_until))
            st.warning("⏸️ Reentrenos **pausados** por circuit breaker.") if paused else st.success("▶️ Reentrenos **activos**.")
        except Exception as e:
            st.warning(f"No se pudo leer health.json: {e}")
    else:
        st.info("Sin estado de circuit breaker todavía.")

def show_drift_section(pipeline_name: str):
    st.subheader("🌊 Drift")
    dirs = _find_pipeline_dirs(pipeline_name)
    metrics_dir = dirs["metrics"]
    control_dir = dirs["control"]
    config_path = dirs["config"] / "config.json"

    # Optional: timeline de bloques
    blocks_report_path = metrics_dir / "blocks_training_report.json"
    if blocks_report_path.exists():
        try:
            rpt = _read_json(blocks_report_path)
            st.markdown("## 🧭 Decisiones por bloque")
            if isinstance(rpt.get("decisions"), list) and rpt["decisions"]:
                df_dec = pd.DataFrame(rpt["decisions"])
                st.dataframe(df_dec)
                cnt = df_dec["decision"].value_counts()
                st.plotly_chart(px.bar(x=cnt.index, y=cnt.values, title="Conteo de decisiones"), use_container_width=True)
        except Exception as e:
            st.info(f"(opcional) No se pudo cargar blocks_training_report.json: {e}")

    drift_path = metrics_dir / "drift_results.json"
    if not drift_path.exists():
        st.info("Aún no hay drift_results.json.")
        return

    results = _read_json(drift_path) or {}
    decision = results.get("decision")
    if decision:
        if decision == "no_drift":
            st.success("🟢 Sin drift — se mantiene el modelo actual")
        elif decision == "previous_promoted":
            reason = results.get("promotion_reason")
            msg = "🔄 Se promocionó el modelo anterior"
            if reason: msg += f" (motivo: `{reason}`)"
            st.warning(msg)
        elif decision == "retrain":
            st.error("🛠 Drift detectado — reentrenar")
        elif decision == "train":
            st.info("🆕 Primera ejecución — entrenar desde cero")
        elif decision == "end_error":
            st.warning("⚠️ Error cargando modelo actual")
        else:
            st.info(f"ℹ️ Decisión: {decision}")

    tests = results.get("tests", {})
    flags = results.get("drift", {})

    if flags:
        st.markdown("## 📌 Resumen de drift")
        summary_data = [{"Test": k, "Result": "⚠️ Drift" if bool(v) else "✅ OK"} for k, v in flags.items()]
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
    else:
        st.info("Sin resultados de drift per-feature / multivar.")
        return

    # Feature explorer: reference vs bloque del último dataset
    prev_path = control_dir / "previous_data.csv"
    df_prev = pd.read_csv(prev_path) if prev_path.exists() else None

    data_dir = None
    if config_path.exists():
        cfg = _read_json(config_path) or {}
        data_dir = cfg.get("data_dir")

    df_curr = None; last_file = None
    if data_dir:
        ctrl_file = control_dir / "control_file.txt"
        if ctrl_file.exists():
            with open(ctrl_file, "r") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            if lines:
                last_file = lines[-1].split(",")[0]
                p = Path(data_dir) / last_file
                if p.exists():
                    df_curr = _load_any_dataset(p)

    st.markdown("## 🔎 Comparación de variables (Reference vs. Bloque)")
    if df_prev is None or df_curr is None:
        st.info("Necesito `previous_data.csv` y el último dataset cargado para comparar.")
        return

    block_col = _detect_block_col(pipeline_name, df_curr)
    if block_col and block_col in df_curr.columns:
        options = _sorted_blocks(df_curr[block_col])
        sel_block = st.selectbox("Bloque actual a comparar", options=[str(b) for b in options])
        df_view = df_curr[df_curr[block_col].astype(str) == sel_block]
    else:
        st.info("No hay columna de bloque — uso dataset completo como 'actual'.")
        df_view = df_curr

    common_cols = [c for c in df_prev.columns if c in df_view.columns]
    num_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_prev[c]) and pd.api.types.is_numeric_dtype(df_view[c])]
    if not num_cols:
        st.info("No hay columnas numéricas comunes para graficar.")
        return

    left, right = st.columns([2, 1])
    with right:
        col = st.selectbox("Variable", options=num_cols, index=0)
        bins = st.slider("Bins", min_value=10, max_value=100, value=40, step=5)
        show_logx = st.checkbox("Log X-axis", value=False)
        show_norm = st.checkbox("Normalizar histogramas", value=True)

    s_prev = df_prev[col].dropna().astype(float)
    s_curr = df_view[col].dropna().astype(float)
    with left:
        st.markdown(f"### {col}")

    # Hist
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=s_prev, name="Reference", opacity=0.55, nbinsx=bins,
                                    histnorm="probability" if show_norm else ""))
    hist_fig.add_trace(go.Histogram(x=s_curr, name="Bloque", opacity=0.55, nbinsx=bins,
                                    histnorm="probability" if show_norm else ""))
    hist_fig.update_layout(barmode="overlay", title=f"Histograma – {col}",
                           xaxis_title=col, yaxis_title="Densidad" if show_norm else "Cuenta")
    if show_logx: hist_fig.update_xaxes(type="log")
    st.plotly_chart(hist_fig, use_container_width=True)

    # ECDF
    def _ecdf(x):
        x_sorted = np.sort(x)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y
    x1, y1 = _ecdf(s_prev.values); x2, y2 = _ecdf(s_curr.values)
    ecdf_fig = go.Figure()
    ecdf_fig.add_trace(go.Scatter(x=x1, y=y1, mode="lines", name="Reference"))
    ecdf_fig.add_trace(go.Scatter(x=x2, y=y2, mode="lines", name="Bloque"))
    ecdf_fig.update_layout(title=f"ECDF – {col}", xaxis_title=col, yaxis_title="Prob. acumulada")
    if show_logx: ecdf_fig.update_xaxes(type="log")
    st.plotly_chart(ecdf_fig, use_container_width=True)

def show_train_section(pipeline_name: str):
    st.subheader("🤖 Train / Retrain (IPIP)")
    dirs = _find_pipeline_dirs(pipeline_name)
    train_path = dirs["metrics"] / "train_results.json"

    if not train_path.exists():
        st.info("Aún no hay resultados de entrenamiento.")
        return

    results = _read_json(train_path) or {}
    if not results:
        st.warning("train_results.json vacío.")
        return

    MODE_LABELS = {
        0: "Full retraining",
        1: "Incremental (partial_fit)",
        2: "Windowed (rolling)",
        3: "Ensemble old + new",
        4: "Stacking old + cloned(old)",
        5: "Replay mix (prev + current)",
        6: "Recalibration (IPIP → full)",
    }

    st.markdown("## 📌 Información")
    cols = st.columns(2)
    with cols[0]:
        st.write(f"**Tipo:** {results.get('type', results.get('tipo', '-'))}")
        st.write(f"**Fichero:** {results.get('file', results.get('archivo', '-'))}")
        st.write(f"**Fecha:** {results.get('timestamp', '-')}")
    with cols[1]:
        st.write(f"**Modelo:** {results.get('model', results.get('modelo', '-'))}")
        if "mode" in results:
            label = MODE_LABELS.get(results['mode'], f"Unknown ({results['mode']})")
            st.write(f"**Mode:** {results['mode']} — {label}")
        if "strategy" in results:
            st.write(f"**Strategy:** {results['strategy']}")

    st.markdown("## 📊 Métricas")
    if "classification_report" in results:
        bal = results.get("balanced_accuracy", results.get("balanced_accuracy_score"))
        try:
            st.write(f"**Balanced Accuracy:** {float(bal):.4f}")
        except Exception:
            st.write(f"**Balanced Accuracy:** {bal}")
        st.write(f"**Accuracy:** {results.get('accuracy', '-')}")
        st.write(f"**F1 (macro):** {results.get('f1_macro', results.get('f1', '-'))}")
        rep = results["classification_report"]
        df_rep = pd.DataFrame(rep)
        if set(["precision", "recall", "f1-score", "support"]).issubset(df_rep.index):
            st.dataframe(df_rep.transpose())
        else:
            st.dataframe(df_rep)
    elif all(k in results for k in ["r2", "rmse", "mae", "mse"]):
        st.write(f"**R2:** {results['r2']}")
        st.write(f"**RMSE:** {results['rmse']}")
        st.write(f"**MAE:** {results['mae']}")
        st.write(f"**MSE:** {results['mse']}")
    else:
        st.warning("No se encontraron métricas esperadas.")

    st.markdown("## ⚙️ IPIP meta")
    meta = results.get("meta") or {}
    if meta:
        df_meta = pd.DataFrame([meta]).transpose().rename(columns={0: "value"})
        st.table(df_meta)
    else:
        st.info("Sin metadatos IPIP en train_results.json.")

    with st.expander("🔎 JSON bruto de entrenamiento"):
        st.json(results)

def show_ipip_section(pipeline_name: str):
    st.subheader("🧩 IPIP — Estructura del ensemble")
    dirs = _find_pipeline_dirs(pipeline_name)
    model = _load_current_model(dirs["models"], pipeline_name)

    if model is None:
        st.info("No se pudo cargar el modelo actual.")
        return

    # Intenta obtener nombres de features desde previous_data o desde último dataset
    feat_names = None
    prev_path = dirs["control"] / "previous_data.csv"
    if prev_path.exists():
        try:
            df_prev = pd.read_csv(prev_path)
            # quita target si está (última heurística)
            feat_names = [c for c in df_prev.columns]
        except Exception:
            pass

    summary = _summarize_ipip(model, feature_names=feat_names)
    if not summary["is_ipip"]:
        st.info("El modelo cargado no parece un IPIPClassifier.")
        return

    cols = st.columns(4)
    cols[0].metric("Ensembles (p)", summary["p"])
    cols[1].metric("Max models/ensemble (b)", summary["b"])
    cols[2].metric("Modelos base", summary["total_base_models"])
    cols[3].metric("Base estimator", summary["base_estimator"] or "—")

    st.markdown("## 📦 Tamaño por ensemble")
    sizes = pd.DataFrame({"ensemble": list(range(1, summary["p"] + 1)), "n_models": summary["ensemble_sizes"]})
    st.dataframe(sizes)
    st.plotly_chart(px.bar(sizes, x="ensemble", y="n_models", title="Modelos por ensemble"), use_container_width=True)

    st.markdown("## ⚙️ Metadatos IPIP")
    st.table(pd.DataFrame([summary["meta"]]).transpose().rename(columns={0: "value"}))

    if isinstance(summary.get("feature_importances"), pd.DataFrame) and not summary["feature_importances"].empty:
        st.markdown("## 🌟 Feature Importances (agregado)")
        topk = st.slider("Top K", min_value=5, max_value=50, value=20, step=5)
        df_imp = summary["feature_importances"].head(topk)
        st.dataframe(df_imp)
        st.plotly_chart(px.bar(df_imp, x="feature", y="importance", title=f"Top {topk} importances"), use_container_width=True)
    else:
        st.info("La base no expone `feature_importances_` o no fue posible agregarlas.")

def parse_logs_to_df(log_text: str) -> pd.DataFrame:
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

def show_logs_section(pipeline_name: str):
    st.subheader("📜 Logs")
    dirs = _find_pipeline_dirs(pipeline_name)
    logs_path = dirs["logs"] / "pipeline.log"
    error_logs_path = dirs["logs"] / "pipeline_errors.log"
    warning_logs_path = dirs["logs"] / "pipeline_warnings.log"

    st.markdown("### 📘 General")
    general_log_text = _load_log_text(logs_path)
    df_general = parse_logs_to_df(general_log_text)
    if not df_general.empty:
        st.dataframe(
            df_general.style.map(
                lambda v: "color: red;" if v == "ERROR" else "color: orange;" if v == "WARNING" else "color: green;",
                subset=["level"]
            )
        )
    else:
        st.info("Sin logs generales.")

    st.markdown("### ❌ Errores")
    error_log_text = _load_log_text(error_logs_path)
    if error_log_text.strip():
        st.code(error_log_text, language="bash")
    else:
        st.success("✅ Sin errores.")

    st.markdown("### ⚠️ Warnings")
    warning_log_text = _load_log_text(warning_logs_path)
    if warning_log_text.strip():
        st.code(warning_log_text, language="bash")
    else:
        st.success("✅ Sin warnings.")

# =========================================================
# Main App
# =========================================================
st.set_page_config(page_title="Dashboard IPIP", layout="wide")
st.title("📌 Dashboard — IPIP")

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_name", type=str, required=True)
args, _ = parser.parse_known_args()
pipeline_name = args.pipeline_name

cfg_path = Path("pipelines") / pipeline_name / "config" / "config.json"
if cfg_path.exists():
    cfg = _read_json(cfg_path) or {}
    pipeline_name = cfg.get("pipeline_name", pipeline_name)
    data_dir = cfg.get("data_dir")
else:
    st.error("config.json no encontrado. Inicia el monitor con una pipeline válida.")
    st.stop()

if not data_dir:
    st.error("Falta `data_dir` en config.json.")
    st.stop()

tabs = st.tabs(["📊 Dataset", "🧪 Evaluator", "🌊 Drift", "🤖 Train/Retrain", "🧩 IPIP", "📜 Logs"])

with tabs[0]:
    show_dataset_section(data_dir, pipeline_name)

with tabs[1]:
    show_evaluator_section(pipeline_name)

with tabs[2]:
    show_drift_section(pipeline_name)

with tabs[3]:
    show_train_section(pipeline_name)

with tabs[4]:
    show_ipip_section(pipeline_name)

with tabs[5]:
    show_logs_section(pipeline_name)
