# web_interface/dashboard_ipip.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.get_logger().setLevel('ERROR')
st.set_page_config(page_title="IPIP Dashboard", layout="wide")
st.title("Dashboard — IPIP (embedded approval in Train)")

SUPPORTED_EXT = (".csv", ".arff", ".json", ".xls", ".xlsx", ".parquet", ".txt")


# ------------------------ caches ------------------------
@st.cache_data(show_spinner=False)
def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _read_text(path: str) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    sfx = p.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(p)
    if sfx == ".json":
        return pd.read_json(p)
    if sfx in (".xls", ".xlsx"):
        return pd.read_excel(p)
    if sfx == ".parquet":
        return pd.read_parquet(p)
    if sfx == ".arff":
        from scipy.io import arff
        data, _ = arff.loadarff(p)
        df_ = pd.DataFrame(data)
        for c in df_.select_dtypes(include="object").columns:
            try:
                df_[c] = df_[c].apply(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
            except Exception:
                pass
        return df_
    if sfx == ".txt":
        try:
            return pd.read_csv(p, engine="python")
        except Exception:
            return pd.read_fwf(p)
    raise ValueError(f"Unsupported format: {sfx}")

@st.cache_data(show_spinner=False)
def _load_joblib(path: str):
    return joblib.load(path)


# ------------------------ paths ------------------------
def _dirs(pipeline_name: str) -> Dict[str, Path]:
    base = Path("pipelines") / pipeline_name
    models_dir = base / "modelos"
    if not models_dir.exists():
        models_dir = base / "models"
    return {
        "base": base,
        "models": models_dir,
        "metrics": base / "metrics",
        "logs": base / "logs",
        "control": base / "control",
        "config": base / "config",
    }

def _detect_block_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        low = c.lower()
        if low in ("chunk", "mes", "month", "period", "block", "block_id"):
            return c
        if "block" in low:
            return c
    return None


# ------------------------ helpers for train json ------------------------
def _derive_approval(jr: dict) -> Dict[str, Optional[float]]:
    """
    Devuelve dict con: metric, threshold, achieved, approved
    Compat: usa jr['approval'] si trae detalles; si no, cae a jr['global'].
    """
    approval = jr.get("approval") or {}
    metric = approval.get("metric")
    thr = approval.get("threshold")
    ach = approval.get("achieved")
    approved = bool(approval.get("approved", False))

    g = jr.get("global") or {}
    if metric is None:
        metric = g.get("metric_used")
    if thr is None:
        thr = g.get("threshold")
    if ach is None:
        # Si no hay metric_value, intentar tomar de means según metric_used
        ach = g.get("metric_value")
        if ach is None and g.get("means") and metric:
            mkey = {"BA": "BA", "ACC": "ACC", "F1m": "F1m"}.get(str(metric), None)
            if mkey and mkey in g["means"]:
                ach = g["means"][mkey]

    return {"metric": metric, "threshold": thr, "achieved": ach, "approved": approved}

def _per_block_from_transitions(jr: dict) -> pd.DataFrame:
    """
    Si no existe jr['per_block'], construye per-bloque a partir de transitions,
    interpretando que las métricas son del bloque 'to'.
    """
    if isinstance(jr.get("per_block"), list) and jr["per_block"]:
        return pd.DataFrame(jr["per_block"])

    rows: List[Dict] = []
    for tr in jr.get("transitions", []) or []:
        bj = tr.get("to") or tr.get("block")
        mets = tr.get("metrics") or {}
        if bj is None or not mets:
            continue
        rows.append({
            "block": str(bj),
            "BA": mets.get("BA"),
            "ACC": mets.get("ACC"),
            "F1m": mets.get("F1m"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # por si hay múltiples entradas por bloque, hacer media
    df = df.groupby("block", as_index=False).mean(numeric_only=True)
    return df


# ------------------------ sections ------------------------
def show_train_section(pipeline_name: str):
    st.subheader("🤖 Train / Retrain")
    d = _dirs(pipeline_name)
    tr_path = d["metrics"] / "train_results.json"
    rt_path = d["metrics"] / "retrain_results.json"

    # Mostrar train si existe; si no, retrain; si no, mensaje.
    use_path = tr_path if tr_path.exists() else (rt_path if rt_path.exists() else None)
    if use_path is None:
        st.info("No hay todavía resultados de train/retrain en `metrics/`.")
        return

    jr = _read_json(str(use_path)) or {}
    cols = st.columns(3)
    cols[0].write(f"**Tipo:** {jr.get('type') or jr.get('mode') or '—'}")
    cols[1].write(f"**Fichero:** {jr.get('file','—')}")
    cols[2].write(f"**Timestamp:** {jr.get('timestamp') or jr.get('time') or '—'}")

    # Aprobación
    st.markdown("### Approval")
    appr = _derive_approval(jr)
    metric, thr, ach, ok = appr["metric"], appr["threshold"], appr["achieved"], bool(appr["approved"])
    if metric and thr is not None and ach is not None:
        if ok:
            st.success(f"APROBADO — {metric}: {ach:.4f} ≥ {thr}")
        else:
            st.error(f"NO APROBADO — {metric}: {ach:.4f} < {thr}")
    else:
        st.warning("Sin datos suficientes de aprobación.")

    # Per-block (eval t→t+1)
    st.markdown("### Metrics per block (eval from t→t+1)")
    dfb = _per_block_from_transitions(jr)
    if not dfb.empty:
        st.dataframe(dfb, use_container_width=True)
        for m in ["BA", "ACC", "F1m"]:
            if m in dfb.columns:
                st.plotly_chart(px.bar(dfb, x="block", y=m, title=f"{m} por bloque"), use_container_width=True)
    else:
        st.info("No hay métricas por bloque.")

    # Global
    st.markdown("### Global")
    g = jr.get("global", {})
    if g:
        # Formateo amigable
        to_show = {
            "metric_used": g.get("metric_used"),
            "threshold": g.get("threshold"),
            "metric_value": g.get("metric_value"),
            **{f"mean_{k}": v for k, v in (g.get("means") or {}).items()}
        }
        st.table(pd.DataFrame([to_show]))
    else:
        st.info("Sin métricas globales.")

    # Meta IPIP / extras
    st.markdown("### Metadata")
    meta = jr.get("meta", {}) or {}
    if meta:
        st.table(pd.DataFrame([meta]).T.rename(columns={0: "value"}))
    else:
        st.info("Sin metadatos.")

    with st.expander("🔎 JSON crudo"):
        st.json(jr)


def show_drift_section(pipeline_name: str):
    st.subheader("🌊 Drift")
    d = _dirs(pipeline_name)
    p = d["metrics"] / "drift_results.json"
    if not p.exists():
        st.info("Aún no hay `drift_results.json`.")
        return
    jr = _read_json(str(p)) or {}
    decision = jr.get("decision")
    if decision == "no_drift":
        st.success("🟢 Sin drift")
    elif decision == "retrain":
        st.warning("Drift detected — retrain")
    elif decision == "train":
        st.info("🆕 Primer run — train")
    else:
        st.info(f"Decisión: {decision or '—'}")

    st.markdown("### Bloques con drift")
    blocks = jr.get("drifted_blocks")
    if not blocks:
        blocks = (jr.get("blockwise", {}) or {}).get("drifted_blocks_stats", [])
    if blocks:
        st.dataframe(pd.DataFrame({"block": [str(b) for b in blocks]}), use_container_width=True)
    else:
        st.info("No hay bloques marcados con drift.")


def show_ipip_section(pipeline_name: str):
    st.subheader("🧩 IPIP — Modelo actual")
    d = _dirs(pipeline_name)
    cand = d["models"] / f"{pipeline_name}.pkl"
    if not cand.exists():
        st.info("No hay modelo persistido todavía.")
        return
    try:
        model = _load_joblib(str(cand))
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        return

    if hasattr(model, "ensembles_"):
        sizes = [len(Ek) for Ek in getattr(model, "ensembles_", [])]
        cols = st.columns(4)
        cols[0].metric("Ensembles (p)", len(sizes))
        cols[1].metric("Max modelos/ensemble (b)", max(sizes) if sizes else 0)
        cols[2].metric("Modelos base (total)", sum(sizes))
        cols[3].metric("Base", getattr(model, "meta_", {}).get("base_estimator", "—"))
        df = pd.DataFrame({"ensemble": list(range(1, len(sizes) + 1)), "n_models": sizes})
        st.dataframe(df, use_container_width=True)
        if not df.empty:
            st.plotly_chart(px.bar(df, x="ensemble", y="n_models", title="Tamaño por ensemble"), use_container_width=True)
    else:
        st.info("El modelo cargado no expone `ensembles_` (no parece IPIP).")


def show_logs_section(pipeline_name: str):
    st.subheader("📜 Logs")
    d = _dirs(pipeline_name)
    log = _read_text(str(d["logs"] / "pipeline.log"))
    if log.strip():
        st.code(log[-50_000:], language="bash")
    else:
        st.info("Sin logs.")


# ------------------------ main ------------------------
cfg_path = Path(st.query_params.get("cfg", [None])[0] or (Path.cwd() / "pipelines" / "my_pipeline_ipip" / "config" / "config.json"))
if not cfg_path.exists():
    st.error("No se encontró `config.json`. Lanza el monitor con una pipeline válida.")
    st.stop()

cfg = _read_json(str(cfg_path)) or {}
pipeline_name = cfg.get("pipeline_name") or "my_pipeline_ipip"

tabs = st.tabs(["🤖 Train/Retrain", "🌊 Drift", "🧩 IPIP", "📜 Logs"])
# =========================
# Auto-refresh on control_file change
# =========================
control_file = os.path.join("pipelines", pipeline_name, "control", "control_file.txt")

# Inicializa timestamp en session_state si no existe
if "control_file_mtime" not in st.session_state:
    st.session_state.control_file_mtime = os.path.getmtime(control_file) if os.path.exists(control_file) else 0.0

# Revisa si ha cambiado
current_mtime = os.path.getmtime(control_file) if os.path.exists(control_file) else 0.0
if current_mtime != st.session_state.control_file_mtime:
    st.session_state.control_file_mtime = current_mtime
    st.experimental_rerun()
with tabs[0]:
    show_train_section(pipeline_name)
with tabs[1]:
    show_drift_section(pipeline_name)
with tabs[2]:
    show_ipip_section(pipeline_name)
with tabs[3]:
    show_logs_section(pipeline_name)
