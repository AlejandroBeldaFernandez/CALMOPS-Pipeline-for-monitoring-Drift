import os
import json
import re
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

from calmops.utils import get_pipelines_root
from utils import _load_any_dataset

# =========================
# Shared Constants
# =========================
project_root = get_pipelines_root()

# =========================
# Anti-delta sanitizers
# =========================
_DELTA_KEY_RE = re.compile(r"(?:\bdelta\b|Δ|δ|∆)", flags=re.IGNORECASE)
_DELTA_TOKEN_RE = re.compile(
    r"(?:\bdelta\b\s*:?\s*|Δ\s*:?\s*|δ\s*:?\s*|∆\s*:?\s*)", flags=re.IGNORECASE
)


def _sanitize_text(val: Any) -> str:
    try:
        s = str(val)
    except Exception:
        return ""
    s = _DELTA_TOKEN_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _contains_delta_token(name: str) -> bool:
    try:
        return bool(_DELTA_KEY_RE.search(str(name)))
    except Exception:
        return False


def _drop_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if _contains_delta_token(c)]
    if drop:
        df = df.drop(columns=drop, errors="ignore")
    return df


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = _drop_delta_columns(df)
    df = df.rename(columns=lambda c: _sanitize_text(c))
    try:
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.set_names([_sanitize_text(n) for n in df.index.names])
        else:
            df.index = df.index.rename(_sanitize_text(df.index.name))
    except Exception:
        pass
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            try:
                df[c] = df[c].astype(str).map(_sanitize_text)
            except Exception:
                pass
    return df


def _sanitize_figure(fig: go.Figure) -> go.Figure:
    if fig is None:
        return fig
    try:
        if fig.layout.title and fig.layout.title.text:
            fig.layout.title.text = _sanitize_text(fig.layout.title.text)
    except Exception:
        pass
    # Sanitize axes
    for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"]:
        try:
            axis = getattr(fig.layout, ax)
            if axis and axis.title and axis.title.text:
                axis.title.text = _sanitize_text(axis.title.text)
        except Exception:
            pass
    # Sanitize legend
    try:
        if (
            fig.layout.legend
            and fig.layout.legend.title
            and fig.layout.legend.title.text
        ):
            fig.layout.legend.title.text = _sanitize_text(fig.layout.legend.title.text)
    except Exception:
        pass
    # Sanitize colorbar
    try:
        if (
            hasattr(fig.layout, "coloraxis")
            and fig.layout.coloraxis
            and fig.layout.coloraxis.colorbar
        ):
            cb = fig.layout.coloraxis.colorbar
            if cb.title and cb.title.text:
                cb.title.text = _sanitize_text(cb.title.text)
    except Exception:
        pass
    # Sanitize data traces
    try:
        for tr in fig.data:
            if hasattr(tr, "name") and tr.name:
                tr.name = _sanitize_text(tr.name)
    except Exception:
        pass
    return fig


# =========================
# Safe Rendering Wrappers
# =========================
def _safe_table(df: pd.DataFrame, *, use_container_width: bool = True):
    st.dataframe(_sanitize_df(df), use_container_width=use_container_width)


def _safe_table_static(df: pd.DataFrame):
    st.table(_sanitize_df(df))


def _safe_markdown(text: str):
    st.markdown(_sanitize_text(text))


def _safe_write(text: str):
    st.write(_sanitize_text(text))


def _safe_caption(text: str):
    st.caption(_sanitize_text(text))


def _safe_plot(fig: go.Figure, *, use_container_width: bool = True):
    st.plotly_chart(_sanitize_figure(fig), use_container_width=use_container_width)


def _safe_json_display(obj: Any):
    def _json_sanitize(o: Any):
        if isinstance(o, dict):
            return {
                _sanitize_text(k): _json_sanitize(v)
                for k, v in o.items()
                if not _contains_delta_token(k)
            }
        elif isinstance(o, list):
            return [_json_sanitize(x) for x in o]
        elif isinstance(o, str):
            return _sanitize_text(o)
        return o

    st.json(_json_sanitize(obj))


# =========================
# Caching & Loading Helpers
# =========================
def _mtime(path: str | Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str | Path, stamp: float) -> Optional[dict]:
    try:
        with open(Path(path), "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_text_cached(path: str | Path, stamp: float) -> str:
    try:
        with open(Path(path), "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str | Path, stamp: float) -> pd.DataFrame:
    return pd.read_csv(Path(path))


@st.cache_data(show_spinner=False)
def _load_any_dataset_cached(path: str | Path, stamp: float) -> pd.DataFrame:
    return _load_any_dataset(Path(path))


@st.cache_data(show_spinner=False)
def _load_joblib_cached(path: str | Path, stamp: float):
    return joblib.load(Path(path))


# =========================
# Statistics Helpers
# =========================
def _sorted_blocks(series: pd.Series):
    vals = series.dropna().unique().tolist()
    try:
        nums = [float(v) for v in vals]
        return [x for _, x in sorted(zip(nums, vals))]
    except Exception:
        pass
    try:
        dt = pd.to_datetime(vals, errors="raise")
        return [x for _, x in sorted(zip(dt, vals))]
    except Exception:
        pass
    return sorted(vals, key=lambda x: str(x))


def _sample_series(values: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    n = values.shape[0]
    if n <= max_points:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return values[idx]


def _ecdf_quantile_curve(
    values: np.ndarray, q_points: int = 512, logx: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate empirical cumulative distribution function curve."""
    q = np.linspace(0.0, 1.0, q_points, endpoint=True)
    x = np.quantile(values, q)
    y = q
    if logx:
        x = np.where(x <= 0, np.nan, x)
        mask = ~np.isnan(x)
        return x[mask], y[mask]
    return x, y


def _paired_hist(
    prev: np.ndarray, curr: np.ndarray, bins: int, density: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired histograms for drift detection visualization."""
    if prev.size == 0 or curr.size == 0:
        return np.array([]), np.array([]), np.array([])
    vmin = float(min(np.nanmin(prev), np.nanmin(curr)))
    vmax = float(max(np.nanmax(prev), np.nanmax(curr)))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
        eps = (
            1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax)
            else max(1e-9, abs(vmin) * 0.01 or 1.0)
        )
        vmin, vmax = (
            (0.0 - eps, 0.0 + eps)
            if not np.isfinite(vmin) or not np.isfinite(vmax)
            else (vmin - eps, vmax + eps)
        )
    hist_prev, edges = np.histogram(
        prev, bins=bins, range=(vmin, vmax), density=density
    )
    hist_curr, _ = np.histogram(curr, bins=bins, range=(vmin, vmax), density=density)
    hist_prev, hist_curr, centers = 0.5 * (edges[:-1] + edges[1:])
    return hist_prev, hist_curr, centers


def _get_pipeline_base_dir(pipeline_name: str) -> Path:
    return project_root / "pipelines" / pipeline_name


def _detect_block_col(
    pipeline_name: str, df: pd.DataFrame, default: str = "block_id"
) -> str | None:
    cfg_path = _get_pipeline_base_dir(pipeline_name) / "config" / "config.json"
    try:
        if cfg_path.exists():
            cfg = _read_json_cached(str(cfg_path), _mtime(cfg_path)) or {}
            if cfg.get("block_col") and cfg["block_col"] in df.columns:
                return cfg["block_col"]
    except Exception:
        pass
    if default in df.columns:
        return default
    for c in df.columns:
        if "block" in c.lower():
            return c
    return None
