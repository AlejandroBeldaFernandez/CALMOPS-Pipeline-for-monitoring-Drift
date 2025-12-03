# pipeline/preprocessing_ipip.py
# -*- coding: utf-8 -*-
"""
Simple preprocessing for IPIP.
- Assumes that the target column is called "target" (or the one you pass in target_col).
- Returns X (all columns except the target) and y (the target column).
- Maintains numeric types, tries to convert objects to numeric when it makes sense,
  imputes numerics by median and factorizes categoricals (-1 for missing).
- DOES NOT need to know the block column; the pipeline will reattach it later.
"""

from __future__ import annotations

from typing import Optional
import re

import numpy as np
import pandas as pd

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_LEAK_COLS = {"id", "uuid"}
_BIN_TRUE = {"si", "sí", "yes", "y", "true", "1"}
_BIN_FALSE = {"no", "n", "false", "0"}


def _decode_bytes_col(s: pd.Series) -> pd.Series:
    if s.dtype == object and s.apply(lambda v: isinstance(v, (bytes, bytearray))).any():
        return s.apply(
            lambda v: v.decode("utf-8", errors="ignore")
            if isinstance(v, (bytes, bytearray))
            else v
        )
    return s


def _maybe_numeric_from_object(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    as_num = pd.to_numeric(s, errors="coerce")
    if as_num.notna().mean() >= 0.5:
        return as_num

    def _extract_first_num(x: Optional[str]) -> Optional[float]:
        if x is None or pd.isna(x):
            return np.nan
        m = _NUM_RE.search(str(x))
        return float(m.group()) if m else np.nan

    extracted = s.map(_extract_first_num)
    if extracted.notna().mean() >= 0.5:
        return extracted

    return series


def _factorize_categorical(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("__NA__")
    codes, _ = pd.factorize(s, sort=True)
    codes = pd.Series(codes, index=s.index)
    codes[s.eq("__NA__")] = -1
    return codes.astype("int32")


def _normalize_binary_text(y: pd.Series) -> Optional[pd.Series]:
    if not (y.dtype == object or str(y.dtype).startswith("string")):
        return None
    s = y.astype("string").str.strip().str.lower()
    uniq = set(s.dropna().unique().tolist())

    if len(uniq) == 2 and uniq <= (_BIN_TRUE | _BIN_FALSE):
        mapped = s.map(
            lambda v: 1 if v in _BIN_TRUE else (0 if v in _BIN_FALSE else np.nan)
        )
        return mapped.astype("Int64").fillna(0).astype("int32")

    if uniq == {"si", "no"} or uniq == {"sí", "no"}:
        return s.map({"no": 0, "si": 1, "sí": 1}).astype("int32")

    return None


def data_preprocessing(
    df: pd.DataFrame, target_col: str = "class"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Complete data (includes the 'target' column).
    target_col : str, default="target"
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Preprocessed features (does not include the target column).
    y : pd.Series
        Target vector (int32 if binary/categorical; numeric otherwise).
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype="int32", name=target_col)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    df = df.copy()

    # Decode bytes in all columns
    for col in df.columns:
        df[col] = _decode_bytes_col(df[col])

    # Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Remove obvious leak columns (we DO NOT remove possible block columns)
    leak_cols_present = [c for c in X.columns if c.lower() in _LEAK_COLS]
    if leak_cols_present:
        X = X.drop(columns=leak_cols_present, errors="ignore")

    # Convert object -> numeric when it makes sense
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        X[col] = _maybe_numeric_from_object(X[col])

    # Recalculate types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Numerics: coerce + median (fallback 0)
    if num_cols:
        X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
        for col in num_cols:
            med = X[col].median()
            if pd.isna(med):
                med = 0.0
            X[col] = X[col].fillna(med)

    # Categoricals: factorize; -1 for missing
    for col in cat_cols:
        X[col] = _factorize_categorical(X[col])

    # Final types
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype("int32")
        else:
            X[col] = X[col].astype("float32")

    # Target to binary/categorical integer if applicable
    y_norm = _normalize_binary_text(y)
    if y_norm is not None:
        y = y_norm
    else:
        if y.dtype == object or str(y.dtype).startswith("string"):
            codes, _ = pd.factorize(y.astype("string"), sort=True)
            y = pd.Series(codes, index=y.index, name=target_col).astype("int32")
        else:
            y = pd.to_numeric(y, errors="coerce")
            valid = y.notna()
            X = X.loc[valid]
            y = y.loc[valid].astype(
                y.dtype if np.issubdtype(y.dtype, np.integer) else "float32"
            )
            y.name = target_col

    # Align indices
    X = X.loc[y.index]

    return X, y
