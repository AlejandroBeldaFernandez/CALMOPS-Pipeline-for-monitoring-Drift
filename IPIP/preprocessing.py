# pipeline/preprocessing_ipip.py
# -*- coding: utf-8 -*-
"""
Preprocesamiento básico (IPIP-friendly)

- Target:
    * Si existe 'hosp' -> se usa como y.
    * Si no, si existe 'class' -> se usa como y.
    * Si no, la última columna del DataFrame.
  Se normalizan etiquetas binarias comunes:
    - "NO"/"SI" -> 0/1
    - "no"/"sí"/"si"/"yes"/"true"/"false" -> 0/1 (robusto)
  Si es texto multiclase, se factoriza (0..k-1).

- X:
    * Elimina columnas de bloque/ID evidentes: {'mes','block','block_id','id','uuid'} si existen (para evitar fuga).
    * Decodifica bytes -> str donde aplique.
    * Intenta convertir strings numéricos (o mixtos con dígitos) a numérico.
      - Si más del 50% de la columna se convierte a número, se mantiene la conversión.
      - Soporta casos como "zip_28001" -> 28001 (extrae el primer número).
    * Imputa medianas en numéricas.
    * Categóricas -> factoriza a enteros e imputa con -1 el NA.

- Salida:
    * X: DataFrame solo numérico (int32/float32).
    * y: Series (int32), con nombre del target original si estaba definido.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Iterable, Optional


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

_LEAK_COLS = {"mes", "block", "block_id", "id", "uuid"}

_BIN_TRUE = {"si", "sí", "yes", "y", "true", "1"}
_BIN_FALSE = {"no", "n", "false", "0"}

def _decode_bytes_col(s: pd.Series) -> pd.Series:
    if s.dtype == object and s.apply(lambda v: isinstance(v, (bytes, bytearray))).any():
        return s.apply(lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v)
    return s

def _maybe_numeric_from_object(series: pd.Series) -> pd.Series:
    """
    Intenta convertir object -> número. Si fracasa, intenta extraer el primer número encontrado en el string.
    Aplica conversión solo si >=50% de filas producen número válido.
    """
    s = series.astype("string")
    # 1) intento directo
    as_num = pd.to_numeric(s, errors="coerce")
    if as_num.notna().mean() >= 0.5:
        return as_num

    # 2) extraer primer número del texto (ej: 'zip_28001' -> 28001)
    def _extract_first_num(x: Optional[str]) -> Optional[float]:
        if x is None or pd.isna(x):
            return np.nan
        m = _NUM_RE.search(str(x))
        return float(m.group()) if m else np.nan

    extracted = s.map(_extract_first_num)
    if extracted.notna().mean() >= 0.5:
        return extracted

    # Dejar como está (categórica)
    return series

def _factorize_categorical(series: pd.Series) -> pd.Series:
    """
    Factoriza categóricas a enteros. Reserva -1 para NA.
    """
    s = series.astype("string").fillna("__NA__")
    codes, _ = pd.factorize(s, sort=True)
    codes = pd.Series(codes, index=s.index)
    # asignar -1 a los que eran NA reales
    codes[s.eq("__NA__")] = -1
    return codes.astype("int32")

def _normalize_binary_text(y: pd.Series) -> Optional[pd.Series]:
    """
    Si y es texto y claramente binaria, la convierte a {0,1}.
    Reglas IPIP-friendly:
      - mapear 'NO'/'SI' (en cualquier capitalización) -> 0/1
      - mapear yes/no, true/false, 1/0 (en texto)
    Si no es binaria o ya es numérica, devuelve None.
    """
    if not (y.dtype == object or str(y.dtype).startswith("string")):
        return None

    s = y.astype("string").str.strip().str.lower()
    uniq = set(s.dropna().unique().tolist())
    if len(uniq) == 2:
        # dos clases — intentar mapear
        # detecta presencia de "si/sí/yes/true/1" vs "no/false/0"
        if uniq <= (_BIN_TRUE | _BIN_FALSE):
            mapped = s.map(lambda v: 1 if v in _BIN_TRUE else (0 if v in _BIN_FALSE else np.nan))
            return mapped.astype("Int64").fillna(0).astype("int32")
        # caso específico 'si'/'no' en mayúsculas (por si ya vienen así)
    if uniq == {"si", "no"} or uniq == {"sí", "no"}:
        return s.map({"no": 0, "si": 1, "sí": 1}).astype("int32")
    return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def data_preprocessing(df: pd.DataFrame):
    """
    Preprocesamiento robusto para pipelines con IPIP.
    Devuelve: X (numérico), y (Series int32).
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype="int32", name="target")

    df = df.copy()

    # --- 0) Decodificar bytes en todo el DF donde aplique
    for col in df.columns:
        df[col] = _decode_bytes_col(df[col])

    # --- 1) Selección de la variable objetivo ---
    target_col = None
    for cand in ("hosp", "class"):
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = df.columns[-1]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # --- 2) Evitar fuga: quitar columnas de bloque/ID típicas ---
    leak_cols_present = [c for c in X.columns if c.lower() in _LEAK_COLS]
    if leak_cols_present:
        X = X.drop(columns=leak_cols_present, errors="ignore")

    # --- 3) Intentar convertir object -> numérico cuando tenga sentido ---
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        X[col] = _maybe_numeric_from_object(X[col])

    # --- 4) Recalcular tipos y preparar imputaciones ---
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Numéricas: coerción segura + mediana
    if num_cols:
        X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
        for col in num_cols:
            med = X[col].median()
            X[col] = X[col].fillna(med)

    # Categóricas: factorizar + -1 para NA
    for col in cat_cols:
        X[col] = _factorize_categorical(X[col])

    # Cast ligero para ahorrar memoria (árboles/ensembles van bien con float32)
    # Mantener enteros donde tenga sentido
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype("int32")
        else:
            X[col] = X[col].astype("float32")

    # --- 5) Target a formato numérico coherente ---
    y_norm = _normalize_binary_text(y)
    if y_norm is not None:
        y = y_norm
    else:
        if y.dtype == object or str(y.dtype).startswith("string"):
            # multiclase -> factorizar
            y_codes, _ = pd.factorize(y.astype("string"), sort=True)
            y = pd.Series(y_codes, index=y.index, name=y.name).astype("int32")
        else:
            y = pd.to_numeric(y, errors="coerce")
            # eliminar NA en y (y alinear X)
            valid = y.notna()
            X = X.loc[valid]
            y = y.loc[valid].astype("int32")

    # Asegurar índices alineados
    X = X.loc[y.index]

    # Nombrar y si no tiene nombre
    if y.name is None:
        y.name = "target"

    return X, y
