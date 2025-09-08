import pandas as pd
import numpy as np
from typing import Optional, Union, List

def data_preprocessing(df: pd.DataFrame, block_cols: Optional[Union[str, List[str]]] = None):
    """
    Preprocesado robusto para datasets por bloques (conserva las columnas de bloque dentro de X).

    - Elige y de forma segura (target > class > y > label > último NO-bloque).
    - CONSERVA las columnas de bloque en X (p.ej. chunk, block, block_id...).
      * No se transforman ni eliminan; sólo se normalizan bytes->str.
    - Resto de columnas: convierte bytes->str, intenta numéricos, imputa (mediana / '__NA__'),
      factoriza categóricas, borra constantes.
    Devuelve: X (numérico + columnas de bloque sin tocar) e y (int32/float32).
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype="int32", name="target")

    df = df.copy()

    # --- Bytes -> str donde aplique (incluye posibles columnas de bloque)
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].apply(lambda v: isinstance(v, (bytes, bytearray))).any():
                df[col] = df[col].apply(
                    lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
                )

    # --- Detectar columnas de bloque
    default_block_candidates = {"chunk", "block", "block_id", "blockid", "fold", "batch"}
    detected_blocks: List[str] = []

    if block_cols is None:
        # autodetección por nombre o si contiene "block"
        for c in df.columns:
            lc = str(c).lower()
            if lc in default_block_candidates or "block" in lc:
                detected_blocks.append(c)
    else:
        if isinstance(block_cols, str):
            block_cols = [block_cols]
        detected_blocks = [c for c in block_cols if c in df.columns]

    detected_blocks = list(dict.fromkeys(detected_blocks))  # unique, keep order

    # --- Elegir target evitando columnas de bloque
    target_candidates = ["target", "class", "y", "label"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        cols_no_block = [c for c in df.columns if c not in detected_blocks]
        target_col = cols_no_block[-1] if cols_no_block else df.columns[-1]

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Si por cualquier razón y coincide con alguna col de bloque, intentamos corregir
    for bcol in detected_blocks:
        if bcol in df.columns:
            same = False
            try:
                same = y.equals(df[bcol])
            except Exception:
                same = (y.astype(str).reset_index(drop=True) == df[bcol].astype(str).reset_index(drop=True)).mean() > 0.95
            if same:
                if "target" in df.columns and target_col != "target":
                    y = df["target"].copy()
                    X = df.drop(columns=["target"])
                    target_col = "target"
                else:
                    raise ValueError(
                        f"El target detectado ('{target_col}') coincide con la columna de bloque ('{bcol}'). "
                        "Renombra columnas o asegura que 'target' exista."
                    )

    # --------------------------
    # ¡A PARTIR DE AQUÍ NO TOCAMOS LAS COLUMNAS DE BLOQUE!
    # --------------------------
    non_block_cols = [c for c in X.columns if c not in detected_blocks]

    # 3) Eliminar columnas completamente vacías (solo NO-bloque)
    all_nan_cols = [c for c in non_block_cols if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        non_block_cols = [c for c in non_block_cols if c not in all_nan_cols]

    # 4) Intento de conversión a numérico en object (solo NO-bloque)
    obj_cols = [c for c in non_block_cols if X[c].dtype == object]
    for col in obj_cols:
        as_num = pd.to_numeric(X[col], errors="coerce")
        if as_num.notna().mean() >= 0.70:
            X[col] = as_num

    # 5) Booleanos -> int8 (solo NO-bloque)
    bool_cols = [c for c in non_block_cols if X[c].dtype == bool]
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")

    # 6) Imputación (solo NO-bloque)
    num_cols = X[non_block_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in non_block_cols if c not in num_cols]

    if num_cols:
        X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
        for col in num_cols:
            med = X[col].median()
            X[col] = X[col].fillna(med)
        X[num_cols] = X[num_cols].astype("float32")

    for col in cat_cols:
        s = X[col].astype("string")
        na_mask = s.isna()
        s = s.fillna("__NA__")
        codes, _ = pd.factorize(s, sort=True)
        codes = pd.Series(codes, index=s.index)
        codes[na_mask] = -1
        X[col] = codes.astype("int32")

    # 7) Quitar constantes (solo NO-bloque)
    if not X.empty:
        constant_cols = [c for c in non_block_cols if X[c].nunique(dropna=False) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            non_block_cols = [c for c in non_block_cols if c not in constant_cols]

    # 8) y a tipo numérico estable
    if y.dtype == object or str(y.dtype).startswith(("string",)):
        y = y.astype("string").str.strip()
        mapping = {"false": 0, "true": 1, "False": 0, "True": 1, "0": 0, "1": 1}
        y = y.map(lambda v: mapping[v] if v in mapping else v)
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.notna().mean() >= 0.90:
            y = y_num

    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    if not valid.all():
        X = X.loc[valid]
        y = y.loc[valid]

    if np.allclose(y, y.astype("int64", copy=False), equal_nan=False):
        y = y.astype("int32")
    else:
        y = y.astype("float32")

    # 9) Alinear índices
    X = X.loc[y.index]

    # 10) Sanidad: evitar target degenerado
    if y.nunique() < 2:
        raise ValueError(f"El target '{target_col}' tiene una sola clase tras el preprocesado.")

    return X, y
