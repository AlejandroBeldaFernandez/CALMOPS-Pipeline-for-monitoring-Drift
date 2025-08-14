import pandas as pd
import numpy as np

def data_preprocessing(df: pd.DataFrame):
    """
    Preprocesamiento robusto y neutro para escenarios por bloques.
    - NO usa columnas de bloque para el modelo (se eliminan si están).
    - Evita fuga: elimina IDs obvios y columnas con cardinalidad ~ total.
    - Convierte numéricos en texto -> numérico si hay evidencia suficiente.
    - Imputa: mediana (numéricos), categoría '__NA__' (categóricas).
    - Codifica categóricas con factorize (con -1 para NA reales).
    - Limpia columnas constantes y alinea índices.
    Devuelve:
        X (solo numérico, dtypes float32/int32), y (Series int32/float32)
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype="int32", name="target")

    df = df.copy()

    # --- 0) Bytes -> str, solo donde aplique
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].apply(lambda v: isinstance(v, (bytes, bytearray))).any():
                df[col] = df[col].apply(
                    lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
                )

    # --- 1) Target
    if "class" in df.columns:
        y = df["class"].copy()
        X = df.drop(columns=["class"])
        y.name = "class"
    else:
        y = df.iloc[:, -1].copy()
        X = df.drop(columns=[df.columns[-1]])
        y.name = y.name or "target"

    # --- 2) Quitar columnas de bloque e IDs obvios (para evitar fuga)
    #     (el run_pipeline ya coge la info de bloque del df original)
    leak_like = {"block", "block_id", "blockid"}
    # heurística de nombres de id (no agresiva)
    id_like = {c for c in X.columns if any(tok in c.lower() for tok in ["_id", "id", "uuid"])}
    drop_cols = [c for c in X.columns if c.lower() in leak_like] + list(id_like)
    drop_cols = list(dict.fromkeys(drop_cols))  # dedup
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # --- 3) Eliminar columnas vacías (todo NaN)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # --- 4) Intento de conversión a numérico en columnas object
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        as_num = pd.to_numeric(X[col], errors="coerce")
        # Convertimos si >= 70% pueden ser numéricos
        if as_num.notna().mean() >= 0.70:
            X[col] = as_num

    # --- 5) Recomputar tipos, manejar booleanos
    # Booleans a int8 para coherencia numérica
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # --- 6) Imputación
    # Numéricas: mediana
    if num_cols:
        X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
        for col in num_cols:
            med = X[col].median()
            X[col] = X[col].fillna(med)
        # Cast ligero
        X[num_cols] = X[num_cols].astype("float32")

    # Categóricas: factorizar + -1 para NA reales
    for col in cat_cols:
        s = X[col].astype("string")
        na_mask = s.isna()
        s = s.fillna("__NA__")
        codes, _ = pd.factorize(s, sort=True)
        codes = pd.Series(codes, index=s.index)
        # -1 para NA reales originales
        codes[na_mask] = -1
        X[col] = codes.astype("int32")

    # --- 7) Limpiar columnas constantes (varianza cero)
    if not X.empty:
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)

    # --- 8) Target a formato numérico estable
    if y.dtype == object or str(y.dtype).startswith(("string",)):
        y = y.astype("string")
        y_codes, _ = pd.factorize(y, sort=True)
        y = pd.Series(y_codes, index=y.index, name=y.name).astype("int32")
    elif y.dtype == bool:
        y = y.astype("int32")
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        if not valid.all():
            X = X.loc[valid]
            y = y.loc[valid]
        # si es float con valores enteros, lo bajamos a int32
        if np.allclose(y, y.astype("int64", copy=False), equal_nan=False):
            y = y.astype("int32")
        else:
            y = y.astype("float32")

    # --- 9) Alinear índices por seguridad
    X = X.loc[y.index]

    return X, y
