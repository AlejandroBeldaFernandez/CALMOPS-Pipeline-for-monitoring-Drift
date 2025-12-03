import pandas as pd
import numpy as np
from typing import Optional, Union, List

def data_preprocessing(df: pd.DataFrame, block_cols: Optional[Union[str, List[str]]] = None):
    """
    Robust preprocessing for block-based datasets (preserves block columns within X).

    - Safely chooses y (target > class > y > label > last NON-block column).
    - PRESERVES block columns in X (e.g., chunk, block, block_id...).
      * They are not transformed or removed; only bytes->str normalization is applied.
    - Other columns: converts bytes->str, attempts numeric conversion, imputes (median / '__NA__'),
      factorizes categoricals, removes constants.
    Returns: X (numeric + untouched block columns) and y (int32/float32).
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype="int32", name="target")

    df = df.copy()

    # --- Bytes -> str where applicable (includes potential block columns)
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].apply(lambda v: isinstance(v, (bytes, bytearray))).any():
                df[col] = df[col].apply(
                    lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
                )

    # --- Detect block columns
    default_block_candidates = {"chunk", "block", "block_id", "blockid", "fold", "batch"}
    detected_blocks: List[str] = []

    if block_cols is None:
        # autodetection by name or if it contains "block"
        for c in df.columns:
            lc = str(c).lower()
            if lc in default_block_candidates or "block" in lc:
                detected_blocks.append(c)
    else:
        if isinstance(block_cols, str):
            block_cols = [block_cols]
        detected_blocks = [c for c in block_cols if c in df.columns]

    detected_blocks = list(dict.fromkeys(detected_blocks))  # unique, keep order

    # --- Choose target, avoiding block columns
    target_candidates = ["target", "class", "y", "label"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        cols_no_block = [c for c in df.columns if c not in detected_blocks]
        target_col = cols_no_block[-1] if cols_no_block else df.columns[-1]

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # If for any reason y matches a block column, try to correct it
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
                        f"The detected target ('{target_col}') matches the block column ('{bcol}'). "
                        "Rename columns or ensure 'target' exists."
                    )

    # --------------------------
    # FROM NOW ON, DO NOT TOUCH THE BLOCK COLUMNS!
    # --------------------------
    non_block_cols = [c for c in X.columns if c not in detected_blocks]

    # 3) Remove completely empty columns (non-block only)
    all_nan_cols = [c for c in non_block_cols if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        non_block_cols = [c for c in non_block_cols if c not in all_nan_cols]

    # 4) Attempt numeric conversion on object columns (non-block only)
    obj_cols = [c for c in non_block_cols if X[c].dtype == object]
    for col in obj_cols:
        as_num = pd.to_numeric(X[col], errors="coerce")
        if as_num.notna().mean() >= 0.70:
            X[col] = as_num

    # 5) Booleans -> int8 (non-block only)
    bool_cols = [c for c in non_block_cols if X[c].dtype == bool]
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")

    # 6) Imputation (non-block only)
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

    # 7) Remove constants (non-block only)
    if not X.empty:
        constant_cols = [c for c in non_block_cols if X[c].nunique(dropna=False) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            non_block_cols = [c for c in non_block_cols if c not in constant_cols]

    # 8) y to a stable numeric type
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

    # 9) Align indices
    X = X.loc[y.index]

    # 10) Sanity check: avoid degenerate target
    if y.nunique() < 2:
        raise ValueError(f"Target '{target_col}' has only one class after preprocessing.")

    return X, y
