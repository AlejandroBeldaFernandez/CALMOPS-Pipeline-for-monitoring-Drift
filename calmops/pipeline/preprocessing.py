import pandas as pd

def data_preprocessing(df):
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    y = df["class"]
    X = df.drop(columns=["class"]).copy()

    # The new data format has numeric features and a numeric target (0/1).
    # The previous transformations for byte strings and string-based targets
    # are no longer needed.

    # We just need to ensure the target column is of integer type.
    y = y.astype(int)

    return X, y
