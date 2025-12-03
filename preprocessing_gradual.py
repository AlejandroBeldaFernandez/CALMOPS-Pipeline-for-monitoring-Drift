import pandas as pd


def data_preprocessing(df: pd.DataFrame):
    """
    Preprocessing for Gradual Drift Scenario (SEA).
    Features: col_0, col_1, col_2
    Target: target
    Metadata: block, timestamp
    """
    # Define special columns to exclude from features
    target_col = "target"
    block_col = "block"
    time_col = "timestamp"

    # Separate X and y
    # We drop block/time from X as they are metadata
    drop_cols = [target_col, block_col, time_col]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]
    y = y.astype(int)
    return X, y
