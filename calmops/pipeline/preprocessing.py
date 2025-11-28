import pandas as pd
from typing import Tuple


def data_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the input DataFrame by separating features and target.

    Args:
        df (pd.DataFrame): The input DataFrame containing both features and the target column 'class'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): The features DataFrame.
            - y (pd.Series): The target Series.
    """
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
