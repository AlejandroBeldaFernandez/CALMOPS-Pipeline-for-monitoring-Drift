import pandas as pd

def data_preprocessing(df):
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    y = df["class"]
    X = df.drop(columns=["class"]).copy()
    X['elevel'] = X['elevel'].apply(lambda x: x.decode('utf-8')).str.extract(r'(\d+)').astype(int)
    X['car'] = X['car'].apply(lambda x: x.decode('utf-8')).str.extract(r'(\d+)').astype(int)
    X['zipcode'] = X['zipcode'].apply(lambda x: x.decode('utf-8')).str.extract(r'(\d+)').astype(int)

    y = y.apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))
    y = y.map({'groupA': 0, 'groupB': 1})

    return X, y
