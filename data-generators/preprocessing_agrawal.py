import pandas as pd


final_df = pd.read_csv("/home/alex/demo/data-generators/generated-data/agrawal_sin_drift.csv")



final_df['elevel'] = final_df['elevel'].apply(lambda x: x.decode('utf-8')).str.extract(r'(\d+)').astype(int)
final_df['car'] = final_df['car'].apply(lambda x: x.decode('utf-8')).str.extract(r'(\d+)').astype(int)
final_df['zipcode'] = final_df['zipcode'].apply(lambda x: x.decode('utf-8')).str.extract(r'(\d+)').astype(int)

final_df.to_csv("/home/alex/demo/data-generators/generated-data/agrawal_sin_drift.csv", index=False)