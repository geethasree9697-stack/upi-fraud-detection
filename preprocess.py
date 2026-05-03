import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset/paysim.csv")

df = df[['step','type','amount',
         'oldbalanceOrg','newbalanceOrig',
         'oldbalanceDest','newbalanceDest',
         'isFraud']]

encoder = LabelEncoder()
df['type'] = encoder.fit_transform(df['type'])

df.to_csv("dataset/processed.csv", index=False)
print("Preprocessing done")
print(df.columns)
print(df["type"].unique())
print(df.head())