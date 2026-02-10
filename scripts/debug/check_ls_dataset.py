import pandas as pd

df = pd.read_csv("gvns/ls_data/FI/C102_30_3.csv")

print(df.columns)
print(df.head())
print(df.isnull().sum())
print(df.describe())