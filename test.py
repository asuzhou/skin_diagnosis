import pandas as pd

df1=pd.read_csv('data/metadata.csv')
print(df1['diagnostic'].unique())