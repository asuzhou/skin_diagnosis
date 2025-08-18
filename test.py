import pandas as pd

df1=pd.read_csv('data/metadata2.csv')
print(df1['nine_partition_label'].unique())