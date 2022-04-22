# Important libraries
import pandas as pd

# load data

pth = 'data/london_merged.csv'
data = pd.read_csv(pth)
print(data.head())