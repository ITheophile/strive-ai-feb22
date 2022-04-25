# Important libraries
import pandas as pd
from helpers import train_test_sets, mean_cross_val_score

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


# load data
pth = 'data/london_merged.csv'
data = pd.read_csv(pth)


# 1. Model performance on raw data
X_train, X_test, y_train, y_test = train_test_sets(data, 'cnt', ['timestamp'])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

rf = RandomForestRegressor(random_state=0)

mean_score = mean_cross_val_score(X_train, y_train, rf)
print(mean_score)


# Reframing the problem to predicting daily bike shares
"""
High variability due to hourly measurements might make it hard for the user
algorithm to find the underlying trend in the data.
"""
