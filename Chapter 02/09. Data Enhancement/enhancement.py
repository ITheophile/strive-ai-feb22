# Important libraries
import pandas as pd
from helpers import train_test_sets, mean_cross_val_score
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


# 2. Reframing the problem to predicting daily bike shares
"""
High variability due to hourly measurements might make it hard for the user
algorithm to find the underlying trend in the data.
"""
# To do this we'll group the data by the day and then aggregate

# Let's change timestamp to datetime object
data['timestamp'] = pd.to_datetime(data.timestamp)

# New `date` column
data['date'] = data.timestamp.dt.date

# Let's also create a new feature `is_good_weather` (1 == weather is good, 0 otherwise)
# This will help reduce the cardinality and potentially helps the model improve.
# Good weather means here, there's no precipitation (is_good_weather == 1).
# Otherwise, is_good_weather == 0.


data['is_good_weather'] = data['weather_code'].apply(
    lambda x: 1 if x < 4 else 0)

# Grouping variables by datatype to apply different aggregations
df_num = data[['date', 'cnt', 't1', 't2', 'hum', 'wind_speed']]
df_cat = data[['date', 'is_good_weather', 'weather_code',
               'is_holiday', 'is_weekend', 'season']]

df_num_grp = df_num.groupby('date').agg('mean')
df_cat_grp = df_cat.groupby('date').agg(pd.Series.mode)

# aggregation created an additional class [0, 1] (for bimodal cases)
df_cat_grp['is_good_weather'] = df_cat_grp.is_good_weather.astype(
    str).map({'[0 1]': 0, '1': 1, '0': 0})

# join back together
df_daily = pd.merge(df_num_grp, df_cat_grp, left_index=True,
                    right_index=True).reset_index()


# Model performance on daily data
X_train, X_test, y_train, y_test = train_test_sets(
    df_daily, 'cnt', ['weather_code', 'date'])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


mean_score = mean_cross_val_score(X_train, y_train, rf)
print(f'performance on daily data {mean_score}')

"""
Reframing the problem such as to predict daily bike shares gives a much better performance.
From 0.28 on raw data to 0.76 on daily data.
"""

# 3. Augmenting data
