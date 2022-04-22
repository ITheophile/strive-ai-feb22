# Important libraries
from lib2to3.pgen2.token import STAR
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# load data

pth = 'data/london_merged.csv'
data = pd.read_csv(pth)


# feature selection
"""
Based on EDA, t1 and t2 are highly correlated(0.99), so one of them need to be droped. Since t1 seems to have
a slightly higher correlation to cnt, we'll drop t2
"""

data = data.drop('t2', axis=1)
print(data.columns)

# feature generation
"""
EDA also reveals that month could a feature with potentially good explanotry power.
So we'll create it from the timestamp column and then drop that column
"""

data['month'] = pd.to_datetime(data['timestamp']).dt.month

data = data.drop(columns='timestamp')

# 1 Model without data enhacement

# training and test sets
X = data.drop(columns='cnt')
y = data['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


cat_vars = ['season','is_weekend','is_holiday','month','weather_code']
num_vars = ['t1', 'hum','wind_speed']


# Preprocessing. 
""" 
Though there is no missing values and categorical variables seems to be already in numerical form, 
we'll still want to be able to handle future data.
"""

num_pipe = make_pipeline(steps=[
    (SimpleImputer(strategy='constant', fill_value=-9999))
])

cat_pipe = make_pipeline(steps=[
    (SimpleImputer(strategy='constant', fill_value='missing')),
    (OrdinalEncoder(handle_unknown='ignore'))
])

preprocessor = make_column_transformer(transformers=[
    (num_pipe, num_vars),
    (cat_pipe, cat_vars),
], remainder='drop')


# Training

models = {'rf': RandomForestRegressor(random_state=0), 
          'gb': GradientBoostingRegressor(random_state=0)}

models_pipe = {name: make_pipeline([(preprocessor), (model)]) for name, model in models.items()}
results = pd.DataFrame({'Model': [], 'train_cv_score': [], 'MSE': [], 'MAB': [], 'Time': []})

for name, model in models_pipe.items():
    start_time = time.time()


