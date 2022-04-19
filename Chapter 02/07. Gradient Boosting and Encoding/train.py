import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Splitting train and test data sets
x_train, x_test, y_train, y_test = dh.get_data("./insurance.csv")

# Check shapes

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# first checks of performance of each model with default paramaters
models = [RandomForestRegressor(random_state= 0), 
          GradientBoostingRegressor(random_state= 0), 
          XGBRegressor(random_state= 0), 
          AdaBoostRegressor(random_state= 0)]

for model in models:
    print(cross_val_score(model, x_train, y_train).mean())

