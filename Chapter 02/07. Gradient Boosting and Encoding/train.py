import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Splitting train and test data sets

x_train, x_test, y_train, y_test = dh.get_data('insurance.csv')


# Check shapes

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# first idea of performance of each model with default paramaters
models = {'rf': RandomForestRegressor(random_state= 0), 
          'grdB':GradientBoostingRegressor(random_state= 0), 
          'xgB':XGBRegressor(random_state= 0), 
          'adaB':AdaBoostRegressor(random_state= 0)}

for model in models:
    print(model + ' :', cross_val_score(models[model], x_train, y_train).mean())


best_model = models['grdB'].fit(x_train, y_train)
