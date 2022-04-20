
from best_forest_select import BestForest
from preprocessing import x_train, y_train
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import joblib



# Instance of BestForest with models to select from
models = {'rf': RandomForestRegressor(random_state= 0), 
          'grdB':GradientBoostingRegressor(random_state= 0), 
          'xgB':XGBRegressor(random_state= 0), 
          'adaB':AdaBoostRegressor(random_state= 0)}

bf = BestForest(models)

# Select best
selected_model = bf.get_best(x_train, y_train)
print('Selected_model: ', selected_model, '\n')
print('All forest scores:',  bf.get_all_forest_scores())


# fit and save selected model for later use
final_model = selected_model.fit(x_train, y_train)
joblib.dump(final_model, 'model.joblib')
