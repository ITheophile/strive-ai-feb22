import numpy as np
import pandas as pd
import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

# reading data, splitting into train and test data sets
x_train, x_test , y_train, y_test = dh.get_data('insurance.csv')

# check shapes of splitted datasets
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# preprocessing
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi']

ct = dh.preprocessor(numerical_features= numerical_features, categorical_features= categorical_features)

x_train = ct.fit_transform(x_train)


# select best model
def best_model(models, x_train, y_train, output_all = False):
    """
    Given a number of models, select the best one based on the mean of cross validation scores.

    Args:
        models(dictionary): A dictionary of selected models. keys are models names and values are
        instances of said models.
        x_train (Pandas Dataframe | Numpy array): features from traing set
        y_train (Pandas Series | Numpy array): target from training set
        output_all (bool): whether or not to  return instead the mean scores (sorted) of all models.

    Returns:
        A tuple of best model's name and scores mean. If `output_all` is `True`, score means of all
        models are returned.
    """
    # for storing means of cross validation scores
    scores = []
    model_names = []

    for name in models:
        score = cross_val_score(models[name], x_train, y_train).mean()
        scores.append( round(score, 3)) 
        model_names.append(name)
    
    # locations of sorted scores from highest to lowest
    idx = np.argsort(scores)[::-1]

    sorted_models = [(model_names[i], scores[i]) for i in idx]

    if output_all:
        return sorted_models

    else:
        return sorted_models[0]


# Model selection
models = {'rf': RandomForestRegressor(random_state= 0), 
          'grdB':GradientBoostingRegressor(random_state= 0), 
          'xgB':XGBRegressor(random_state= 0), 
          'adaB':AdaBoostRegressor(random_state= 0)}



print("best model:", best_model(models, x_train, y_train)) # outputs ('grdB', 0.839)

final_model_name = best_model(models, x_train, y_train)[0]
final_model = models[final_model_name]


final_model.fit(x_train, y_train)
#save final model and preprocessor for later use
joblib.dump(final_model, 'model.joblib')
joblib.dump(ct, 'preprocessor.joblib')


# hyparamater tuning
params  = {
        #    'n_estimators': np.linspace(100, 1000, 10).astype(int),
           'max_depth': np.linspace(2, 7).astype(int), 
           'learning_rate': np.linspace(0.001, 0.1, 10)}

grid = GridSearchCV(final_model, params)

grid.fit(x_train, y_train)

print(f'grid best score: {grid.best_score_}') # output grid best score: 0.8468105384491948
print(f'grid best parameters: {grid.best_params_}')#grid best score: {'learning_rate': 0.045000000000000005, 'max_depth': 3}

tuned_model = GradientBoostingRegressor(random_state= 0, learning_rate= 0.045).fit(x_train, y_train)

#saving tuned model
# joblib(tuned_model,  'tuned_model.joblib')