import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load data
data = pd.read_csv('climate.csv').drop(columns='Date Time')
# print(data.shape)
print(data.head(14))


# Function to extract sequences and target variable
def pairing(data, seq_len=6):
    x = []
    y = []
    for i in range(0, (data.shape[0] - seq_len + 1), seq_len + 1):

        seq = np.zeros((seq_len, data.shape[1]))

        for j in range(seq_len):
            seq[j] = data.values[i + j]
            # print(i, j, i + seq_len)

        x.append(seq)
        y.append(data['T (degC)'][i + seq_len])

    return np.array(x), np.array(y)


x, y = pairing(data)
print(x.shape)
print(y.shape)

# Extract features


def getfeatures(data):

    # for holding extracted features
    new_data = []

    # get each sequence
    for i in range(data.shape[0]):

        row = []  # For holding new features per sequence
        for j in range(data.shape[2]):

            # column 0 ; p (armb)
            if j == 0:
                row.append(np.max(data[i, :, j]))
                row.append(np.min(data[i, :, j]))

            # column 1 ; T (degC):
            elif j == 1:
                row.append(np.mean(data[i, :, j]))
                row.append(np.std(data[i, :, j]))
                row.append(data[i, :, j][-1])

             # column 2 ; Tpot (K)
            elif j == 2:
                row.append(np.mean(data[i, :, j]))
                row.append(np.std(data[i, :, j]))

             # column 3; Tdew (degC)
            elif j == 3:
                row.append(np.mean(data[i, :, j]))
                row.append(np.std(data[i, :, j]))

             # column 4 ; rh (%)
            elif j == 4:
                row.append(np.max(data[i, :, j]) - np.min(data[i, :, j]))

            # column 5 ; VPmax (mbar)
            elif j == 5:
                row.append(np.max(data[i, :, j]) - np.min(data[i, :, j]))


            # column 6
            elif j == 6:
                row.append(np.median(data[i, :, j]))
                row.append(np.min(data[i, :, j]))

            # any column
            else:
                row.append(np.mean(data[i, :, j]))

        new_data.append(row)

    return np.array(new_data)


new_data = getfeatures(x)


# Train Test sets
X_train, X_test, y_train, y_test = train_test_split(
    new_data, y, test_size=0.3, shuffle=False, random_state=0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# models
models = {'rf': RandomForestRegressor(random_state=0),
          'mlp': MLPRegressor(random_state=0),
          'sgd': SGDRegressor(random_state=0)}


# Pipeline
models = {name: Pipeline([("scaler", StandardScaler()), ("regressor", model)])
          for name, model in models.items()}


# Performance
def model_perfomance(models, x, y):
    results = pd.DataFrame()
    for name, pipe in models.items():

        # training time
        t0 = time.time()
        pipe.fit(x, y)
        t1 = time.time() - t0

        # cross validation scores
        scores = cross_val_score(pipe, x, y, cv=4)

        metrics = pd.DataFrame({'name': [name], 'mean_score': [
                               scores.mean()], 'std_score': [scores.std()], 'training_time': [t1]})
        results = pd.concat([results, metrics])

    return results


results = model_perfomance(models, X_train, y_train)
print(results.sort_values('mean_score', ascending=False))


# Final model
model = Pipeline([('scaler', StandardScaler()),
                 ('sgd', SGDRegressor(random_state=0))])

model.fit(X_train, y_train)

preds = model.predict(X_test)

print('test_score: ', r2_score(y_test, preds))
