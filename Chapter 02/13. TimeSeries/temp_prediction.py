# Import necessary libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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


# pairing(data.head(50))

# Extract chunks (sequence) of data and the target variable
x, y = pairing(data)

# check
print(x.shape)
print(y.shape)
# print(x[0])
# print(y[0])


# Extract features
def getfeatures(data):

    # for holding extracted features
    new_data = []
    # get each group
    for i in range(data.shape[0]):

        group = []   # to hold extracted elements from each column
        names = []
        # get each column within each group
        for j in range(data.shape[2]):

            group.append(np.mean(data[i][:, j]))  # mean
            group.append(np.std(data[i][:, j]))  # standard deviation
            group.append(data[i][:, j][-1])      # last element

        new_data.append(group)

    return np.array(new_data)


x = getfeatures(x)
print(x.shape)


# Train Test sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)


# models
models = {'rf': RandomForestRegressor(random_state=0),
          'sv': SVR(),
          'mlp': MLPRegressor(random_state=0)}


# Pipelines
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

        # results into Dataframe
        metrics = pd.DataFrame({'name': [name], 'mean_score': [
                               scores.mean()], 'std_score': [scores.std()], 'training_time': [t1]})
        results = pd.concat([results, metrics])

    return results


print(model_perfomance(models, X_train, y_train))
