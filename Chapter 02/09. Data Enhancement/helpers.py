import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def train_test_sets(data, target, drop_columns):
    """Splits the data into training and test sets

    Args:
        data (Pandas DataFrame): The data to split
        drop_columns(list of str) : columns to drop
        target (str): The variable to predict
    Returns:
        x_train, x_test, y_train, y_test
    """
    X = data.drop(columns=[*drop_columns, target])
    y = data[target]

    return train_test_split(X, y, test_size=0.2, random_state=0)


def mean_cross_val_score(x_train, y_train, model):
    """
    Computes the mean of cross validation scores
    Args:
        x_train (2D Numpy array | Pandas DataFrame): Features to train on
        y_train (1D Numpy array | Pandas Series): Target variable
        model (Estimator): Model to fit

    Returns:
        float: mean of cross validation scores
    """
    
    return cross_val_score(model, x_train, y_train).mean()