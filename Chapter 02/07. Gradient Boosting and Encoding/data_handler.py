import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


def get_data(pth):
    """
    read data in; split into train and test datasets
    Args(str): path to the dataset
    Returns (tuple of pandas Dataframes and pandas Series): x_train and x_test (dfs), y_train and y_test (series)
    """

    # get data
    data = pd.read_csv(pth)

    # split
    x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.2, random_state = 0)
    
    return x_train, x_test, y_train, y_test



def preprocessor(numerical_features, categorical_features):
    """
    Set up transformations (ordinal encoding and standardization) to apply on selected features.
        Ordinal encoding is apply to `categorical_features` and standardization on `numerical_features`

    Args (list): List of column names to apply transformations on.
    Outputs (sklearn ColumnTransformer): A combination of transformers.

    """
    
    oe = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1)
    scaler = StandardScaler()
        
    ct = ColumnTransformer([('ordinal', oe, categorical_features ), 
                                ('scaler', scaler, numerical_features )], 
                                remainder='passthrough')

    return ct