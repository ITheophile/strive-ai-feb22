import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer


class LoaderPreprocessor():
    """
    class for loading and splitting the data into training and testing (through `loader method`).
    Also 
    """
    def __init__(self):
        # raw_data for EDA
        self.raw_data = None

        # training and testing data sets
        self.splitted_data = None
        
    def loader(self, pth):
        """
        read data in; split into train and test datasets
        Args(str): path to the dataset
        Update `self.raw_data` and `self.splitted_data` as  x_train and x_test (dfs), y_train and y_test (series)
        """

        # get data
        self.raw_data = pd.read_csv(pth)

        # split
        x_train, x_test, y_train, y_test = train_test_split(self.raw_data.iloc[:,:-1], self.raw_data.iloc[:,-1], test_size=0.2, random_state = 0)
        
        self.splitted_data =  x_train, x_test, y_train, y_test


    def set_transformer(self, numerical_features, categorical_features):
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


class BestForest():
    """
    Class for selecting the most accurate forest. Additionally, possibility to see the scores (cross validation means)
    of all forest candidates.
    """
    def __init__(self, models):
        """ models(dictionary): A dictionary of selected models. keys are models names and values are
            instances of said models."""
        self.models = models
        self.all_forest_scores = None

    def get_best(self, x_train, y_train):
        """
        Given a number of models (`self.models`), select the best one based on the mean of cross validation scores.

        Args:
            
            x_train (Pandas Dataframe | Numpy array): features from traing set
            y_train (Pandas Series | Numpy array): target from training set

        Returns:
            A tuple of best model's name and scores mean.
        """
        # for storing means of cross validation scores
        scores = []
        model_names = []

        for name in self.models:
            score = cross_val_score(self.models[name], x_train, y_train).mean()
            scores.append( round(score, 3)) 
            model_names.append(name)
        
        # locations of sorted scores from highest to lowest
        idx = np.argsort(scores)[::-1]

        sorted_models = [(model_names[i], scores[i]) for i in idx]

        self.all_forest_scores = sorted_models
        best_model_name = sorted_models[0][0]
        
        return self.models[best_model_name]

        
    def get_all_forest_scores(self):
        return self.all_forest_scores