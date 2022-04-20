import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


class LoaderPreprocessor():
    """
    class for loading and splitting the data into training and testing (through `loader method`).
    Also 
    """
    def __init__(self):
        # raw_data for EDA
        self.raw_data = None
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
