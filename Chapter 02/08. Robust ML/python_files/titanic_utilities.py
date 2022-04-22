
import numpy    as np
import pandas   as pd

from sklearn.pipeline import Pipeline   
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# load
def load(pth):
    """
    Load data file.
    Args (str). Location of file to read in
    Returns (Pandas DataFrame). A DataFrame with index set to `PassengerId`
    """
    return pd.read_csv(pth, index_col='PassengerId')



class CleanData():
    """
    Class for handling the cleaning of data
    """
    def __init__(self, data):
        self.data = data
        self.title_column = None


    def get_title(self, name_column, title_column):
        """
        Extract the title from Passenger's name and create a new column in `self.data` as `name_column`.  
        Args:
          name_column (str): The name of column containing passengers' names.
          title_column (str): The name of column where the passengers' titles will be stored.
        Returns:
          self
        """
        get_Title_from_Name = lambda x: x.split(',')[1].split('.')[0].strip()
        self.data[title_column] = self.data[name_column].apply(get_Title_from_Name)
        self.title_column = title_column
        
        return self

    def map_title(self, mapper:dict):
        """
        Create broader categories of titles based on a specified `mapper` and overwrite the `title_column`
        Args:
          mapper (dict): A dictionary of old titles (keys) and new titles (values)

        Returns:
          self
        """

        self.data[self.title_column] = self.data[self.title_column].map(mapper)
        return self

    def drop_features(self, column_names:list):
        """
        Removes columns and overwrite `self.data`.
        Args:
          column_names (list): List of columns' names to remove
        """
        self.data.drop(columns = column_names, inplace = True)