
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
    Returns (Pandas DataFrame).
    """
    return pd.read_csv(pth, index_col='PassengerId')



class Cleaner():
    """
    Class for handling the cleaning of data
    """
    def __init__(self, data):
        self.data = data
        