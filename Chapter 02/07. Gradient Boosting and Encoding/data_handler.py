import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


def get_data(pth):

    data = pd.read_csv(pth)

    x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.2, random_state = 0)
    
    oe = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1)
    scaler = StandardScaler()
    
    ct = ColumnTransformer([('ordinal', oe, ['sex', 'smoker', 'region']), 
                            ('scaler', scaler, ['age', 'bmi'])], 
                            remainder='passthrough')

    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)
    
    return x_train, x_test, y_train, y_test


