from titanic_utilities import Preprocessor, train_val

from load_clean import df, df_test



# 1. splitting into train and validation sets

X_train, X_val, y_train, y_val = train_val(df, 'Survived')

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

