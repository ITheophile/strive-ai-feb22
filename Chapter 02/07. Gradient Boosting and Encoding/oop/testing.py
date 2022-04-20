import joblib
from preprocessing import x_test, y_test


# Loading model and preprocessor
ct = joblib.load('column_transformer.joblib')
model = joblib.load('model.joblib')

# confirm x_test and y_test shapes
print(f'Test dataset shape: {x_test.shape, y_test.shape} \n')

# preprocessing
x_test = ct.transform(x_test)

# model performance on test set
print(f'model accuracy on test set: {round(model.score(x_test, y_test), 2)}')