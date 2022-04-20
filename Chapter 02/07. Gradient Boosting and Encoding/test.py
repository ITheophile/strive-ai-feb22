import joblib

from train import x_test, y_test

# Loading model and preprocessor
ct = joblib.load('preprocessor.joblib')
model = joblib.load('model.joblib')
# tuned_model = joblib.load('tuned_model_joblib')


# confirm x_test and y_test shapes
print(x_test.shape, y_test.shape)

# preprocessing
x_test = ct.transform(x_test)

# tuned and non-tuned model performance on test set
print(f'model accuracy on test set: {round(model.score(x_test, y_test), 2)}')
# print(f'tuned_model accuracy on test set: {round(tuned_model.score(x_test, y_test), 2)}')

