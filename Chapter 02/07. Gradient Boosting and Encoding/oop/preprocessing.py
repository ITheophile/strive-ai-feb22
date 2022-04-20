from best_forest_select import LoaderPreprocessor
import joblib

# LoaderPreprocessor object
lp = LoaderPreprocessor()

# Loading/reading data
lp.loader('../insurance.csv')

# show a random sample of 10 observations
print(lp.raw_data.sample(10), '\n')


# retrieve training and test datasets
x_train, x_test, y_train, y_test = lp.splitted_data

# check shapes
print(f'raw data: {lp.raw_data.shape} \n')
print('splitted datasets: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# preprocessing
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi']

ct = lp.set_transformer(numerical_features= numerical_features, categorical_features= categorical_features)

x_train = ct.fit_transform(x_train)


# save transformer for later use
joblib.dump(ct, 'column_transformer.joblib')