import joblib
import pandas as pd

# Loading model and preprocessor
ct = joblib.load('preprocessor.joblib')
model = joblib.load('model.joblib')

# prediction using user data through command line interface
while True:

    age = int(input("How old are you? \n"))
    sex = input("Are you male or female? \n")
    bmi = float(input("What is your bmi? \n"))
    children = int(input("How many children do you have? \n"))
    region = input("""In which of the following regions do you live?
    (southwest, southeast, northwest, northeast) \n""")
    smoker = input("Do you smoke? \n")

    
    user_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
    columns= ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    
    #Preprocess
    user_data_transfomed = ct.transform(user_data)

    #predict
    pred = model.predict(user_data_transfomed)[0]
    
    
    # display prediction
    print(f"your predicted insurance charges amount to {round(pred, 2)}")

    
