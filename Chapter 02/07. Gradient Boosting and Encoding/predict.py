from train import best_model
import pandas as pd
model = best_model

while True:

    age = int(input("How old are you? \n"))
    sex = input("Are you male or Female? \n")
    bmi = float(input("What is your bmi? \n"))
    children = int(input("How many children do you have? \n"))
    region = input('From which of the following regions are you coming from (southwest, southeast, northwest, northeast)?')
    smoker = input("Do you smoke? \n")

    user_data = pd.DataFrame({'age': age, 'sex': sex, 
                              'bmi':bmi, 'children':children, 
                              'smoker':smoker, 'region':region})
    
    '''
    Preprocess
    predict
    
    '''
    
    
    
    print("You are too fucked up 1 milly")
    # print(model.predict(user_data))
    
