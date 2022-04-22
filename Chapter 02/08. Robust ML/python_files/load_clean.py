from titanic_utilities import load, CleanData


# 1. Loading the titanic's train and test sets
pth_train = '../titanic/train.csv'
pth_test = '../titanic/test.csv'

df = load(pth_train)
df_test = load(pth_test)


# print(df.sample(10))
# print(df_test.sample(10))

# 2. Cleaning
# Add Title column
train = CleanData(df)
test = CleanData(df_test)

df = train.get_title('Name', 'Title').data

df_test = test.get_title('Name', 'Title').data


# Map titles to broader categories
title_dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

df = train.map_title(title_dictionary).data
df_test = test.map_title(title_dictionary).data






