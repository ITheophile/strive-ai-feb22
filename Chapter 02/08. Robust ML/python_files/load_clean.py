from titanic_functions import load


# Loading the titanic's train and test sets
pth_train = '../titanic/train.csv'
pth_test = '../titanic/test.csv'

df = load(pth_train)
df_test = load(pth_test)

print(df.sample(10))
print(df_test.sample(10))