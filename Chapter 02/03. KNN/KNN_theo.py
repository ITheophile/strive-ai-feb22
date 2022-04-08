import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd

class KNN:

    def __init__(self, k):
        self.k = k

    def add_reference(self, x, y):
        # training data
        self.x = x
        self.y = y

    def euclidian(self, v1, v2):
        # distance between v1 and v2
        assert v1.shape == v2.shape
        return np.sqrt( ((v1-v2) **2).sum())

    def predict(self, x_test):
        # predicting the class (y_h) of each point in x_test
        self.y_h = []

        for i in range(x_test.shape[0]): # loop over the points in test dataset
            dif = []
            for j in range(self.x.shape[0]): # for each point in test dataset, loop over all data points in traing dataset x
                
                euc = self.euclidian(x_test[i], self.x[j]) # distance between each point in test dataset and each point in train dataset
                dif.append( [ euc, int(self.y[j]) ] ) # store the distance and real value (from training dataset) of the corresponding class
            
            df = pd.DataFrame(dif, columns=['dist','target']) # After getting all distances for a given point, store them in a pd Dataframe, together
                                                                # with the corresponding class
            ord_dif = df.sort_values(by='dist') 
            self.y_h.append( ord_dif['target'][:self.k].mode()[0] ) # access the first k elements from the column with classes, get the 
                                                                    # most frequent (mode) class and store it in y_h (predictions list)
        return self.y_h

    def evaluate(self, y_test):

        self.acc = 0
        for i in range(y_test.shape[0]): # for each class in the test dataset 
            if y_test[i] == self.y_h[i]: # if the class was predicted correctly, increase the accuracy count by 1
                self.acc += 1

        return self.acc / y_test.shape[0] # output how many correct predictions over the total number of predictions made


x, y = make_blobs(n_samples=300, n_features= 2, cluster_std = 0.6, random_state = 0, centers = 4 )
knn = KNN(5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

knn.add_reference(x_train, y_train)

predictions = knn.predict(x_test)

acc = knn.evaluate(y_test)
print(acc)