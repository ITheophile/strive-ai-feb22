# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 06:19:31 2022

@author: Theo
"""

import numpy as np
import matplotlib.pyplot as plt


class LogReg():
    
    def __init__(self):
        self.weights = None
        self.sum_errors = None
        self.itirations = None
        
    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))
    
    def fit(self, features, target, num_steps, learning_rate, add_intercept = False):
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
        
        weights = np.zeros(features.shape[1])
        
        self.sum_errors = []
        self.itirations = [itr for itr in range(num_steps)]
        for step in range(num_steps):
        
            scores = features.dot(weights)
            preds = self.sigmoid(scores)
            error = target - preds
            gradient = features.T.dot(error)
            weights = weights + learning_rate*gradient
            
            # for plotting in learning curve
            self.sum_errors.append(np.abs(error).sum() )
            
            self.weights = weights
            
            
            
    def predict(self, features):
        
         scores = features.dot(self.weights) 
         
         return np.round(self.sigmoid(scores))
     
        
    def score(self, predictions, target):
         
         return ((predictions == target).sum()) / target.shape[0]
     
        
    def learning_curve(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.itirations, self.sum_errors)
        plt.axhline(y = 0, c = 'lightgrey')
        plt.xlabel("Number of iterations")
        plt.ylabel('sum of errors(absolute values)')
        
        
        
         
            
        
    
    
    
    