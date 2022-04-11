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
        
    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))
    
    def fit(self, features, target, num_steps, learning_rate, add_intercept = False):
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
        
        weights = np.zeros(features.shape[1])
    
        for step in range(num_steps):
        
            scores = features.dot(weights)
            preds = self.sigmoid(scores)
            error = target - preds
            gradient = features.T.dot(error)
            weights = weights + learning_rate*gradient
            
            self.weights = weights
            
            
            
    def predict(self, features):
        
         scores = features.dot(self.weights) 
         
         return np.round(self.sigmoid(scores))
     
        
    def score(self, predictions, target):
         
         return ((predictions == target).sum()) / target.shape[0]
         
            
        
    
    
    
    