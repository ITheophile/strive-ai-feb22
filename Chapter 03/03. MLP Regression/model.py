import torch
import torch.nn.functional as F
from torch import nn
# from torchsummary import summary


def nn_regression(n_features, hidden_sizes):
    neuralnet = nn.Sequential(
                                nn.Linear(n_features,hidden_sizes[0]), # 1st hidden layer
                                nn.ReLU(), 
                                nn.Linear(hidden_sizes[0], hidden_sizes[1]), # 2nd hidden layer
                                nn.ReLU(), 
                                nn.Linear(hidden_sizes[1], hidden_sizes[2]), # 3rd hidden layer
                                nn.ReLU(),
                                nn.Linear(hidden_sizes[2], 1), # output layer
                                                    
                                )
    return neuralnet


        