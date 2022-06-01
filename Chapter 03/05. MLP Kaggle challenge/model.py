import torch.nn as nn

input_size = 28*28
hiddensizes = [512, 256, 128, 64]
output_size = 10
percent_drop = 0.10

neuralnet = nn.Sequential(
    nn.Linear(input_size, hiddensizes[0]),
    nn.Dropout(percent_drop),
    # nn.BatchNorm1d(hiddensizes[0]),
    nn.ReLU(),

    nn.Linear(hiddensizes[0], hiddensizes[1]),
    nn.Dropout(percent_drop),
    # nn.BatchNorm1d(hiddensizes[1]),
    nn.ReLU(),

    nn.Linear(hiddensizes[1], hiddensizes[2]),
    nn.Dropout(percent_drop),
    # nn.BatchNorm1d(hiddensizes[2]),
    nn.ReLU(),

    nn.Linear(hiddensizes[2], hiddensizes[3]),
    nn.Dropout(percent_drop),
    # nn.BatchNorm1d(hiddensizes[2]),
    nn.ReLU(),

    nn.Linear(hiddensizes[3], output_size),

)
