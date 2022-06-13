from torchvision import models
import torch.nn as nn
from collections import OrderedDict


model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(OrderedDict([
    ('linear1', nn.Linear(model.fc.in_features, 256)),
    ('relu1', nn.ReLU()),
    ('drop', nn.Dropout(0.2)),
    ('linear2', nn.Linear(256, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.fc = classifier
