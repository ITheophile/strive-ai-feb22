from torchvision import models
import torch

model = models.resnext50_32x4d(pretrained=True)

inputs = model.fc.in_features
outputs = 6
clf = torch.nn.Linear(inputs, outputs)


model.fc = clf
