from sklearn.feature_extraction import img_to_graph
from sympy import dict_merge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


# Load model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.fc1 = nn.Linear(128*28*28, 128)
        self.drop1 = nn.Dropout(0.10)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.20)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


model = Net()

model_state = torch.load('convnet_state.pth')
model.load_state_dict(model_state)


# Load and preprocess


preprocessor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

img = Image.open('cat_dog_images/cat4.jpg')
img = preprocessor(img)
img = img.view(1, *img.shape)


# Get prediction


def get_prediction(img, model, classes_dict):
    model.eval()
    with torch.no_grad():
        probs = torch.exp(model(img))
        prob, pred = torch.max(probs, dim=1)
        print(
            f"It's an image of {classes_dict[pred.item()]} [{prob.item()*100  :.2f}%]")


classes_dict = {0: 'cat', 1: 'dog'}
get_prediction(img, model, classes_dict)
