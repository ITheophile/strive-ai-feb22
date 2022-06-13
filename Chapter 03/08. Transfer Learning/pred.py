from PIL import Image
import torch
from torchvision import transforms
from model_set_up import model
import matplotlib.pyplot as plt
import numpy as np


model_state = torch.load('model.pth')
model.load_state_dict(model_state)


# Load and preprocess

preprocessor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('cat_dog_images/dog0.jpg')
np_image = np.array(img)
img = preprocessor(img)
img = img.view(1, *img.shape)


# Get prediction
def get_prediction(img, model, classes):
    model.eval()
    with torch.no_grad():
        probs = torch.exp(model(img))

        prob, pred = torch.max(probs, dim=1)
    return classes[pred.item()], prob.item()


# Prediction
classes = ['cat', 'dog']
pred, confidence = get_prediction(img, model, classes)


# Plotting

plt.figure(figsize=(8, 6))
plt.imshow(np_image)
plt.title(f'Prediction: {pred} | Confidence:{confidence * 100 : .2f}% ')
plt.xticks([])
plt.yticks([])
plt.show()
