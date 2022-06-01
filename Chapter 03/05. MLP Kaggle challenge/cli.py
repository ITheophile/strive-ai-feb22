import torch
from torchvision import transforms
from model import neuralnet
import argparse
from PIL import Image

# Loading model
state_dict = torch.load('model.pth')

neuralnet.load_state_dict(state_dict)

neuralnet.eval()

# Setting up the CLI
parser = argparse.ArgumentParser(
    description='Predict the category of the images according Fashion MNIST')

parser.add_argument('image_path', type=str,
                    help='Path to the image')

args = parser.parse_args()

path_to_image = args.image_path

# Load image

img = Image.open(path_to_image)

# img.show()

# transform
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Resize((28, 28)),
                                transforms.Normalize((0.5,), (0.5,))])
img = transform(img)


# Predictions
article_name = {0: 'T-shirt/top',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle Boot'}


pred = torch.argmax(neuralnet(img.view(1, -1)), dim=1).item()

print(article_name[pred])
