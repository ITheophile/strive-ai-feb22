import torch
from torchvision import datasets, transforms
torch.manual_seed(0)


def load_data(pth):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    trainset = datasets.FashionMNIST(
        pth, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.FashionMNIST(
        pth, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True)
    return trainloader, testloader
