import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
from model import neuralnet
from data_handler import load_data

torch.manual_seed(0)

# Data
trainloader, testloader = load_data('./data')

# TRAINING AND VALIDATION
learning_rate = 0.001
epochs = 100  # Many more epochs are needed(minimum 100)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(neuralnet.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
benchmark_accuracy = 0.89
for epoch in tqdm(range(epochs)):
    train_batch_accuracies = []
    train_batch_losses = []
    # training
    for x_train_batch, y_train_batch in trainloader:

        optimizer.zero_grad()

        # forward pass
        logits = neuralnet(x_train_batch.view(x_train_batch.shape[0], -1))
        train_preds = torch.argmax(logits.detach(), dim=1)

        # loss
        train_loss = criterion(logits, y_train_batch)
        train_batch_losses.append(train_loss.item())

        # train accuracy
        train_batch_accuracies.append(
            accuracy_score(y_train_batch, train_preds))

        # backward pass
        train_loss.backward()

        optimizer.step()

    # mean loss (all batches losses divided by the total number of batches)
    train_losses.append(sum(train_batch_losses) / len(trainloader))
    train_accuracies.append(sum(train_batch_accuracies) / len(trainloader))
    # mean accuracies

    # validation
    neuralnet.eval()
    with torch.no_grad():
        test_batch_accuracies = []
        test_batch_losses = []

        for x_test_batch, y_test_batch in testloader:

            # logits
            test_logits = neuralnet(
                x_test_batch.view(x_test_batch.shape[0], -1))

            # predictions
            test_preds = torch.argmax(test_logits, dim=1)

            # accuracy
            test_batch_accuracies.append(
                accuracy_score(y_test_batch, test_preds))

            # loss
            test_loss = criterion(test_logits, y_test_batch)
            test_batch_losses.append(test_loss.item())

        # mean accuracy for each epoch
        test_accuracies.append(sum(test_batch_accuracies)/len(testloader))

        # mean loss for each epoch
        test_losses.append(sum(test_batch_losses)/len(testloader))

        # saving best model
        # is current mean score (mean per epoch) greater than or equal to the benchmark?
        if test_accuracies[-1] > benchmark_accuracy:
            # save model
            torch.save(neuralnet.state_dict(), 'model.pth')

            # update benckmark
            benchmark_accuracy = test_accuracies[-1]

    neuralnet.train()


# Plots
x_epochs = list(range(epochs))
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(x_epochs, train_losses, marker='o', label='train')
plt.plot(x_epochs, test_losses, marker='o', label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_epochs, train_accuracies, marker='o', label='train')
plt.plot(x_epochs, test_accuracies, marker='o', label='test')
plt.axhline(benchmark_accuracy, c='grey', ls='--',
            label=f'benchmark_accuracy({benchmark_accuracy :.2f})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./learning_curves2.png', dpi=200)
plt.show()
