# import the needed libraries
import data_handler as dh
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from model import nn_regression
torch.manual_seed(0)


# DATA
x_train, x_test, y_train, y_test = dh.load_data('data/turkish_stocks.csv')

print(x_train.shape, y_train.shape, x_test.shape, y_train.shape)

x_train, x_test, y_train, y_test = dh.to_batches(
    x_train, x_test, y_train, y_test, batch_size=16)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# MODEL

neural_network = nn_regression(x_train.shape[2], hidden_sizes=[400, 200, 100])
print(neural_network)

# TRAINING AND VALIDATION
learning_rate = 0.01
epochs = 50
criterion = nn.L1Loss()
optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
r_squared_scores = []
benchmark_score = 0.90
for epoch in range(epochs):

    running_loss = 0
    # training
    for x_train_batch, y_train_batch in zip(x_train, y_train):

        optimizer.zero_grad()
        # predictions
        train_preds = neural_network(x_train_batch)

        # loss
        train_loss = criterion(train_preds, y_train_batch)
        running_loss += train_loss.item()

        # backward pass
        train_loss.backward()

        optimizer.step()

        # print(f'Loss: {train_loss.item() :.4f}')

    # mean loss (all batches losses divided by the total number of batches)
    train_losses.append(running_loss/x_train.shape[0])

    # validation
    with torch.no_grad():
        running_rsquared = 0
        running_loss = 0
        neural_network.eval()
        for x_test_batch, y_test_batch in zip(x_test, y_test):
            # predictions
            test_preds = neural_network(x_test_batch)

            # running score over batches
            running_rsquared += r2_score(y_test_batch, test_preds)

            # loss
            test_loss = criterion(test_preds, y_test_batch)
            running_loss += test_loss.item()

        # mean score for each epoch
        r_squared_scores.append(running_rsquared/x_test.shape[0])

        # mean loss for each epoch
        test_losses.append(running_loss/x_test.shape[0])

        # saving best model
        # is current mean score (mean per epoch) greater than or equal to the benchmark?
        if r_squared_scores[-1] > benchmark_score:
            # save model
            torch.save(neural_network, 'model.pth')

            # update benckmark
            benchmark_score = r_squared_scores[-1]


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
plt.plot(x_epochs, r_squared_scores, marker='o',
         c='red', label='r2_score_test')
plt.xlabel('Epoch')
plt.ylabel('R squared')
plt.axhline(benchmark_score, c='grey', ls='--',
            label=f'benchmark_score({benchmark_score :.2f})')
plt.legend()

plt.savefig('losses_rsquared.jpg')
plt.show()
