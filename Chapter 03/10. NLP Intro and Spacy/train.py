import torch
from torch.autograd import Variable
from torch.nn import NLLLoss
from torch.nn.functional import log_softmax
import numpy as np

from utils import *


preprocessed = preprocess('data/article.txt')


cleaned = clean_sentences(preprocessed)


word_idx, vocab_size = get_dicts(cleaned)


pairs = get_pairs(cleaned, word_idx, 4)


def input_layer(word_idx, vocab_size):
    x = torch.zeros(vocab_size)
    x[word_idx] = 1.0
    return x


def train(dataset, word_idx, n_epochs, lr, embedding_size, vocab_size):

    W1 = Variable(torch.random(
        vocab_size, embedding_size).float(), requires_grad=True)
    W2 = Variable(torch.random(embedding_size,
                  vocab_size).float(), requires_grad=True)

    for epoch in n_epochs:

        loss_val = 0

        for data, target in dataset:

            x = Variable(input_layer(data[0]), vocab_size).float()
            y_true = Variable(torch.from_numpy(np.array([target]))).long()

            z1 = np.matmul(x, W1)
            z2 = np.matmul(z1, W2)

            log_softmax = log_softmax(z2, dim=0)
            loss = NLLLoss(log_softmax(1, -1), y_true)

            loss_val += loss

            W1.data -= lr * W1.gradient_data
            W2.data -= lr * W2.gradient_data

            W1.gradient_data = 0
            W2.gradient_data = 0

            if epoch % 10 == 0:
                print(f'Loss at epoch {epoch}: {loss_val/len(dataset)}')
