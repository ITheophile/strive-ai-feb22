from torchtext import datasets

train, test = datasets.AG_NEWS()

sentences = []
target = []

for targ, sent in train:
    sentences.append(sent)
    target.append(targ)


print(sentences[0])
print(target[0])
