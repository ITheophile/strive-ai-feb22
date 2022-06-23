import pandas as pd
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText

np.random.seed(0)

nlp = spacy.load("en_core_web_sm")
fasttext = FastText("simple")


def load_data(pth):
    """
    Load the dataset and return a dataframe with columns ([star, review])
    """
    df = pd.read_csv(pth,  header=None,
                     skiprows=1, usecols=[1, 2, 3])
    df.rename({1: "star", 2: "rating1", 3: "rating2"}, axis=1, inplace=True)
    df["review"] = df["rating1"] + " " + df["rating2"]
    df.drop(columns=["rating1", "rating2"], inplace=True)
    return df


def train_test_split(df, train_size=0.7):

    df_idx = [i for i in range(len(df))]
    np.random.shuffle(df_idx)

    len_train = int(len(df) * train_size)
    train_idx, test_idx = df_idx[:len_train], df_idx[len_train:]

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def preprocessing(sentence):
    """
    Get rid of punctuations, space, stop words, currency symbols,
    email-alike sequences, url-alike sequences numbers and non-ascii characters.
    params sentence: a str containing the sentence we want to preprocess

    return the tokens list
    """
    doc = nlp(sentence)

    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop
              and not token.is_currency and not token.like_email and not token.like_url and not token.is_digit
              and token.is_ascii]
    return tokens


def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0


def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + \
        (max_seq_len - len(list_of_indexes))*[padding_index]
    return output[:max_seq_len]


class DataClass(Dataset):
    def __init__(self, df, max_seq_len=32):
        self.max_seq_len = max_seq_len

        train_iter = iter(df.review.values)
        self.vec = FastText("simple")
        # replacing the vector associated with 1 (padded value) to become a vector of -1.
        self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0])
        # replacing the vector associated with 0 (unknown) to become zeros
        self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0])
        self.vectorizer = lambda x: self.vec.vectors[x]
        self.labels = df.star
        sequences = [padding(encoder(preprocessing(
            sequence), self.vec), max_seq_len) for sequence in df.review.tolist()]
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]


# IMPLIMENTING ABOVE FUNCTIONS
BATCH_SIZE = 16
df = load_data("data/Amazon/3000test.csv")

train_df, test_df = train_test_split(df)

train_df, test_df = DataClass(train_df), DataClass(test_df)


def collate_train(batch, vectorizer=train_df.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token)
                         for token in sentence[0]]) for sentence in batch])
    # Use long tensor to avoid unwanted rounding
    target = torch.LongTensor([item[1] for item in batch])
    return inputs, target


def collate_test(batch, vectorizer=test_df.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token)
                         for token in sentence[0]]) for sentence in batch])
    # Use long tensor to avoid unwanted rounding
    target = torch.LongTensor([item[1] for item in batch])
    return inputs, target


train_loader = DataLoader(train_df, batch_size=BATCH_SIZE,
                          collate_fn=collate_train, shuffle=True)


test_loader = DataLoader(
    test_df, batch_size=BATCH_SIZE, collate_fn=collate_test)
