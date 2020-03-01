#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from random import shuffle
import util
import gc

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import LeakyReLU
from pathlib import Path
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from sklearn.model_selection import KFold
import argparse

DEFAULT_SEQ_LEN = 8
BATCH_SIZE = 256

def build_model(sequence_length, vocab_size):
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, vocab_size)))
    model.add(Dropout(0.4))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer="rmsprop", metrics = ['accuracy'])
    return model

def generate_batches(X_data, Y_data, batch_size, vocab_size):
    """
    Generates mini-batches.

    Args:
        X_data: Training data input
        Y_data: Training data target
        batch_size: Size of the mini-batches
        vocab_size: Number of possible characters
    Yields:
        Tuple of numpy arrays (x, y)
    """
    while True:
        x_train = np.zeros((batch_size, X_data.shape[1], vocab_size))
        y_train = np.zeros((batch_size, vocab_size))
        i = 0
        for j in range(X_data.shape[0]):
            for k in range(X_data.shape[1]):
                x_train[i][k][X_data[j][k]] = 1
            y_train[i][Y_data[j]] = 1

            if i >= batch_size-1:
                yield x_train, y_train
                x_train = np.zeros((batch_size, X_data.shape[1], vocab_size))
                y_train = np.zeros((batch_size, vocab_size))
                i = 0
            else:
                i += 1

def perplexity_score(estimator, X_test, Y_test, vocab_size):
    """
    Calculates the perplexity of an estimator.

    Args:
        estimator: Model that estimates the likelihood of the data
        X_test: Test data input
        Y_test: Test data target
        vocab_size: Number of possible characters
    Returns:
        Perplexity, float
    """
    perplexity = 0
    for j in range(2000):
        xs = np.zeros((1, X_test.shape[1], vocab_size))
        for i in range(X_test.shape[1]):
            xs[0][i][X_test[j][i]] = 1

        perplexity += np.log2(estimator.predict(xs)[0][Y_test[j]])
    return np.power(2, -perplexity * 1/2000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input .npy training data", required=True)
    parser.add_argument("-s", "--seqlen", help="Sequence length")
    args = parser.parse_args()

    if not args.seqlen:
        sequence_length = DEFAULT_SEQ_LEN
    else:
        sequence_length = int(args.seqlen)

    df = pd.read_csv("data/songdata.zip")
    path = Path("chars.pkl")
    chars = list()
    if path.is_file():
        chars = util.load_vocab(path)
        print("Loaded from file")
    else:
        vocab = set()
        for song in df["text"]:
            chars = set(song)
            vocab = vocab.union(chars)
        chars = list(vocab)
        util.write_vocab(path, chars)
        print("Generated from source")
        
    vocab_size = len(chars)
    print("Vocab size:", vocab_size)
    
    data = np.load(args.input)
    X = data[:, :-1]
    Y = data[:, -1]

    kfold = KFold(n_splits=4)
    scores = np.zeros((4,))
    for (i, (train_index, test_index)) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        checkpoint = ModelCheckpoint("weights/weights_char_k{}_{}.h5".format(i, "{epoch:01d}"),
            monitor='loss',
            verbose=1,
            mode='auto',
            period=1,
            save_weights_only=True)

        model = build_model(sequence_length, vocab_size)
        model.fit_generator(generate_batches(X_train, Y_train, BATCH_SIZE, vocab_size), samples_per_epoch=300, epochs=10, callbacks=[checkpoint])
        perp = perplexity_score(model, X_test, Y_test, vocab_size)
        print("Local perplexity:", perp)
        scores[i] = perp

        del X_train, X_test, Y_train, Y_test
        del model
        gc.collect()
    print("Perplexity: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()