#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import LeakyReLU
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

import util
import numpy as np

from gensim.models import KeyedVectors

SEQUENCE_LENGTH = 6
BATCH_SIZE = 256

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def build_model(vocab_size, sequence_length, embedding_size):
    model = Sequential()
    model.add(LSTM(128, input_shape=(sequence_length, embedding_size), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(1024))
    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer="rmsprop", metrics = ['accuracy'])
    return model

def generate_ngrams(X_data, Y_data, batch_size, sequence_length, embedding_size, idx2word, w2v):
    x_train = np.zeros((batch_size, sequence_length, embedding_size))
    y_train = np.zeros((batch_size, ))
    i = 0
    for j in range(X_data.shape[0]):
        xs = list(map(lambda x: idx2word[x], X_data[j]))
        xs = util.encode_word_sequence(xs, w2v)
        y = Y_data[j]
        x_train[i] = xs
        y_train[i] = y
        if i >= batch_size-1:
            yield x_train, y_train
            i = 0
        else:
            i += 1

def perplexity_score(estimator, X_test, Y_test, idx2word, w2v):
    perplexity = 0
    for j in range(2000):
        xs = list(map(lambda x: idx2word[x], X_test[j]))
        xs = util.encode_word_sequence(xs, w2v)
        xs = xs.reshape(1, xs.shape[0], xs.shape[1])
        y = Y_test[j]

        perplexity += np.log2(estimator.predict(xs)[0][y])
    return np.power(2, -perplexity * 1/2000)

def main():
    w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)

    words = util.load_vocab("small_vocab.pkl")
    vocab_size = len(words)
    print("Vocab size:", vocab_size)
    print("W2V vocab size:", len(w2v.vocab))
    idx2word = { i:word for i,word in enumerate(words) }

    embedding_size = w2v.vector_size + util.EMBEDDING_EXT
    print("Embedding size:", embedding_size)

    model = build_model(vocab_size, SEQUENCE_LENGTH, embedding_size)
    model.summary()

    data = np.load("ngrams_small.npy")
    X = data[:, :-1]
    Y = data[:, -1]

    kfold = KFold(n_splits=4)
    scores = np.zeros((4,))
    for (i, (train_index, test_index)) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        checkpoint = ModelCheckpoint("weights/weights_word_k{}_{}.h5".format(i, "{epoch:01d}"),
            monitor='loss',
            verbose=1,
            mode='auto',
            period=1,
            save_weights_only=True)

        model.fit_generator(generate_ngrams(X_train, Y_train, BATCH_SIZE, SEQUENCE_LENGTH, embedding_size, idx2word, w2v), samples_per_epoch=300, epochs=4, callbacks=[checkpoint])
        perp = perplexity_score(model, X_test, Y_test, idx2word, w2v)
        print("Local perplexity:", perp)
        scores[i] = perp
    print("Perplexity: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    main()