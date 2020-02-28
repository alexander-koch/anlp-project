#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import LeakyReLU
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

import util
import numpy as np

from gensim.models import KeyedVectors

checkpoint = ModelCheckpoint("weights/weights_word_{epoch:01d}.h5",
    monitor='loss',
    verbose=1,
    mode='auto',
    period=1,
    save_weights_only=True)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

SEQUENCE_LENGTH = 6

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

def generate_ngrams(batch_size, sequence_length, embedding_size, idx2word, w2v):
    with open("ngrams_small.txt", "r") as f:
        x_train = np.zeros((batch_size, sequence_length, embedding_size))
        y_train = np.zeros((batch_size, ))
        i = 0
        for line in f:
            nums = line.split(" ")
            xs = list(map(lambda x: idx2word[x], map(int, nums[:6])))
            xs = util.encode_word_sequence(xs, w2v)
            y = int(nums[-1])
            x_train[i] = xs
            y_train[i] = y
            if i >= batch_size-1:
                yield x_train, y_train
                i = 0
            else:
                i += 1

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

    model.fit_generator(generate_ngrams(256, 6, embedding_size, idx2word, w2v), samples_per_epoch=100, epochs=4, callbacks=[checkpoint])

if __name__ == '__main__':
    main()