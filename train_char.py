#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from random import shuffle
import util

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import LeakyReLU
from pathlib import Path
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop

SEQUENCE_LENGTH = 8
BATCH_SIZE = 256

checkpoint = ModelCheckpoint("weights/weights_char_{epoch:01d}.h5",
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

def build_model(vocab_size):
    model = Sequential()
    model.add(LSTM(512, input_shape=(SEQUENCE_LENGTH, vocab_size)))
    model.add(Dropout(0.4))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer="rmsprop", metrics = ['accuracy'])
    return model

def build_samples(song, buffer_length):
    tokens = song

    x_train = []
    y_train = []
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            continue
            
        x_train.append(tokens[i:i+buffer_length])
        y_train.append(tokens[i+buffer_length])

    return x_train,y_train

def generate_batches(songs, batch_size):
    x_train, y_train = [], []
    for song in songs:
        xs, ys = build_samples(song, SEQUENCE_LENGTH)
        x_train.extend(xs)
        y_train.extend(ys)
        if len(x_train) >= batch_size:
            yield x_train[0:batch_size], y_train[0:batch_size]
            x_train = x_train[batch_size:]
            y_train = y_train[batch_size:]
    if len(x_train) > 0:
        yield x_train, y_train

def generate_samples(songs, batch_size, vocab_size, char2idx):
    while True:
        batches = generate_batches(songs, batch_size)
        for xs_batch, ys_batch in batches:
            c = list(zip(xs_batch, ys_batch))
            shuffle(c)
            xs_batch, ys_batch = zip(*c)

            batch_size = len(xs_batch)
            x_train = np.zeros((batch_size, SEQUENCE_LENGTH, vocab_size))
            y_train = np.zeros((batch_size, vocab_size))

            for i in range(batch_size):
                x_train[i] = util.one_hot_encode_sequence(xs_batch[i], char2idx)
                y_train[i] = util.one_hot_encode(ys_batch[i], char2idx)

            yield x_train, y_train

def main():
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
    char2idx = { char:i for i,char in enumerate(chars) }
    
    model = build_model(vocab_size)
    model.fit_generator(generate_samples(df['text'], BATCH_SIZE, vocab_size, char2idx), samples_per_epoch=300, epochs=10, callbacks=[checkpoint])

if __name__ == '__main__':
    main()