#!/usr/bin/env python3
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras import optimizers
from gensim.models import KeyedVectors

SEQUENCE_LENGTH = 4
HIDDEN_SIZE = 256

EMBEDDING_SIZE_ORIG = 100
EMBEDDING_SIZE = 102

def encode_word(word, w2v):
    if word == "<pad>":
        v = np.zeros((EMBEDDING_SIZE,))
        v[EMBEDDING_SIZE-1] = 1
        return v
    elif word == "<newline>":
        v = np.zeros((EMBEDDING_SIZE,))
        v[EMBEDDING_SIZE-2] = 1
        return v
    else:
        v = w2v[word]
        w = np.zeros((2,))
        return np.append(v, w, axis=0)

def encode_words(words, w2v):
    vec = np.zeros((len(words), EMBEDDING_SIZE))
    for (i,word) in enumerate(words):
        vec[i] = encode_word(word, w2v)
    return vec

def prepare_song(song, buffer_length):
    tokens = song# + ["<end>"]

    x_train = []
    y_train = []
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            pad_length = (i+buffer_length+1) - len(tokens)
            tokens += ['<pad>'] * pad_length

        x_train.append(tokens[i:i+buffer_length])
        y_train.append(tokens[i+buffer_length])

    return x_train,y_train

def build_model():
    model = Sequential()
    #model.add(Embedding(vocab_size, 100, SEQUENCE_LENGTH))
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, EMBEDDING_SIZE), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(EMBEDDING_SIZE, activation='softmax'))
# opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss = 'categorical_crossentropy', optimizer="adam", metrics = ['accuracy'])
    print(model.summary())
    return model

def main():
    print("* Reading sentences...")
    words = {'<pad>'}
    songs = []
    with open("sentences.txt", "r") as f:
        for line in f.readlines()[:100]:
            tokens = [token for token in line.rstrip().split(" ")]
            songs.append(tokens)
            words = words.union(set(tokens))

    print("* Loading model...")
    w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.txt.word2vec', binary=False)

    print("* Testing...")
    vec = encode_word("world", w2v)
    print(vec)
    print(vec.shape)

    #print(songs[0])

    #vocab_size = len(words)
    #word2idx = { word:i for i,word in enumerate(words) }
    #idx2word = { i:word for i,word in enumerate(words) }

    #print("Vocab size:", vocab_size)

    print("* Preparing data...")

    x_vec, y_vec = prepare_song(songs[0], SEQUENCE_LENGTH)

    print("* Encoding...")

    num_samples = len(x_vec)
    x_train = np.zeros((num_samples, SEQUENCE_LENGTH, EMBEDDING_SIZE))
    y_train = np.zeros((num_samples, EMBEDDING_SIZE))
    for i in range(num_samples):
        x_train[i] = encode_words(x_vec[i], w2v)
        y_train[i] = encode_word(y_vec[i], w2v)

    print("* Done")

    print(x_train.shape)
    print(y_train.shape)

    print(x_train[0])
    print(np.argmax(x_train[0], axis=1))
    print(y_train[0])
    print(np.argmax(y_train[0], axis=0))


if __name__ == '__main__':
    main()
