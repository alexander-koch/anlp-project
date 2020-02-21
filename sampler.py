#!/usr/bin/env python3
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import LeakyReLU
from gensim.models import KeyedVectors
import numpy as np
from nltk import word_tokenize
from typing import List

EMBEDDING_EXT = 3

def encode_word(word: str, w2v: KeyedVectors):
    """
    Encodes a word as a Word2Vec vector.
    Increases the dimensionality of the vector by three,
    to store the tokens <pad>, <newline> and <unk>.
    
    Args:
        word: Word to encode
        w2v: Word2Vec instance

    Returns:
        Word2Vec vector
    """
    embedding_size = w2v.vector_size+EMBEDDING_EXT

    if word == "<pad>":
        v = np.zeros((embedding_size,))
        v[embedding_size-1] = 1
        return v
    elif word == "<newline>":
        v = np.zeros((embedding_size,))
        v[embedding_size-2] = 1
        return v
    elif word == "<unk>" or word not in w2v:
        v = np.zeros((embedding_size,))
        v[embedding_size-3] = 1
        return v
    else:        
        v = w2v[word]
        w = np.zeros((3,))
        return np.append(v, w, axis=0)

def encode_words(words, w2v):
    """
    Encodes a sequence of words into the Word2Vec format.

    Args:
        words: List/Iterator of words
        w2v: Word2Vec instance

    Returns:
        Numpy array of encoded words
    """
    vec = np.zeros((len(words), w2v.vector_size+EMBEDDING_EXT))
    for (i,word) in enumerate(words):
        vec[i] = encode_word(word, w2v)
    return vec

def build_keras_model(vocab_size: int, embedding_size: int):
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, embedding_size), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(1024))
    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer="rmsprop", metrics = ['accuracy'])
    return model

def sample(preds, temperature=1.0):
    preds = preds.reshape(preds.shape[1])
    arr = np.asarray(preds).astype('float64')
    log_preds_scaled = np.log(arr) / temperature
    preds_scaled = np.exp(log_preds_scaled)
    softmaxed = preds_scaled / np.sum(preds_scaled)
    probas = np.random.multinomial(1, softmaxed, 1)
    return np.argmax(probas)

SEQUENCE_LENGTH = 6
EMBEDDING_SIZE = 103

def load_vocab(path):
    vocab = list()
    with open(path, "r") as f:
        for line in f.readlines():
            vocab.append(line.rstrip())
    return vocab

class Sampler:
    def __init__(self, weights_path, vocab_path):
        self.w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)
        self.words = load_vocab(vocab_path)
        self.vocab_size = len(self.words)

        self.model = build_keras_model(self.vocab_size, EMBEDDING_SIZE)
        self.model.load_weights(weights_path)
        self.idx2word = { i:word for i,word in enumerate(self.words) }

    def _sample(self, seed, num_words, temperature):
        words_seq = encode_words(seed, self.w2v).reshape(1, SEQUENCE_LENGTH, EMBEDDING_SIZE)
        result = seed.copy()
        for j in range(num_words):
            word = self.idx2word[sample(self.model.predict(words_seq), temperature)]
            result.append(word)

            new_words = np.zeros((1, SEQUENCE_LENGTH, EMBEDDING_SIZE))
            for i in range(SEQUENCE_LENGTH-1):
                new_words[0, i] = words_seq[0, i+1]
            new_words[0, SEQUENCE_LENGTH-1] = encode_word(word, self.w2v)
            words_seq = new_words

        return result

    def sample(self, seed: List[str], num_words: int, temperature=1.0):
        if len(seed) < SEQUENCE_LENGTH:
            raise Exception(f"Expected at least {SEQUENCE_LENGTH} words")
        elif len(seed) > SEQUENCE_LENGTH:
            new_seed = seed[len(seed)-SEQUENCE_LENGTH:]
            prefix = seed[:len(seed)-SEQUENCE_LENGTH]
            return prefix + self._sample(new_seed, num_words, temperature)
        else:
            return self._sample(seed, num_words, temperature)

    def sample_sent(self, seed: str, num_words: int, temperature=1.0):
        new_seed = word_tokenize(seed)
        return self.sample(new_seed, num_words, temperature)

    def sample_random(self, num_words, temperature=1.0):
        idxs = np.random.choice(self.vocab_size, SEQUENCE_LENGTH, replace=True)
        seed = [self.words[i] for i in idxs]
        return self._sample(seed, num_words, temperature)