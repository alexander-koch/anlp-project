#!/usr/bin/env python3
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import LeakyReLU
from gensim.models import KeyedVectors
import numpy as np
from nltk import word_tokenize
from typing import List
import util

def build_keras_model(vocab_size: int, sequence_length: int, embedding_size: int):
    model = Sequential()
    model.add(LSTM(128, input_shape=(sequence_length, embedding_size), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(1024))
    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer="rmsprop", metrics = ['accuracy'])
    return model

class Sampler:
    def __init__(self, weights_path, vocab_path, sequence_length):
        self.w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)
        self.words = util.load_vocab(vocab_path)
        self.vocab_size = len(self.words)
        self.embedding_size = self.w2v.vector_size + util.EMBEDDING_EXT
        self.sequence_length = sequence_length

        self.model = build_keras_model(self.vocab_size, self.sequence_length, self.embedding_size)
        self.model.load_weights(weights_path)
        self.idx2word = { i:word for i,word in enumerate(self.words) }

    def _sample(self, seed, num_words, temperature):
        words_seq = util.encode_words(seed, self.w2v).reshape(1, self.sequence_length, self.embedding_size)
        result = seed.copy()
        for j in range(num_words):
            word = self.idx2word[util.sample(self.model.predict(words_seq), temperature)]
            result.append(word)

            new_words = np.zeros((1, self.sequence_length, self.embedding_size))
            for i in range(self.sequence_length-1):
                new_words[0, i] = words_seq[0, i+1]
            new_words[0, self.sequence_length-1] = util.encode_word(word, self.w2v)
            words_seq = new_words

        return result

    def sample(self, seed: List[str], num_words: int, temperature=1.0):
        if len(seed) < self.sequence_length:
            raise Exception(f"Expected at least {self.sequence_length} words")
        elif len(seed) > self.sequence_length:
            new_seed = seed[len(seed)-self.sequence_length:]
            prefix = seed[:len(seed)-self.sequence_length]
            return prefix + self._sample(new_seed, num_words, temperature)
        else:
            return self._sample(seed, num_words, temperature)

    def sample_sent(self, seed: str, num_words: int, temperature=1.0):
        new_seed = word_tokenize(seed)
        return self.sample(new_seed, num_words, temperature)

    def sample_random(self, num_words, temperature=1.0):
        idxs = np.random.choice(self.vocab_size, self.sequence_length, replace=True)
        seed = [self.words[i] for i in idxs]
        return self._sample(seed, num_words, temperature)