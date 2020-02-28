#!/usr/bin/env python3
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import LeakyReLU
from gensim.models import KeyedVectors
import numpy as np
from nltk import word_tokenize
from typing import List
import util

def build_word_level_model(vocab_size: int, sequence_length: int, embedding_size: int):
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

def build_character_level_model(vocab_size: int, sequence_length: int):
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, vocab_size)))
    model.add(Dropout(0.4))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer="rmsprop", metrics = ['accuracy'])
    return model

class Sampler:
    def __init__(self, model, vocab, encode_fn, sequence_length, embedding_size):
        self.model = model
        self.vocab = vocab
        self.encode_fn = encode_fn
        self.encode_seq_fn = lambda seq: util.encode_sequence(seq, encode_fn, embedding_size)

        self.sequence_length = sequence_length
        self.embedding_size = embedding_size

        self.idx2element = { i:element for i,element in enumerate(vocab) }
        self.element2idx = { element:i for i,element in enumerate(vocab) }

    def _sample(self, seed, num_elements, temperature=1.0):
        element_seq = self.encode_seq_fn(seed).reshape(1, self.sequence_length, self.embedding_size)
        result = list(seed)
        for _ in range(num_elements):
            element = self.idx2element[util.sample(self.model.predict(element_seq), temperature)]
            result.append(element)

            new_element_seq = np.zeros((1, self.sequence_length, self.embedding_size))
            for i in range(self.sequence_length-1):
                new_element_seq[0, i] = element_seq[0, i+1]
            new_element_seq[0, self.sequence_length-1] = self.encode_fn(element)
            element_seq = new_element_seq

        return result

    def proba(self, input_seq, expected_element):
        element_seq = self.encode_seq_fn(input_seq).reshape(1, self.sequence_length, self.embedding_size)
        preds = self.model.predict(element_seq)
        idx = self.element2idx[expected_element]
        return preds[0][idx]

    def sample(self, seed, num_elements, temperature=1.0):
        if len(seed) < self.sequence_length:
            raise Exception(f"Expected at least {self.sequence_length} elements")
        elif len(seed) > self.sequence_length:
            new_seed = seed[len(seed)-self.sequence_length:]
            prefix = seed[:len(seed)-self.sequence_length]
            return list(prefix) + self._sample(new_seed, num_elements, temperature)
        else:
            return self._sample(seed, num_elements, temperature)

    def sample_random(self, num_elements, temperature=1.0):
        idxs = np.random.choice(len(self.vocab), self.sequence_length, replace=True)
        seed = [self.vocab[i] for i in idxs]
        return self._sample(seed, num_elements, temperature)

class WordSampler(Sampler):
    def __init__(self, model, w2v, words, sequence_length, embedding_size):
        self.w2v = w2v
        encode_fn = lambda word: util.encode_word(word, w2v)
        super().__init__(model, words, encode_fn, sequence_length, embedding_size)

    @classmethod
    def from_paths(cls, weights_path, vocab_path, sequence_length):
        w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)
        words = util.load_vocab(vocab_path)
        embedding_size = w2v.vector_size + util.EMBEDDING_EXT

        model = build_word_level_model(len(words), sequence_length, embedding_size)
        model.load_weights(weights_path)
        return cls(model, w2v, words, sequence_length, embedding_size)

    def sample_sent(self, seed: str, num_words: int, temperature=1.0):
        new_seed = word_tokenize(seed)
        return self.sample(new_seed, num_words, temperature)

class CharacterSampler(Sampler):
    def __init__(self, model, chars, sequence_length):        
        char2idx = { char:i for i,char in enumerate(chars) }
        encode_fn = lambda ch: util.one_hot_encode(ch, char2idx)
        super().__init__(model, chars, encode_fn, sequence_length, len(chars))

    @classmethod
    def from_paths(cls, weights_path, vocab_path, sequence_length):
        chars = util.load_vocab(vocab_path)
        model = build_character_level_model(len(chars), sequence_length)
        model.load_weights(weights_path)
        return cls(model, chars, sequence_length)
