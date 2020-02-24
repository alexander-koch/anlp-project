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

class WordSampler:
    # def __init__(self, weights_path, vocab_path, sequence_length):
    #     self.w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)
    #     self.words = util.load_vocab(vocab_path)
    #     self.vocab_size = len(self.words)
    #     self.embedding_size = self.w2v.vector_size + util.EMBEDDING_EXT
    #     self.sequence_length = sequence_length

    #     self.model = build_keras_model(self.vocab_size, self.sequence_length, self.embedding_size)
    #     self.model.load_weights(weights_path)
    #     self.idx2word = { i:word for i,word in enumerate(self.words) }

    def __init__(self, model, w2v, words, sequence_length):
        self.model = model
        self.w2v = w2v
        self.words = words
        self.sequence_length = sequence_length
        self.vocab_size = len(self.words)
        self.embedding_size = self.w2v.vector_size + util.EMBEDDING_EXT
        self.idx2word = { i:word for i,word in enumerate(self.words) }

    @classmethod
    def from_paths(cls, weights_path, vocab_path, sequence_length):
        w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)
        words = util.load_vocab(vocab_path)
        embedding_size = w2v.vector_size + util.EMBEDDING_EXT

        model = build_keras_model(len(words), sequence_length, embedding_size)
        model.load_weights(weights_path)
        return cls(model, w2v, words, sequence_length)

    def _sample(self, seed, num_words, temperature):
        words_seq = util.encode_word_sequence(seed, self.w2v).reshape(1, self.sequence_length, self.embedding_size)
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

class CharacterSampler:
    def __init__(self, model, chars, sequence_length):
        self.model = model
        self.chars = chars
        self.sequence_length = sequence_length
        self.vocab_size = len(chars)
        
        self.char2idx = { char:i for i,char in enumerate(chars) }
        self.idx2char = { i:char for i,char in enumerate(chars) }

    def _sample(self, seed: str, num_chars: int, temperature=1.0):
        text = seed + ""
        for i in range(num_chars):
            enc_seq = util.one_hot_encode_sequence(seed, self.char2idx).reshape(1, self.sequence_length, self.vocab_size)
            preds = self.model.predict(enc_seq)
            next_char = self.idx2char[util.sample(preds, temperature=temperature)]
            text += next_char
            seed = seed[1:]
            seed += next_char
        return text

    def sample(self, seed: str, num_chars: int, temperature=1.0):
        if len(seed) < self.sequence_length:
            raise Exception(f"Expected at least {self.sequence_length} characters")
        elif len(seed) > self.sequence_length:
            new_seed = seed[len(seed)-self.sequence_length:]
            prefix = seed[:len(seed)-self.sequence_length]
            return prefix + self._sample(new_seed, num_chars, temperature)
        else:
            return self._sample(seed, num_chars, temperature)

    def sample_random(self, num_chars: int, temperature=1.0):
        idxs = np.random.choice(self.vocab_size, self.sequence_length, replace=True)
        seed = [self.chars[i] for i in idxs]
        return self._sample(seed, num_chars, temperature)