#!/usr/bin/env python

import numpy as np
from gensim.models import KeyedVectors
from typing import List
from pathlib import Path
import pickle

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

def encode_sequence(seq, encode_fn, embedding_size):
    v = np.zeros((len(seq), embedding_size))
    for (i, element) in enumerate(seq):
        v[i] = encode_fn(element)
    return v

def encode_word_sequence(words: List[str], w2v: KeyedVectors):
    encode_fn = lambda word: encode_word(word, w2v)
    return encode_sequence(words, encode_fn, w2v.vector_size+EMBEDDING_EXT)

# def encode_word_sequence(words: List[str], w2v: KeyedVectors):
#     """
#     Encodes a sequence of words into the Word2Vec format.

#     Args:
#         words: List/Iterator of words
#         w2v: Word2Vec instance

#     Returns:
#         Numpy array of encoded words
#     """
#     vec = np.zeros((len(words), w2v.vector_size+EMBEDDING_EXT))
#     for (i,word) in enumerate(words):
#         vec[i] = encode_word(word, w2v)
#     return vec

def load_vocab(path, txt=False):
    if isinstance(path, str):
        path = Path(path)
    vocab = list()
    if txt:
        with path.open("r") as f:
            for line in f.readlines():
                vocab.append(line.rstrip())
    else:
        with path.open("rb") as f:
            vocab = pickle.load(f)
    return vocab

def write_vocab(path, words, txt=False):
    if isinstance(path, str):
        path = Path(path)
    if txt:
        with path.open("w") as f:
            for word in words:
                f.write(word + "\n")
    else:
        words = list(words)
        with path.open("wb") as f:
            pickle.dump(words, f)

def sample(preds, temperature=1.0):
    preds = preds.reshape(preds.shape[1])
    arr = np.asarray(preds).astype('float64')
    log_preds_scaled = np.log(arr) / temperature
    preds_scaled = np.exp(log_preds_scaled)
    softmaxed = preds_scaled / np.sum(preds_scaled)
    probas = np.random.multinomial(1, softmaxed, 1)
    return np.argmax(probas)

def one_hot_encode(word, word2idx):
    v = np.zeros((len(word2idx), ))
    v[word2idx[word]] = 1
    return v

def one_hot_decode(word, idx2word):
    return idx2word[np.argmax(word)]

def one_hot_encode_sequence(words, word2idx):
    encode_fn = lambda word: one_hot_encode(word, word2idx)
    return encode_sequence(words, encode_fn, len(word2idx))

# def one_hot_encode_sequence(words, word2idx):
#     v = np.zeros((len(words), len(word2idx)))
#     for (i, word) in enumerate(words):
#         v[i] = one_hot_encode(word, word2idx)
#     return v

# def one_hot_decode_sequence(vec, idx2word):
#     words = list()
#     for i in range(vec.shape[0]):
#         words.append(one_hot_decode(vec[i], idx2word))
#     return words

def generate_batches(data_length, mini_batch_size):
    for begin in range(0, data_length, mini_batch_size):
        end = min(begin + mini_batch_size, data_length)
        yield begin, end