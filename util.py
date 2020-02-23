#!/usr/bin/env python

import numpy as np
from gensim.models import KeyedVectors
from typing import List
from pathlib import Path

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

def encode_words(words: List[str], w2v: KeyedVectors):
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

def load_vocab(path):
    if isinstance(path, str):
        path = Path(path)
    vocab = list()
    with path.open("r") as f:
        for line in f.readlines():
            vocab.append(line.rstrip())
    return vocab

def write_vocab(path, words):
    if isinstance(path, str):
        path = Path(path)
    with path.open("w") as f:
        for word in words:
            f.write(word + "\n")

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
    v = np.zeros((len(words), len(word2idx)))
    for (i, word) in enumerate(words):
        v[i] = one_hot_encode(word, word2idx)
    return v

def one_hot_decode_sequence(vec, idx2word):
    words = list()
    for i in range(vec.shape[0]):
        words.append(one_hot_decode(vec[i], idx2word))
    return words