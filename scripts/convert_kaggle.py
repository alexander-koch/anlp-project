#!/usr/bin/env python3
import csv
from nltk import word_tokenize
import re
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
import itertools
import pickle

NUM_SONGS = 57650
SRC_PATH = "data/songdata.zip"
OUTPUT_PATH_TOKENS = "data/sentences.txt"
OUTPUT_PATH_VOCAB = "vocab_kaggle.pkl"

def convert_token(token):
    token = token.lower()
    if token == "newline":
        token = "<newline>"
    return token

def line2tokens(line):
    """
    Removes trailing whitespace and tokenizes the input.
    Every input token is then converted to lowercase

    Args:
        line: Input text

    Returns:
        List of tokens
    """
    return [convert_token(token) for token in word_tokenize(line.rstrip())]

def convert_text(text):
    """
    Replaces one or multiple newlines by the token 'newline',
    removes trailing whitespace and tokenizes the whole text using line2tokens.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    text = re.sub(r'\n+\s*', ' newline ', text.rstrip())
    return line2tokens(text)

def generate_sentences_pd(path, chunksize, songs):
    """
    Reads a pandas dataframe in chunks.
    Only reads songs number of rows and converts the text column into tokens.

    Args:
        path: Path to the csv file
        chunksize: Chunksize to keep in memory
        songs: Number of songs to read from the csv

    Yields:
        Converted tokens per song
    """
    for chunk in pd.read_csv(path, chunksize=chunksize, nrows=songs):
        yield chunk['text'].apply(convert_text)

def build_csv():
    print("Tokenizing songs...")
    vocab = set()
    with open(OUTPUT_PATH_TOKENS, "w") as f:
        for song in tqdm(generate_sentences_pd(SRC_PATH, 20, NUM_SONGS), total=NUM_SONGS/20):
            for tokens in song:
                vocab = vocab.union(set(tokens))
                f.write(' '.join(tokens) + "\n")
    print("Writing vocab to {}...".format(OUTPUT_PATH_VOCAB))
    with open(OUTPUT_PATH_VOCAB, "wb") as f:
        pickle.dump(list(vocab), f)

if __name__ == '__main__':
    build_csv()
