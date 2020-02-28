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

def convert_token(token):
    token = token.lower()
    if token == "newline":
        token = "<newline>"
    return token

def line2tokens(line):
    return [convert_token(token) for token in word_tokenize(line.rstrip())]

def convert_text(text):
    text = re.sub(r'\n+\s*', ' newline ', text.rstrip())
    return line2tokens(text)

def generate_sentences_pd(path, chunksize, songs):
    for chunk in pd.read_csv(path, chunksize=chunksize, nrows=songs):
        yield chunk['text'].apply(convert_text)

def build_csv():
    vocab = set()
    with open("data/sentences.txt", "w") as f:
        for song in tqdm(generate_sentences_pd("data/songdata.zip", 20, NUM_SONGS), total=NUM_SONGS/20):
            for tokens in song:
                vocab = vocab.union(set(tokens))
                f.write(' '.join(tokens) + "\n")
    with open("vocab_kaggle.pkl", "wb") as f:
        pickle.dump(list(vocab), f)

if __name__ == '__main__':
    build_csv()
