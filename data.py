import csv
from nltk import word_tokenize
import re
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
import itertools

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
    with open("data/sentences.txt", "w") as f:
        for song in tqdm(generate_sentences_pd("data/songdata.csv", 20, 400), total=400/20):
            for tokens in song:
                f.write(' '.join(tokens) + "\n")

# def build_w2v():
#     model = Word2Vec()
#     update = True
#
#     for songs in tqdm(generate_sentences_pd("songdata.csv", 20, 400), total=400/20):
#         sentences = list(itertools.chain.from_iterable(songs.values))
#         if update:
#             model.build_vocab(sentences)
#             update = False
#         else:
#             model.build_vocab(sentences, update=True)
#         model.train(sentences, total_examples=len(sentences), epochs=80)
#
#     print(model)
#     model.save("model.bin")

if __name__ == '__main__':
    build_csv()
