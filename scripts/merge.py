#!/usr/bin/env python3
from glob import glob
from nltk import word_tokenize
import json
import re

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

def main():
    songs = []
    for path in glob("data/*.json"):
        with open(path, "r") as f:
            data = json.load(f)
            for key in data.keys():
                song = convert_text(data[key])
                songs.append(song)
    
    with open("data/genius_lyrics.txt", "w") as f:
        for tokens in songs:
            f.write(' '.join(tokens) + "\n")


if __name__ == '__main__':
    main()