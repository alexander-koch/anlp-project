#!/usr/bin/env python3
import pandas as pd
import util
from pathlib import Path
import numpy as np
from tqdm import tqdm

SEQ_LEN = 6
BUFFER_INC = 4096

def build_samples(song, buffer_length):
    tokens = song
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            continue
        yield tokens[i:i+buffer_length+1]

def main():
    df = pd.read_csv("data/songdata.zip")
    path = Path("chars.pkl")
    chars = list()
    if path.is_file():
        chars = util.load_vocab(path)
        print("Loaded from file")
    else:
        vocab = set()
        for song in df["text"]:
            chars = set(song)
            vocab = vocab.union(chars)
        chars = list(vocab)
        util.write_vocab(path, chars)

    vocab_size = len(chars)
    print("Vocab size:", vocab_size)
    char2idx = { char:i for i,char in enumerate(chars) }

    buffer_size = BUFFER_INC
    buffer = np.zeros((buffer_size,SEQ_LEN+1))
    i = 0
    for song in tqdm(df['text']):
        for xs in build_samples(song, SEQ_LEN):
            buffer[i] = [char2idx[x] for x in xs]
            i += 1

            if i >= buffer_size:
                buffer_size += BUFFER_INC
                buffer.resize((buffer_size,SEQ_LEN+1))
    buffer.resize(i, SEQ_LEN+1)
    np.save("chars.npy", buffer)

if __name__ == '__main__':
    main()