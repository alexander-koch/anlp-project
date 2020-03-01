#!/usr/bin/env python3
import pandas as pd
import util
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse

DEFAULT_SEQ_LEN = 8
BUFFER_INC = 4096
SRC_PATH = "data/songdata.zip"

def build_samples(song, buffer_length):
    """
    Builds samples from characters.

    Args:
        song: String of characters
        buffer_length: Number of characters to keep in a sequence
    
    Yields:
        String of characters with length buffer_length
    """

    tokens = song
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            continue
        yield tokens[i:i+buffer_length+1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seqlen", help="Sequence length")
    parser.add_argument("-o", "--output", help="Output .npy path", required=True)
    args = parser.parse_args()

    if not args.seqlen:
        seq_len = DEFAULT_SEQ_LEN
    else:
        seq_len = int(args.seqlen)

    df = pd.read_csv(SRC_PATH)
    path = Path("chars.pkl")
    chars = list()
    if path.is_file():
        chars = util.load_vocab(path)
        print("Loaded vocabulary from file")
    else:
        vocab = set()
        for song in df["text"]:
            chars = set(song)
            vocab = vocab.union(chars)
        chars = list(vocab)
        util.write_vocab(path, chars)
        print("Generated character vocabulary as chars.pkl")

    vocab_size = len(chars)
    print("Vocab size:", vocab_size)
    char2idx = { char:i for i,char in enumerate(chars) }

    print("Generating training samples...")
    buffer_size = BUFFER_INC
    buffer = np.zeros((buffer_size,seq_len+1), dtype=np.int64)
    i = 0
    for song in tqdm(df['text']):
        for xs in build_samples(song, seq_len):
            buffer[i] = [char2idx[x] for x in xs]
            i += 1

            if i >= buffer_size:
                buffer_size += BUFFER_INC
                buffer.resize((buffer_size,seq_len+1))
    buffer.resize(i, seq_len+1)
    print("Saving to {}...".format(args.output))
    np.save(args.output, buffer)

if __name__ == '__main__':
    main()