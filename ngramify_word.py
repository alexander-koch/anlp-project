#!/usr/bin/env python3
import util
from tqdm import tqdm
import numpy as np
import argparse

SRC_PATH = "data/sentences.txt"
BUFFER_INC = 4096
DEFAULT_SEQ_LEN = 6

def ngramify(song, buffer_length, word2idx):
    """
    Builds n-grams with one lookahead token.

    Args:
        song: List of tokens
        buffer_length: Number of tokens to keep in buffer
        word2idx: Mapping from a word to an index

    Yields:
        Tuple of buffer_length tokens and a lookahead token
    """
    tokens = song
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            continue
            
        xs = tokens[i:i+buffer_length]
        y = tokens[i+buffer_length]
        discard = False
        for x in xs:
            if x not in word2idx:
                discard = True
                break
        if discard or y not in word2idx:
            continue
            
        yield xs, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input vocabulary", required=True)
    parser.add_argument("-s", "--seqlen", help="Sequence length")
    parser.add_argument("-o", "--output", help="Output .npy path", required=True)
    args = parser.parse_args()

    if not args.seqlen:
        seq_len = DEFAULT_SEQ_LEN
    else:
        seq_len = int(args.seqlen)
    
    print("Reading songs...")
    songs = None
    with open(SRC_PATH, "r") as f:
        songs = [line.rstrip().split(" ") for line in f]

    print("Loading vocab...")
    words = util.load_vocab(args.input)
    word2idx = { word:i for i,word in enumerate(words) }
    buffer_size = BUFFER_INC

    print("Generating ngrams...")
    buffer = np.zeros((buffer_size,seq_len+1), dtype=np.int64)
    i = 0
    for song in tqdm(songs):
        for xs, y in ngramify(song, seq_len, word2idx):
            xs = [word2idx[x] for x in xs]
            y = word2idx[y]
            xs.append(y)
            buffer[i] = xs
            i += 1

            if i >= buffer_size:
                buffer_size += BUFFER_INC
                buffer.resize((buffer_size,seq_len+1))
    buffer.resize(i, seq_len+1)
    print("Saving to {}...".format(args.output))
    np.save(args.output, buffer)

if __name__ == '__main__':
    main()