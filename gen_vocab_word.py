#!/usr/bin/env python3
from random import random
import math
import util

SRC_VOCAB_PATH = "vocab_kaggle.pkl"
DST_VOCAB_PATH = "small_vocab.pkl"

def main():
    print("Calculating word frequencies...")
    freqs = dict()
    with open("data/sentences.txt", "r") as f:
        for line in f:
            for token in line.rstrip().split(" "):
                if token not in freqs:
                    freqs[token] = 1
                else:
                    freqs[token] += 1

    total_words = len(freqs.keys())
    threshold = 1e-4
    discard = set()
    for word in freqs.keys():
        z = freqs[word] / total_words
        p = (math.sqrt(z / threshold) + 1) * (threshold / z)
        if random() <= p:
            discard.add(word)

    print("Total words:", total_words)
    print("Discarded words:", len(discard))
    print("Target vocab size:", total_words - len(discard))

    words = util.load_vocab(SRC_VOCAB_PATH)
    new_words = list(set(words).difference(discard))
    util.write_vocab(DST_VOCAB_PATH, new_words)
    print("Vocab written to:", DST_VOCAB_PATH)

if __name__ == '__main__':
    main()