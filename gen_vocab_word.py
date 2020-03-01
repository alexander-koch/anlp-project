#!/usr/bin/env python3
from random import random
import math
import util
import argparse

SRC_PATH = "data/sentences.txt"
SRC_VOCAB_PATH = "vocab_kaggle.pkl"
DEFAULT_THRESH = 1e-4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", help="Threshold to drop words")
    parser.add_argument("-o", "--output", help="Output .pkl path", required=True)
    args = parser.parse_args()

    if not args.threshold:
        threshold = DEFAULT_THRESH
    else:
        threshold = float(args.threshold)
    print("Threshold:", threshold)

    print("Calculating word frequencies...")
    freqs = dict()
    with open(SRC_PATH, "r") as f:
        for line in f:
            for token in line.rstrip().split(" "):
                if token not in freqs:
                    freqs[token] = 1
                else:
                    freqs[token] += 1

    total_words = len(freqs.keys())
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
    util.write_vocab(args.output, new_words)
    print("Vocab written to:", args.output)

if __name__ == '__main__':
    main()