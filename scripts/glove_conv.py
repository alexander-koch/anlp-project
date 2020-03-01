#!/usr/bin/env python3
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

print("Converting glove txt -> word2vec txt...")
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

print("Converting word2vec txt -> word2vec bin...")
w2v = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
w2v.save_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)