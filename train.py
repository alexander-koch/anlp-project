import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import numpy as np

def encode_word(word, w2v):
    embedding_size = w2v.vector_size+3

    if word == "<pad>":
        v = np.zeros((embedding_size,))
        v[embedding_size-1] = 1
        return v
    elif word == "<newline>":
        v = np.zeros((embedding_size,))
        v[embedding_size-2] = 1
        return v
    elif word == "<unk>" or word not in w2v:
        v = np.zeros((embedding_size,))
        v[embedding_size-3] = 1
        return v
    else:        
        v = w2v[word]
        w = np.zeros((3,))
        return np.append(v, w, axis=0)

def encode_words(words, w2v):
    vec = np.zeros((len(words), w2v.vector_size))
    for (i,word) in enumerate(words):
        vec[i] = encode_word(word, w2v)
    return vec

class LyricsModel(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(LyricsModel, self).__init__()

        self.hidden_size_1 = 128
        self.hidden_size_2 = 1024
        self.hidden_size_3 = 2048
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        
        # Load weights from pre-trained GloVe vectors as embedding weights
        #weights = torch.FloatTensor(w2v.vectors)
        #self.embedding = nn.Embedding.from_pretrained(weights)
        #self.embedding.weight.requires_grad = False

        self.gru_1 = nn.GRU(self.embedding_size, self.hidden_size_1)
        self.dropout_1 = nn.Dropout(p=0.4)
        self.gru_2 = nn.GRU(self.hidden_size_1, self.hidden_size_2)
        self.dropout_2 = nn.Dropout(p=0.4)
        self.linear_1 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.relu = nn.LeakyReLU()
        self.linear_2 = nn.Linear(self.hidden_size_3, vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, sequence):
        x = self.gru_1(sequence)
        x = self.dropout_1(x)
        x = self.gru_2(x)
        x = self.dropout_2(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return self.softmax(x)

def main():
    w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)
    embedding_size = w2v.vector_size

    model = LyricsModel(embedding_size, 5000)


if __name__ == '__main__':
    main()