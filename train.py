import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
import numpy as np
from random import shuffle

EMBEDDING_EXT = 3

def encode_word(word: str, w2v: KeyedVectors):
    """
    Encodes a word as a Word2Vec vector.
    Increases the dimensionality of the vector by three,
    to store the tokens <pad>, <newline> and <unk>.
    
    Args:
        word: Word to encode
        w2v: Word2Vec instance

    Returns:
        Word2Vec vector
    """
    embedding_size = w2v.vector_size+EMBEDDING_EXT

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
    """
    Encodes a sequence of words into the Word2Vec format.

    Args:
        words: List/Iterator of words
        w2v: Word2Vec instance

    Returns:
        Numpy array of encoded words
    """
    vec = np.zeros((len(words), w2v.vector_size+EMBEDDING_EXT))
    for (i,word) in enumerate(words):
        vec[i] = encode_word(word, w2v)
    return vec

def decode_vec(vec: np.ndarray, w2v: KeyedVectors):
    """
    Decodes a vector into the corresponding word

    Args:
        vec: Vector representation of word
        w2v: Word2Vec instance

    Returns:
        The word as a string
    """
    embedding_size = w2v.vector_size

    base_vec = vec[:embedding_size]
    ext_vec = vec[embedding_size:]
    if ext_vec[0]:
        return "<unk>"
    elif ext_vec[1]:
        return "<newline>"
    elif ext_vec[2]:
        return "<pad>"
    else:
        return w2v.similar_by_vector(base_vec)[0][0]

def build_ngrams(song, buffer_length):
    """Builds ngram training data."""
    tokens = song

    x_train = []
    y_train = []
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            pad_length = (i+buffer_length+1) - len(tokens)
            tokens += ['<pad>'] * pad_length

        x_train.append(tokens[i:i+buffer_length])
        y_train.append(tokens[i+buffer_length])

    return x_train,y_train

def load_songs_dataset(path, sequence_length):
    """Loads song training data in n-gram format.
    Expects a text file where every line contains one song.
    The song must have been tokenized before.
    """
    token_vocab = {'<pad>', '<unk>'}
    x_vec = []
    y_vec = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # Tokenize the line, append to vocab
            tokens = [token for token in line.rstrip().split(" ")]
            token_vocab = token_vocab.union(set(tokens))

            x_vec_i, y_vec_i = build_ngrams(tokens, sequence_length)
            x_vec.extend(x_vec_i)
            y_vec.extend(y_vec_i)
    return x_vec, y_vec, token_vocab

def one_hot_encode(word: str, word2idx: dict):
    """One-hot encodes a word based on a given dictionary"""
    v = np.zeros((len(word2idx, )), dtype=np.int64)
    v[word2idx[word]] = 1
    return v

def one_hot_decode(vec: np.ndarray, idx2word: dict):
    return idx2word[np.argmax(vec)]

def generate_batches(data_length, mini_batch_size):
    for begin in range(0, data_length, mini_batch_size):
        end = min(begin + mini_batch_size, data_length)
        yield begin, end

def load_batch(xs, ys, begin, end, w2v, word2idx):
    batch_size = end-begin
    if batch_size <= 0:
        raise ValueError

    embedding_size = w2v.vector_size+EMBEDDING_EXT
    vocab_size = len(word2idx)

    xs_batch = xs[begin:end]
    ys_batch = ys[begin:end]

    sequence_length = len(xs_batch[0])
    
    x_train = np.zeros((batch_size, sequence_length, embedding_size), dtype=np.float64)
    y_train = np.zeros((batch_size,), dtype=np.int64)
    
    c = list(zip(xs_batch, ys_batch))
    shuffle(c)
    xs_batch, ys_batch = zip(*c)
    
    for i in range(batch_size):
        x_train[i] = encode_words(xs_batch[i], w2v)
        y_train[i] = one_hot_encode(ys_batch[i], word2idx)
    
    return x_train, y_train

class LyricsModel(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(LyricsModel, self).__init__()

        self.hidden_size_1 = 128
        self.hidden_size_2 = 1024
        self.hidden_size_3 = 2048
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.gru_1 = nn.GRU(self.embedding_size, self.hidden_size_1, batch_first=True)
        self.dropout_1 = nn.Dropout(p=0.4)
        self.gru_2 = nn.GRU(self.hidden_size_1, self.hidden_size_2, batch_first=True)
        self.dropout_2 = nn.Dropout(p=0.4)
        self.linear_1 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.relu = nn.LeakyReLU()
        self.linear_2 = nn.Linear(self.hidden_size_1, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence):
        hidden1 = torch.zeros((1, sequence.shape[0], self.hidden_size_1))
        #hidden2 = torch.zeros((1, sequence.shape[0], self.hidden_size_2))

        x, h = self.gru_1(sequence, hidden1)
        #x = self.dropout_1(x)
        #_, h = self.gru_2(x, hidden2)
        #x = self.dropout_2(h[-1])
        #x = self.linear_1(x)
        #x = self.relu(x)
        #print("X:",x.shape)
        #print("H:",h.shape)
        #print("Last X:", x[:,-1,:].shape)

        x = self.linear_2(x[:,-1,:])
        return self.softmax(x)

SEQUENCE_LENGTH = 6
BATCH_SIZE = 256

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import LeakyReLU
def build_keras_model(vocab_size, embedding_size):
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, embedding_size), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(1024))
    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer="rmsprop", metrics = ['accuracy'])
    return model

def main():
    #w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)

    # Load song dataset
    x_train, y_train, vocab = load_songs_dataset("data/sentences.txt", SEQUENCE_LENGTH)
    vocab_size = len(vocab)
    print(x_train[0], y_train[0])
    print("Vocab size:", vocab_size)

    # Create index mapping for output layer
    words = list(vocab)
    word2idx = { word:i for i,word in enumerate(words) }
    idx2word = { i:word for i,word in enumerate(words) }
    
    with open("vocab.txt", "w") as f:
        for word in words:
            f.write(word + "\n")


    # Generate the model
    # embedding_size = w2v.vector_size+EMBEDDING_EXT

    # model = build_keras_model(vocab_size, embedding_size)
    # for (i, (begin, end)) in enumerate(generate_batches(len(x_train), BATCH_SIZE)):
    #     x_batch, y_batch = load_batch(x_train, y_train, begin, end, w2v, word2idx)

    #     model.fit(x_batch, y_batch, epochs=10)

def main2():
    w2v = KeyedVectors.load_word2vec_format('glove.6B.100d.bin.word2vec', binary=True)

    # Load song dataset
    x_train, y_train, vocab = load_songs_dataset("data/sentences.txt", SEQUENCE_LENGTH)
    vocab_size = len(vocab)
    print(x_train[0], y_train[0])
    print("Vocab size:", vocab_size)

    # Create index mapping for output layer
    words = list(vocab)
    word2idx = { word:i for i,word in enumerate(words) }
    idx2word = { i:word for i,word in enumerate(words) }

    # Generate the model
    embedding_size = w2v.vector_size+EMBEDDING_EXT
    model = LyricsModel(embedding_size, vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Big Brain Time!
    for epoch in range(100):

        running_loss = 0.0
        for (i, (begin, end)) in enumerate(generate_batches(len(x_train), BATCH_SIZE)):
            optimizer.zero_grad()
            x_batch, y_batch = load_batch(x_train, y_train, begin, end, w2v, word2idx)
            
            x_batch = torch.FloatTensor(x_batch)
            y_batch = torch.LongTensor(y_batch)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            running_loss += loss.item()

if __name__ == '__main__':
    main()