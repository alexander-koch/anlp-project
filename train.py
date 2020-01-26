import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras import optimizers
from gensim.models import Word2Vec

SEQUENCE_LENGTH = 4
HIDDEN_SIZE = 256

def one_hot_encode(word, word2idx):
    vec = np.zeros((len(word2idx),))
    vec[word2idx[word]] = 1
    return vec

def encode_words(words, word2idx):
    vec = np.zeros((len(words), len(word2idx)))
    for (i,word) in enumerate(words):
        vec[i, word2idx[word]] = 1
    return vec

def prepare_song(song, buffer_length):
    tokens = song + ["<end>"]

    x_train = []
    y_train = []
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            pad_length = (i+buffer_length+1) - len(tokens)
            tokens += ['<pad>'] * pad_length

        x_train.append(tokens[i:i+buffer_length])
        y_train.append(tokens[i+buffer_length])

    return x_train,y_train

def build_model(vocab_size):
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, vocab_size), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
# opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss = 'categorical_crossentropy', optimizer="adam", metrics = ['accuracy'])
    print(model.summary())
    return model

def main():
    words = {'<end>', '<pad>'}
    songs = []
    with open("sentences.txt", "r") as f:
        for line in f.readlines()[:100]:
            tokens = [token for token in line.rstrip().split(" ")]
            songs.append(tokens)
            words = words.union(set(tokens))

    #w2v = Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    #print(songs[0])

    vocab_size = len(words)
    word2idx = { word:i for i,word in enumerate(words) }
    idx2word = { i:word for i,word in enumerate(words) }

    print("Vocab size:", vocab_size)

    x_vec, y_vec = prepare_song(songs[0], SEQUENCE_LENGTH)

    num_samples = len(x_vec)
    x_train = np.zeros((num_samples, SEQUENCE_LENGTH, vocab_size))
    y_train = np.zeros((num_samples, vocab_size))
    for i in range(num_samples):
        x_train[i] = encode_words(x_vec[i], word2idx)
        y_train[i] = one_hot_encode(y_vec[i], word2idx)

    print(x_train.shape)
    print(y_train.shape)

    print(x_train[0])
    print(np.argmax(x_train[0], axis=1))
    print(y_train[0])
    print(np.argmax(y_train[0], axis=0))

    model = build_model(vocab_size)
    model.load_weights("model_4.h5")

    #print(words)

    #model.fit(x_train, y_train, batch_size=128, epochs=600)
    #model.save_weights("model_4.h5")

    words = np.array(["i", "am", "not", "like"])
    words = encode_words(words, word2idx)
    words = words.reshape(1, SEQUENCE_LENGTH, vocab_size)

    #words = x_train[0].reshape(1, SEQUENCE_LENGTH, vocab_size)
    for i in range(SEQUENCE_LENGTH):
        print(idx2word[np.argmax(words[0, i])])

    for i in range(30):
        preds = model.predict(words)
        #print(preds)

        idx = np.argmax(preds)
        word = idx2word[idx]
    # print([idx2word[i] for i in np.argmax(x_train[0], axis=1)])
        print("Next word is", word)

        new_words = np.zeros((1, SEQUENCE_LENGTH, vocab_size))
        new_words[0, 0] = words[0, 1]
        new_words[0, 1] = words[0, 2]
        new_words[0, 2] = words[0, 3]
        new_words[0, 3] = one_hot_encode(word, word2idx)

        words = new_words
    

if __name__ == '__main__':
    main()
