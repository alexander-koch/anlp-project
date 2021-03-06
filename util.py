import numpy as np
from gensim.models import KeyedVectors
from typing import List
from pathlib import Path
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer

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

def encode_sequence(seq, encode_fn, embedding_size):
    """
    Encodes a sequence of elements using an encoding function
    that uses the specified embedding space.

    Args:
        seq: Sequence to encode
        encode_fn: Function that takes a word and returns the encoded word
        embedding_size: Size of the embedding used by the encoding function

    Returns:
        Numpy array with dimensionality (sequence_length, embedding_size)
    """
    v = np.zeros((len(seq), embedding_size))
    for (i, element) in enumerate(seq):
        v[i] = encode_fn(element)
    return v

def decode_sequence(seq, decode_fn):
    """
    Decodes a sequence of elements using the decoding function.

    Args:
        seq: Sequence of elements
        decode_fn: Function to decode a single element

    Returns:
        List of decoded elements
    """
    elements = list()
    for i in range(seq.shape[0]):
        elements.append(decode_fn(seq[i]))
    return elements

def encode_word_sequence(words: List[str], w2v: KeyedVectors):
    """
    Encodes a sequence of words using encode_word.

    Args:
        words: List of words
        w2v: Word2Vec

    Returns:
        Numpy array of encoded sequence (sequence_length, w2v_vector+EMBEDDING_EXT)
    """

    encode_fn = lambda word: encode_word(word, w2v)
    return encode_sequence(words, encode_fn, w2v.vector_size+EMBEDDING_EXT)

def decode_vector(v, w2v):
    base_vec = v[:w2v.vector_size]
    ext_vec = v[w2v.vector_size:]
    if ext_vec[0]:
        return "<unk>"
    elif ext_vec[1]:
        return "<newline>"
    elif ext_vec[2]:
        return "<pad>"
    else:
        return w2v.similar_by_vector(base_vec)[0][0]

def load_vocab(path, txt=False):
    """
    Loads a vocab file from a path.
    The vocabulary consists of a list of word/characters.
    Args:
        path: Path to load vocabulary from
        txt: Whether to treat the file as a text file
    Returns:
        List of items in vocabulary
    """
    if isinstance(path, str):
        path = Path(path)
    vocab = list()
    if txt:
        with path.open("r") as f:
            for line in f.readlines():
                vocab.append(line.rstrip())
    else:
        with path.open("rb") as f:
            vocab = pickle.load(f)
    return vocab

def write_vocab(path, words, txt=False):
    """
    Writes a vocabulary to a path.

    Args:
        path: Path to write vocabulary to
        words: List of elements in the vocabulary
        txt: Write as text file or use the pickle library
    """

    if isinstance(path, str):
        path = Path(path)
    if txt:
        with path.open("w") as f:
            for word in words:
                f.write(word + "\n")
    else:
        words = list(words)
        with path.open("wb") as f:
            pickle.dump(words, f)

def sample(preds, temperature=1.0):
    """
    Samples a word from a probability distribution using temperature sampling.

    Args:
        preds: Probability distribution
        temperature: Temperature scalar to divide the values by
    Returns:
        Random index based on distribution and temperature
    """
    preds = preds.reshape(preds.shape[1])
    arr = np.asarray(preds).astype('float64')
    log_preds_scaled = np.log(arr) / temperature
    preds_scaled = np.exp(log_preds_scaled)
    softmaxed = preds_scaled / np.sum(preds_scaled)
    probas = np.random.multinomial(1, softmaxed, 1)
    return np.argmax(probas)

def one_hot_encode(word, word2idx):
    """
    One-hot encodes a word using a mapping.

    Args:
        word: Word to encode
        word2idx: Mapping from a word to an index
    Returns:
        Numpy array with the corresponding bit set to one
    """

    v = np.zeros((len(word2idx), ))
    v[word2idx[word]] = 1
    return v

def one_hot_decode(word, idx2word):
    """
    Decodes the one-hot encoded vector.

    Args:
        word: Word vector to decode
        idx2word: Mapping from an index to the corresponding word
    Returns:
        Word of the encoding
    """
    return idx2word[np.argmax(word)]

def one_hot_encode_sequence(words, word2idx):
    encode_fn = lambda word: one_hot_encode(word, word2idx)
    return encode_sequence(words, encode_fn, len(word2idx))

def generate_batches(data_length, mini_batch_size):
    """
    Generates mini-batch indices based on a given data length.

    Args:
        data_length: Total length of the data
        mini_batch_size: Size of the mini-batches

    Returns:
        Tuples of mini-batch indices
    """

    for begin in range(0, data_length, mini_batch_size):
        end = min(begin + mini_batch_size, data_length)
        yield begin, end

def revert_ctrl_token(token):
    if token == "<pad>":
        return ""
    elif token == "<unk>":
        return ""
    elif token == "<newline>":
        return "\n"
    else:
        return token

def textify(tokens):
    """
    De-tokenizes a list of tokens back into normal text.

    Args:
        tokens: List of tokens

    Returns:
        String of text
    """
    d = TreebankWordDetokenizer()
    tokens = filter(lambda x: x != "", map(revert_ctrl_token, tokens))
    return d.detokenize(tokens).replace("\n ", "\n")