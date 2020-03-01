# ANLP Final Project

## Goal

Create a language model to generate fake lyrics using RNNs.

## Requirements

This project requires python3.

```
pip3 install -r requirements.txt
```

## Setup & Training

### Word-level language model

Download the Kaggle dataset from [here](https://www.kaggle.com/mousehead/songlyrics) or use the provided zip file from data/songdata.
Use the `scripts/convert_kaggle.py` script to generate the word vocabulary and to pre-tokenize all the songs.

Next the the vocabulary size needs to be reduced by using the `gen_vocab_word.py` script.
This will use Mikolov subsampling to generate a vocabulary file that removes less common words.

To train the word-level model, the training data needs to be generated beforehand.
This can be done by running the `ngramify_word.py` script. For a fixed sequence length and a given
vocabulary, the script will generate a numpy array consisting of training samples. Each sample
is encoded using the provided vocabulary.

Now the word-level model can be trained using `train_word.py`. Make sure to create a
directory named `weights` and to download the [GloVe embedding vectors](http://nlp.stanford.edu/data/glove.6B.zip).
The GloVe model needs to be converted into the word2vec format using `scripts/glove_conv.py`.

### Character-level language model

To train the character-level model, training samples need to be generated beforehand aswell.
Run `ngramify_char.py`, similar to the word-level model.
Now the word-level model can be trained using `train_char.py`.

### Quickstart

Setup

```sh
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python3 scripts/glove_conv.py
python3 scripts/convert_kaggle.py
mkdir weights
```

Training the word-level model

```sh
python3 gen_vocab_word.py -o small_vocab.pkl
python3 ngramify_word.py -i small_vocab.pkl -o ngrams_small.npy
python3 train_word.py -i ngrams_small.npy -v small_vocab.pkl
```

Training the character-level model

```sh
python3 ngramify_char.py -o ngrams_chars.npy
python3 train_char.py -i ngrams_chars.npy
```

- Vocabulary files will be saved in *.pkl files.
- Weights can be retrieved from the weights directory.
- Format: `weights_{word/char}_{k-Fold number}_{epoch number}`

## Sampling

Sampling from the models can be done using the `sampler.py` module.

```python
from sampler import *
import util

ws = WordSampler.from_paths("weights/weights_word_k3_4.h5", "small_vocab.pkl", 6)
cs = CharacterSampler.from_paths("weights/weights_char_k3_10.h5", "chars.pkl", 8)

print(util.textify(ws.sample_sent("never gonna give you up, never gonna", 42)))
print(''.join(cs.sample("somebody once told me", 1337)))
```

To convert the tokens back into text, once can use the textify function of the util module.

## Evaluation

The models are evaluted with k-Fold cross validation based on the model's perplexity.
This happens automatically while the model is training (`train_word.py` or `train_char.py`).

## Development

Previous development versions of the different models and a statistical analysis can be found in the various IPython notebooks.

## File structure

* sampler.py - Allows to sample from the character- and word-level models, given the paths to their weights and vocabularies
* util.py - Collection of functions used to encode and pre-process the data into the correct formats
* train_word.py - Word-level language model
* train_char.py - Character-level language model
* ngramify_word.py - Generates n-gram samples encoded using the vocabulary indices
* ngramify_char.py - Same as ngramify_word.py, just with the character vocabulary
* gen_vocab_word.py - Mikolov subsampling for words/tokens based on all songs

## Papers

- [Deep Poetry](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2762063.pdf)
- [GhostWriter](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP221.pdf)
- [Karpathy's Blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Accuracy of RNNs for Lyrics Gen](http://ejosue.com/wp-content/uploads/COMPSCI380-2018-LSTM-RNN.pdf)
- [Medium Article](https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)
- [Computing Text Similarity](http://tuprints.ulb.tu-darmstadt.de/4342/1/TUD-CS-2015-0017.pdf)
