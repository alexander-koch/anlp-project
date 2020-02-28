# ANLP Final Project

## Goal

Create a language model to generate fake lyrics using RNNs.

## Requirements

This project requires python3.

```
pip3 install -r requirements.txt
```

## Setup & Training

Download the Kaggle dataset from [here](https://www.kaggle.com/mousehead/songlyrics) or use the provided zip file from data/songdata.
Use the `convert_kaggle.py` script to generate the word vocabulary and to pre-tokenize all the songs.

Next the the vocabulary size needs to be reduced by using the `gen_vocab_word.py` script.
This will use Mikolov subsampling to generate a vocabulary file that removes less common words.

To train the word-level model, the training data needs to be generated beforehand.
This can be done by running the `ngramify.py` script. For a fixed sequence length and a given
vocabulary, the script will generated a text file consisting of training samples. Each sample
is encoded using the provided vocabulary.

Now the word-level model can be trained using `train_word.py`. Make sure to create a
directory named `weights` and to download the [GloVe embedding vectors](http://nlp.stanford.edu/data/glove.6B.zip).
The GloVe model needs to be converted into the word2vec format using `glove_conv.py`.

To train the character-level model, no setup is required. The vocabulary will be built
if it was not previously generated.

## Development

Previous development versions of the different models can be found in the various IPython notebooks.

## File structure

* sampler.py - Allows to sample from the character- and word-level models, given the paths to their weights and vocabularies
* util.py - Collection of functions used to encode and pre-process the data into the correct formats
* train_word.py - Word-level language model
* train_char.py - Character-level language model

## Papers

- [Deep Poetry](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2762063.pdf)
- [GhostWriter](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP221.pdf)
- [Karpathy's Blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Accuracy of RNNs for Lyrics Gen](http://ejosue.com/wp-content/uploads/COMPSCI380-2018-LSTM-RNN.pdf)
- [Medium Article](https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)
- [Computing Text Similarity](http://tuprints.ulb.tu-darmstadt.de/4342/1/TUD-CS-2015-0017.pdf)
