# ANLP Final Project

## Goal

Create a language model to generate fake lyrics using RNNs.

## Requirements

This project requires python3.

```
pip3 install -r requirements.txt
```

## Setup

Download the Kaggle dataset from [here](https://www.kaggle.com/mousehead/songlyrics) and extract it. Then convert the data using the `data.py` script.

This will generate a file called `sentences.txt` which contains
the tokenized song lyrics. Every song is contained in one line.

The current development model is in the Dev.ipynb IPython notebook.
It uses GloVe vectors as the embedding.

Download the pre-trained model [here](http://nlp.stanford.edu/data/glove.6B.zip)

Then convert the glove model into a word2vec format, using the following command

```
python3 glove_conv.py
```

## Papers

- [Deep Poetry](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2762063.pdf)
- [GhostWriter](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP221.pdf)
- [Karpathy's Blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Accuracy of RNNs for Lyrics Gen](http://ejosue.com/wp-content/uploads/COMPSCI380-2018-LSTM-RNN.pdf)
- [Medium Article](https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)