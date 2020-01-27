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