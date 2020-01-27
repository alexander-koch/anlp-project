# ANLP Final Project

## Goal

Generate a model to generate fake lyrics using RNNs.

## Requirements

This project requires python3.

```
pip3 install -r requirements.txt
```

## Setup

The date folder contains example data that can be converted using
the `data.py` script

```
python3 data.py
```

this will generate a file called `sentences.txt` which contains
the tokenizes song lyrics. Every song is contained in one line.

The current development model is in the Dev.ipynb IPython notebook.
It uses GloVe vectors as an embedding

Download the pre-trained model [here](http://nlp.stanford.edu/data/glove.6B.zip)
Then convert the glove model into a word2vec format, using the following command

```
python3 glove_conv.py
```