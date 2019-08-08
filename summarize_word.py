import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_X = train["question_text"].fillna("").values
test_X = test["question_text"].fillna("").values

## Tokenize the sentences
tokenizer = Tokenizer(oov_token= True)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
## 1. Glove summary
EMBEDDING_FILE = 'embeddings/glove.840B.300d/glove.840B.300d.txt'
embeddings_index1 = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='latin'))

## 2. paragram summary
EMBEDDING_FILE = 'embeddings/paragram_300_sl999/paragram_300_sl999.txt'
embeddings_index2 = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))

## 3. wiki summary
EMBEDDING_FILE = 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
embeddings_index3 = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))

## 4.google summary
from gensim.models import KeyedVectors
EMBEDDING_FILE = 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index4 = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)


n1,n2,n3,n4 = 0,0,0,0
k1,k2,k3,k4 = 0,0,0,0
u1,u2,u3,u4 = 0,0,0,0
d1,d2,d3,d4 = set(),set(),set(),set()


for word in tokenizer.word_counts:
    if word in embeddings_index1:
        n1 += 1
        k1 += tokenizer.word_counts[word]
    else:
        d1.add(word)
        u1 += tokenizer.word_counts[word]

    if word in embeddings_index2:
        n2 += 1
        k2 += tokenizer.word_counts[word]
    else:
        d2.add(word)
        u2 += tokenizer.word_counts[word]

    if word in embeddings_index3:
        n3 += 1
        k3 += tokenizer.word_counts[word]
    else:
        d3.add(word)
        u3 += tokenizer.word_counts[word]

    if word in embeddings_index4:
        n4 += 1
        k4 += tokenizer.word_counts[word]
    else:
        d4.add(word)
        u4 += tokenizer.word_counts[word]


print("glove  vocab coverage {0:.4f}%".format(n1/len(tokenizer.word_counts)*100))
print("glove  words coverage {0:.4f}%".format(k1/(k1+u1)*100))

print("parag  vocab coverage {0:.4f}%".format(n2/len(tokenizer.word_counts)*100))
print("parag  words coverage {0:.4f}%".format(k2/(k2+u2)*100))

print("wiki  vocab coverage {0:.4f}%".format(n3/len(tokenizer.word_counts)*100))
print("wiki  words coverage {0:.4f}%".format(k3/(k3+u3)*100))

print("google  vocab coverage {0:.4f}%".format(n4/len(tokenizer.word_counts)*100))
print("google  words coverage {0:.4f}%".format(k4/(k4+u4)*100))



"""
glove  vocab coverage 51.4034%
glove  words coverage 98.8647%
parag  vocab coverage 60.6852%
parag  words coverage 99.1889%
wiki  vocab coverage 38.8647%
wiki  words coverage 98.1467%
google  vocab coverage 31.2295%
google  words coverage 88.0577%
"""
print("done")






