# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, Concatenate
from keras.models import Model


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ", train.shape)
print("Test shape : ", test.shape)

print("train zeros = {0} {1:2f}, ones = {2} {3:2f}".format(train.shape[0]- train["target"].sum(),
                                                     (train.shape[0] - train["target"].sum())/train.shape[0] ,
                                                     train["target"].sum(),
                                                     train["target"].sum()/train.shape[0],))

train_X = train["question_text"].fillna(" ")
test_X = test["question_text"].fillna(" ")

## clean text
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, "")
    specials = {'\u200b': '', '…': '', '\ufeff': '', 'करना': '','है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    return text

train_X = train_X.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
test_X = test_X.apply(lambda x: clean_special_chars(x, punct, punct_mapping))

train["question_text"] = train_X
test["question_text"] = test_X

## split to train and val
train_df, val_df = train_test_split(train, test_size=0.1, random_state=2019)
test_df = test

## some config values
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("").values
val_X = val_df["question_text"].fillna("").values
test_X = test_df["question_text"].fillna("").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen, padding = 'post')
val_X = pad_sequences(val_X, maxlen=maxlen, padding = 'post')
test_X = pad_sequences(test_X, maxlen=maxlen, padding = 'post')

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

## glove embeddings
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='latin'))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

embedding_matrix1 = embedding_matrix
del embedding_matrix,embeddings_index

# para embeddings
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

embedding_matrix2 = embedding_matrix
del embedding_matrix,embeddings_index

# concatenate two embedings, 300-> 600 features
embedding_matrix = np.concatenate((embedding_matrix1,embedding_matrix2 ), axis=1)


# set up model
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size*2, weights=[embedding_matrix])(inp)
x = SpatialDropout1D(0.3)(x)
x1 = Bidirectional(LSTM(256, return_sequences=True))(x)
x2 = Bidirectional(GRU(128, return_sequences=True))(x1)
max_pool1 = GlobalMaxPool1D()(x1)
max_pool2 = GlobalMaxPool1D()(x2)
conc = Concatenate()([max_pool1, max_pool2])
predictions = Dense(1, activation='sigmoid')(conc)
model = Model(inputs=inp, outputs=predictions)

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# train model
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_com_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_com_val_y>thresh).astype(int))))

# prediction on test set
pred_com_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = (pred_com_test_y>0.33).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
