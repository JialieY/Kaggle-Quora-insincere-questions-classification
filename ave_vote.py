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
from keras.models import Model, load_model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2019)
## fill up the missing values
train_X = train_df["question_text"].fillna("").values
val_X = val_df["question_text"].fillna("").values
test_X = test_df["question_text"].fillna("").values
## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values


xg_train = pd.DataFrame({"target":train_y})
xg_val = pd.DataFrame({"target":val_y})
xg_test = pd.DataFrame({"target":np.zeros(len(test_df))})

def record_output(model, feature, X_train,X_val, X_test ):
    pred1 = model.predict([X_train], batch_size=1024, verbose=1)
    pred2 = model.predict([X_val], batch_size=1024, verbose=1)
    pred3 = model.predict([X_test], batch_size=1024, verbose=1)
    xg_train[feature] = pred1
    xg_val[feature] = pred2
    xg_test[feature] = pred3
    del model
    del pred1, pred2, pred3

## set up models
# 1. word count
count_vec = CountVectorizer(max_df=0.9, min_df=5)
X_train = count_vec.fit_transform(train_X)
X_val = count_vec.transform(val_X)
X_test = count_vec.transform(test_X)

model = Sequential()
model.add(Dense(64, input_dim=len(count_vec.vocabulary_), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, train_y, batch_size=512, epochs=2, validation_data=(X_val, val_y))
record_output(model,"wc", X_train, X_val, X_test)


# 2. tf-idf
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5)
X_train = vectorizer.fit_transform(train_X)
X_val = vectorizer.transform(val_X)
X_text = vectorizer.transform(test_X)

model = Sequential()
model.add(Dense(64, input_dim=len(vectorizer.vocabulary_), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, train_y, batch_size=512, epochs=2, validation_data=(X_val, val_y))
record_output(model,"tfidf", X_train, X_val, X_test)

## some config values
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
# max_features = 3000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

## Keras Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen, padding = 'post')
val_X = pad_sequences(val_X, maxlen=maxlen, padding = 'post')
test_X = pad_sequences(test_X, maxlen=maxlen, padding = 'post')


## convert embbedings
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

## build up model
def build_model(EMBEDDING_FILE):
    if EMBEDDING_FILE == 'embeddings/glove.840B.300d/glove.840B.300d.txt':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='latin'))
        feat = "glove"
        all_embs = np.stack(embeddings_index.values())
    if EMBEDDING_FILE == 'embeddings/paragram_300_sl999/paragram_300_sl999.txt':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)
        feat = "para"
        all_embs = np.stack(embeddings_index.values())
    if EMBEDDING_FILE == 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)
        feat = "wiki"
        all_embs = np.stack(embeddings_index.values())
    if EMBEDDING_FILE =='embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin':
        from gensim.models import KeyedVectors
        embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
        feat = "gg"
        all_embs = embeddings_index.vectors

    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        try:
            embedding_vector = embeddings_index[word]
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        except:
            continue

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

    record_output(model, feat, train_X, val_X, test_X)

    del embeddings_index, all_embs,embedding_matrix
    return

# 3. Keras train default Embeddings
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
# x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
record_output(model,"nopre",train_X, val_X, test_X)


# 4. glove
EMBEDDING_FILE = 'embeddings/glove.840B.300d/glove.840B.300d.txt'
build_model(EMBEDDING_FILE)
del EMBEDDING_FILE

# 5. para
EMBEDDING_FILE = 'embeddings/paragram_300_sl999/paragram_300_sl999.txt'
build_model(EMBEDDING_FILE)
del EMBEDDING_FILE

# 6. wiki
EMBEDDING_FILE = 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
build_model(EMBEDDING_FILE)
del EMBEDDING_FILE

# 7. google
EMBEDDING_FILE = 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
build_model(EMBEDDING_FILE)
del EMBEDDING_FILE


xg_train.to_csv("xg_train50000.csv", index=False)
xg_val.to_csv("xg_val50000.csv", index=False)
xg_test.to_csv("xg_test50000.csv", index=False)

# Average Vote
xg_val["pred"] = xg_val.iloc[:,1:8].mean(axis=1)
xg_test["pred"] = xg_test.iloc[:,1:8].mean(axis=1)

for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(xg_val["target"].values, (xg_val["pred"].values>thresh).astype(int))))


pred_ave_y = (xg_test["pred"]>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_ave_y
out_df.to_csv("submission.csv", index=False)

