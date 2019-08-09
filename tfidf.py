import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
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

vectorizer = TfidfVectorizer(max_df=0.9, min_df=5)
X_train = vectorizer.fit_transform(train_X)
X_text = vectorizer.transform(test_X)

X_train, X_val, y_train, y_val= train_test_split(X_train, train["target"].values, test_size=0.1, random_state=2019)


model = Sequential()
model.add(Dense(64, input_dim=len(vectorizer.vocabulary_), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, batch_size=256, epochs=2, validation_data=(X_val, y_val))

pred_tfidf_val_y = model.predict([X_val], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_val, (pred_tfidf_val_y>thresh).astype(int))))


model.save('my_tfidf_model.h5')
pred_keras_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = pred_keras_test_y
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)

