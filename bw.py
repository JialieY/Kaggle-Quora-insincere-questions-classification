import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_X = train["question_text"].fillna("").values
test_X = test["question_text"].fillna("").values

count_vec = CountVectorizer(max_df=0.9, min_df=5, binary= True)
X_train = count_vec.fit_transform(train_X)
X_text = count_vec.transform(test_X)

X_train, X_val, y_train, y_val= train_test_split(X_train, train["target"].values, test_size=0.1, random_state=2019)

model = Sequential()
model.add(Dense(64, input_dim=len(count_vec.vocabulary_), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, batch_size=512, epochs=2, validation_data=(X_val, y_val))


pred_lg_val_y = model.predict([X_val], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_val, (pred_lg_val_y>thresh).astype(int))))

model.save('my_lg_model.h5')
print("done")
