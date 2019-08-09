import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.layers import Dense, Input, Add, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.models import Model
from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv("train_ori.csv")
test_df = pd.read_csv("test_ori.csv")

def lower(str, lower=True):
    c = 0
    if lower:
        for i in str:
            if(i.islower()):
                c +=1
    else:
        for i in str:
            if(i.isupper()):
                c +=1
    return c

# 1. the number of words
train_df["f1"] = train_df["question_text"].apply(lambda x: len(x.split()))
test_df["f1"]  = test_df["question_text"].apply(lambda x: len(x.split()))
# 2. the number of unique words
train_df["f2"] = train_df["question_text"].apply(lambda x: len(set(x.split())))
test_df["f2"]  = test_df["question_text"].apply(lambda x: len(set(x.split())))
# 3. the number of characters
train_df["f3"] = train_df["question_text"].apply(lambda x: lower(x))
test_df["f3"]  = test_df["question_text"].apply(lambda x: lower(x))
# 4. the number of upper characters
train_df["f4"] = train_df["question_text"].apply(lambda x: lower(x,False))
test_df["f4"]  = test_df["question_text"].apply(lambda x: lower(x,False))


# clean text
train_X = train_df["question_text"]
test_X = test_df["question_text"]

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

# Set up model, first features are word count, second set of features are statistcs data 
count_vec = CountVectorizer(max_df=0.9, min_df=5)
X_train = count_vec.fit_transform(train_X)
X_text = count_vec.transform(test_X)

X_train, X_val, y_train, y_val= train_test_split(X_train, train_df["target"].values, test_size=0.1, random_state=2019)
X_train2,X_val2,_,_ = train_test_split(train_df.iloc[:,-4:].values, train_df["target"].values, test_size=0.1, random_state=2019)

input1 = Input(shape=(len(count_vec.vocabulary_),))
x1 = Dense(64, activation='relu')(input1)
input2 = Input(shape=(4,))
x2 = Dense(1, activation="relu")(input2)
added = Add()([x1, x2])
out = Dense(1, activation="sigmoid")(added)

model = Model(inputs=[input1, input2], outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# train model
model.fit([X_train,X_train2],y_train, batch_size=512, epochs=2, validation_data=([X_val,X_val2] ,y_val))

# predict test data
pred_lg_val_y = model.predict([X_val,X_val2], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_val, (pred_lg_val_y>thresh).astype(int))))

