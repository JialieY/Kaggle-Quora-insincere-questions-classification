import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
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

_num = train_X.apply(lambda x: len(x.split()))
print("longest sentence {}".format(_num.max()))
print("shortest sentence {}".format(_num.min()))


train["question_text"] = train_X
test["question_text"] = test_X

train.to_csv("train2.csv", index=False)
test.to_csv("test2.csv", index=False)



