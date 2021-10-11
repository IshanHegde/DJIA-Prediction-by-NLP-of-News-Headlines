# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import nltk
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Input
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import sklearn.metrics as metrics

df = pandas.read_csv(r"C:\Users\ishan\Desktop\Project\stock\Combined_News_DJIA.csv\Combined_News_DJIA.csv")

df.isna().sum()

df.fillna("0")

df_train=df[df['Date']<'20150101']
df_test=df[df['Date']>'20141231']

def clean_data(dataset):
    data = dataset.iloc[:,2:27]
    data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    return data

cleaned_train = clean_data(df_train)
cleaned_test = clean_data(df_test)



def combine_data(data):
    headlines = []
    for i in range(len(data.index)):
        headlines.append(' '.join(str(x) for x in data.iloc[i, :]))
    return headlines

combined_train = combine_data(cleaned_train)
combined_test = combine_data(cleaned_test)



def lemmatize_data(data, lemmatizer):
    cleaned_dataset = []
    for i in range(len(data)):
        clean_text = data[i].lower()
        clean_text = clean_text.split()
        clean_text = [lemmatizer.lemmatize(word) for word in clean_text if word not in stopwords.words('english')]
        cleaned_dataset.append(' '.join(clean_text))
    return cleaned_dataset

lemmatizer = WordNetLemmatizer()

lemm_train = lemmatize_data(combined_train, lemmatizer)
lemm_test = lemmatize_data(combined_test, lemmatizer)

def vectorize_data(data, cv):
    vectorized_dataset = cv.fit_transform(data)
    return vectorized_dataset

cv = CountVectorizer(ngram_range=(2,2))




vec_data_train = vectorize_data(lemm_train,cv)
vec_data_test = cv.transform(lemm_test)

y_train=df_train['Label']
X_train=vec_data_train


param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['binary:logistic']
}

xgb_model = XGBClassifier()

gsearch = GridSearchCV(estimator = xgb_model, param_grid = param_tuning,cv = 5,n_jobs = -1,verbose = 1)

gsearch.fit(X_train,y_train)

gsearch.best_params_

xtrain= xgb.DMatrix(vec_data_train)

predictors = xtrain
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=42)

xgb1.fit(X_train,y_train)











y_pred = xgb1.predict(vec_data_test)
predictions = [round(value) for value in y_pred]


cm = confusion_matrix(df_test['Label'], predictions)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)

recall = tp/(tp+fn)
print(recall)

precision = tp/(fp+tp)
print(precision)

accuracy = accuracy_score(df_test['Label'], predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))














