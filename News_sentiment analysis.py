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
import scipy
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df = pandas.read_csv(r"C:\Users\ishan\Desktop\Project\DJIA-prediction-from-news\Combined_News_DJIA.csv")

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

## XG boost

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

# recall = 85%

precision = tp/(fp+tp)
print(precision)

# precision = 83% 

accuracy = accuracy_score(df_test['Label'], predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# accuracy = 84%

## Random Forest 


clf_rf = RandomForestClassifier().fit(X_train,y_train)


ypred = clf_rf.predict(vec_data_test)

cm = confusion_matrix(df_test['Label'], ypred)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)

recall = tp/(tp+fn)
print(recall)

# recall = 99.5%

precision = tp/(fp+tp)
print(precision)

# precision =77%

accuracy = accuracy_score(df_test['Label'], ypred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# acucracy = 85 % 

## Naive Bayes


clf = GaussianNB()
clf.fit(X_train.toarray(),y_train)

ypred = clf.predict(vec_data_test.toarray())

cm = confusion_matrix(df_test['Label'], ypred)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)

recall = tp/(tp+fn)
print(recall)

# recall = 87%

precision = tp/(fp+tp)
print(precision)

# precision = 80%

accuracy = accuracy_score(df_test['Label'], ypred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# accuracy 83%


## Multi-Layer Perceptron 

clf = MLPClassifier(hidden_layer_sizes=(100,2),activation='relu',random_state=0, max_iter=300)

clf.fit(X_train, y_train)

ypred =clf.predict(vec_data_test.toarray())

cm = confusion_matrix(df_test['Label'], ypred)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)

recall = tp/(tp+fn)
print(recall)

#recall = 64%

precision = tp/(fp+tp)
print(precision)

# precision = 100% 

accuracy = accuracy_score(df_test['Label'], ypred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# accuracy = 81.5% 


## K-NN 

neigh = KNeighborsClassifier(n_neighbors=8)
neigh.fit(X_train, y_train)

ypred =neigh.predict(vec_data_test)

cm = confusion_matrix(df_test['Label'], ypred)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)

recall = tp/(tp+fn)
print(recall)

# recall = 99%

precision = tp/(fp+tp)
print(precision)

# precision = 76%

accuracy = accuracy_score(df_test['Label'], ypred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


## Ensamble of MLP and Random Forest:
    
mlp_pred_proba = clf.predict_proba(vec_data_test.toarray())

fr_pred_proba = clf_rf.predict_proba(vec_data_test.toarray())

idx = df_test['Label'].index
mlp = pandas.DataFrame(mlp_pred_proba,index=idx)
rf= pandas.DataFrame(fr_pred_proba,index=idx)

combined = pandas.concat([rf[0],mlp[1]],axis=1)

rf2 = pandas.DataFrame(np.array([rf.iloc[i,0]>0.5 for i in range(len(rf))]).astype(int),index=idx)
mlp2 = pandas.DataFrame(np.array([mlp.iloc[i,1]>0.5 for i in range(len(mlp))]).astype(int),index=idx)

average_prob = pandas.concat([rf[0],mlp[1]*0.89],axis=1)

average_prob[0].mean()
max(average_prob[0])
average_prob[1].mean()
max(average_prob[1])

ensamble_pred = np.array([average_prob.iloc[i,1]>average_prob.iloc[i,0] for i in range(len(average_prob)) ]).astype(int)

cm = confusion_matrix(df_test['Label'],ensamble_pred)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)

recall = tp/(tp+fn)
print(recall)

precision = tp/(fp+tp)
print(precision)


accuracy = accuracy_score(df_test['Label'], ensamble_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

eclf = VotingClassifier(estimators=[('rf',clf_rf),('mlp',clf),('KNN',neigh)],voting='soft')

eclf = eclf.fit(X_train,y_train)

ypred = eclf.predict(vec_data_test.toarray())

cm = confusion_matrix(df_test['Label'], ypred)
tn, fp, fn, tp = cm.ravel()
(tn, fp, fn, tp)

recall = tp/(tp+fn)
print(recall)

# recall = 80%

precision = tp/(fp+tp)
print(precision)

# precision = 95%

accuracy = accuracy_score(df_test['Label'], ypred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# accuracy = 88%



