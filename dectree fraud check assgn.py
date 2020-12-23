# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:39:49 2020

@author: UNME
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

fraud = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\Decision Tree Assignment\\Fraud_check.csv")

fraud.head()
fraud.info()
fraud = fraud.rename(columns = {"Taxable.Income":"ti", "Marital.Status":"ms","City.Population":"popu", "Work.Experience":"workex"})
fraud.info()
fraud.isna().sum()
fraud.Undergrad = preprocessing.LabelEncoder().fit_transform(fraud.Undergrad)
fraud.ms = preprocessing.LabelEncoder().fit_transform(fraud.ms)
fraud.Urban = preprocessing.LabelEncoder().fit_transform(fraud.Urban)
fraud.head()

x = fraud.drop(["ti"],1)
y = fraud["ti"]
x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.5, random_state = 50)

#Entropy criteria
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(x_train,y_train)

tree.plot_tree(model)
model.get_depth()

predtrainy = model.predict(x_train)
accuracytrain = accuracy_score(predtrainy, y_train)
accuracytrain

predtesty = model.predict(x_test)
accuracytest = accuracy_score(predtesty, y_test)
accuracytest

#Gini criteria
model1 = DecisionTreeClassifier(max_depth = 5)
model1.fit(x_train, y_train)
tree.plot_tree(model1)

predtrainy1 = model1.predict(x_train)
accuracytrain1 = accuracy_score(predtrainy1, y_train)
accuracytrain1

predtesty1 = model1.predict(x_test)
accuracytest1 = accuracy_score(predtesty1, y_test)
accuracytest1

#Decison Tree Regressor method

model2 = DecisionTreeRegressor(max_depth = 5)
model2.fit(x_train, y_train)

tree.plot_tree(model2)

predtrainy2 = model2.predict(x_train)
accuracytrain2 = accuracy_score(predtrainy2.round(), y_train)
accuracytrain2

predtesty2 = model2.predict(x_test)
accuracytest2 = accuracy_score(predtesty2.round(), y_test)
accuracytest2

#XGBoost

from numpy import loadtxt
from xgboost import XGBClassifier

model5 = XGBClassifier(learning_rate = 0.700, random_state = 40)
model5.fit(x_train, y_train)

predtrain5 = model5.predict(x_train)
accuracytrain5 = accuracy_score(y_train, predtrain5)
accuracytrain5

predtest5 = model5.predict(x_test)
accuracytest5 = accuracy_score(y_test, predtest5)
accuracytest5

#Light GBM

import lightgbm as lgb
traind = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.703
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

model6 = lgb.train(params, traind, 1280)
predtrainy6 = model6.predict(x_train)
accuracytrain6 = accuracy_score(y_train, predtrainy6.round())
accuracytrain6

predtesty6 = model6.predict(x_test)
accuracytest6 = accuracy_score(y_test, predtesty6.round())
accuracytest6
