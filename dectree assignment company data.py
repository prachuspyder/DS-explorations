# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:30:04 2020

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

compd = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\Decision Tree Assignment\\Company_Data.csv")
compd.head()
compd.info()
compd.ShelveLoc = preprocessing.LabelEncoder().fit_transform(compd.ShelveLoc)
compd.head()
compd.ShelveLoc
compd.Urban = preprocessing.LabelEncoder().fit_transform(compd.Urban)
compd.US = preprocessing.LabelEncoder().fit_transform(compd.US)

x = compd.drop(["Sales"],1)
y = compd["Sales"]
y = y.astype(int)
x_train, x_test,y_train,y_test = tts(x,y, test_size=0.2,random_state=40)

#Entropy criteria
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(x_train, y_train)

tree.plot_tree(model)
model.get_depth()

model1 = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
model1.fit(x_train, y_train)

tree.plot_tree(model1)

predtrainy = model1.predict(x_train)

from sklearn.metrics import accuracy_score
accuracytrain = accuracy_score(y_train, predtrainy)
accuracytrain

predtesty = model1.predict(x_test)
accuracytest = accuracy_score(y_test, predtesty)
accuracytest

predtrainy1 = model.predict(x_train)
accuracytrain1 = accuracy_score(y_train, predtrainy1)
accuracytrain1

predtesty1 = model.predict(x_test)
accuracytest1 = accuracy_score(y_test, predtesty1)
accuracytest1

#GINI Method

model3 = DecisionTreeClassifier(criterion="gini")
model3.fit(x_train, y_train)
tree.plot_tree(model3)
model3.get_depth()

predtrainy3 = model3.predict(x_train)
accuracytrain3 = accuracy_score(y_train, predtrainy3)
accuracytrain3

predtesty3 = model3.predict(x_test)
accuracytest3 = accuracy_score(y_test, predtesty3)
accuracytest3

#Decison Tree Regressor method

from sklearn.tree import DecisionTreeRegressor

model4 = DecisionTreeRegressor()
model4.fit(x_train,y_train)

tree.plot_tree(model4)
model4.get_depth()
predtrainy4 = model4.predict(x_train)
accuracytrain4 = accuracy_score(y_train, predtrainy4)
accuracytrain4

predtesty4 = model4.predict(x_test)
accuracytest4 = accuracy_score(y_test, predtesty4)
accuracytest4

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
