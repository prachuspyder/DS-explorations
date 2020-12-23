# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:51:27 2020

@author: UNME
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

fraud = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\Random Forest Assignment\\Fraud_check.csv")
fraud.Undergrad = preprocessing.LabelEncoder().fit_transform(fraud.Undergrad)
fraud.Urban = preprocessing.LabelEncoder().fit_transform(fraud.Urban)
fraud = fraud.rename(columns = {"Taxable.Income":"ti", "Marital.Status":"ms","City.Population":"popu", "Work.Experience":"workex"})
fraud.ms = preprocessing.LabelEncoder().fit_transform(fraud.ms)
fraud.info()
fraud.isna().sum()

x = fraud.drop("ti", 1)
y = fraud["ti"]

kf = KFold(n_splits = 10, random_state = 7, shuffle = True)
model = RandomForestClassifier(n_estimators = 100, max_features = 3)
result = cross_val_score(model, x,y, cv = kf)
result
result.mean()

from sklearn.ensemble import AdaBoostClassifier
model1 = AdaBoostClassifier(n_estimators = 100, random_state = 7)
result1 = cross_val_score(model1, x, y, cv = kf)
result1
result1.mean()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

estimators = []
model2 = LogisticRegression(max_iter=500)
estimators.append(('logistic', model2))
model3 = DecisionTreeClassifier()
estimators.append(('cart', model3))
model4 = SVC()
estimators.append(('svm', model4))

ensemble = VotingClassifier(estimators)
result2 = cross_val_score(ensemble, x, y, cv=kf)
result2.mean()
result2
