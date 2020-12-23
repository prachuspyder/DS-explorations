# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:44:37 2020

@author: UNME
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

compd = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\Random Forest Assignment\\Company_Data.csv")

compd.Urban = preprocessing.LabelEncoder().fit_transform(compd.Urban)
compd.US = preprocessing.LabelEncoder().fit_transform(compd.US)
compd.ShelveLoc = preprocessing.LabelEncoder().fit_transform(compd.ShelveLoc)
compd.Sales = compd.Sales.astype(int)
compd.info()

x = compd.drop("Sales", 1)
y = compd["Sales"]

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
