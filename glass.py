# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:39:26 2020

@author: UNME
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

glass = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\KNN Assignments\\glass.csv")
glass.info()
glass.dtypes

x = glass.drop(["Type"], 1)
y = glass["Type"]

nf = 7
kfold = KFold(n_splits = 7)

knc = KNeighborsClassifier(n_neighbors = 4)
results = cross_val_score(knc,x,y,cv=kfold)
results.mean()

nneigh = np.array(range(1,50))
pg = dict(n_neighbors = nneigh)

grid = GridSearchCV(estimator = knc, param_grid = pg)
grid.fit(x,y)

grid.best_score_, grid.best_params_

krange = range(1,50)
kscores = []
for k in krange:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, x, y, cv=4)
    kscores.append(scores.mean())
    
plt.plot(krange, kscores); plt.xlabel("K Value"); plt.ylabel("Cross Validated Accuracy"); plt.show()

glass.Type.value_counts(sort = False).plot(kind = "bar"); plt.show()

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.33)

def RMSE(pred,actual):
    return np.sqrt(np.mean((actual-pred)**2))

rmse = []
for k in range(1,50):
    k = k+1
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    error = RMSE(y_test, pred)
    rmse.append(error)

rmse

plt.plot(rmse)   
