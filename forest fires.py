# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:55:37 2020

@author: UNME
"""

import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

forfire = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\Neural networks assgn\\forestfires.csv")

forfire.info()

x = forfire.drop("area",1)
y = forfire["area"]

x.month = LabelEncoder().fit_transform(x.month)
x.day = LabelEncoder().fit_transform(x.day)
x.size_category = LabelEncoder().fit_transform(x.size_category)
x.info()

xstd = StandardScaler().fit_transform(x)

#Tuning Batch size and Epochs

def create_model():
    model = Sequential()
    model.add(Dense(12, input_shape = (30,), activation = "relu"))
    model.add(Dense(8, activation = "relu"))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    return model

model = KerasClassifier(build_fn = create_model)
batchsize = [10,20,30]
epochs = [10,50,100]
param = dict(batch_size = batchsize, epochs = epochs)
grid = GridSearchCV(estimator = model, param_grid = param, cv = KFold())
gridresult = grid.fit(xstd, y)

print("Best{}, using{}".format(gridresult.best_score_, gridresult.best_params_))
means = gridresult.cv_results_['mean_test_score']
stds = gridresult.cv_results_['std_test_score']
params = gridresult.cv_results_['params']
for mean, stdev, param1 in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param1))

from keras.layers import Dropout

def createdrpoutmodel(learning_rate, dropout_rate):
    model = Sequential()
    model.add(Dense(12, input_shape = (30,), activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation = "relu"))
    model.add(Dense(1, activation = "sigmoid"))
    adam = Adam(lr = learning_rate)
    model.compile(loss = "binary_crossentropy", optimizer = adam, metrics = ["accuracy"])
    return model


model1 = KerasClassifier(build_fn = createdrpoutmodel,batch_size = 30,epochs = 10)

learning_rate = [0.001,0.01,0.1]
dropout_rate = [0.0,0.1,0.2]

param1 = dict(learning_rate = learning_rate, dropout_rate = dropout_rate)
grid1 = GridSearchCV(estimator = model1, param_grid = param1, cv = KFold())
grid1result = grid1.fit(xstd, y)

print("Best{}, using{}".format(grid1result.best_score_, grid1result.best_params_))
means = grid1result.cv_results_['mean_test_score']
stds = grid1result.cv_results_['std_test_score']
params1 = grid1result.cv_results_['params']
for mean, stdev, param2 in zip(means, stds, params1):
  print('{},{} with: {}'.format(mean, stdev, param2))
  
act_func = ['softmax','relu','tanh','linear']
init = ['uniform','normal','zero']
  
def createafmodel(act_func, init):
    model = Sequential()
    model.add(Dense(12, input_shape = (30,),kernel_initializer = init, activation = act_func))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation = act_func))
    model.add(Dense(1, activation = "sigmoid"))
    adam = Adam(lr = 0.001)
    model.compile(loss = "binary_crossentropy", optimizer = adam, metrics = ["accuracy"])
    return model

model2 = KerasClassifier(build_fn = createafmodel,batch_size = 30,epochs = 10)

param3 = dict(act_func = act_func, init = init)

grid2 = GridSearchCV(estimator = model2,param_grid = param3,cv = KFold())
grid2result = grid2.fit(xstd,y)

print("Best{}, using{}".format(grid2result.best_score_, grid2result.best_params_))
means = grid2result.cv_results_['mean_test_score']
stds = grid2result.cv_results_['std_test_score']
params2 = grid2result.cv_results_['params']
for mean, stdev, param3 in zip(means, stds, params2):
  print('{},{} with: {}'.format(mean, stdev, param3))
  
def createneumodel(neuron1, neuron2):
    model = Sequential()
    model.add(Dense(12, input_shape = (30,),kernel_initializer = "uniform", activation = "linear"))
    model.add(Dropout(0.2))
    model.add(Dense(8, kernel_initializer = "uniform", activation = "linear"))
    model.add(Dense(1, activation = "sigmoid"))
    adam = Adam(lr = 0.001)
    model.compile(loss = "binary_crossentropy", optimizer = adam, metrics = ["accuracy"])
    return model

model3 = KerasClassifier(build_fn = createneumodel,batch_size = 30,epochs = 10)

neuron1 = [4,8,16]
neuron2 = [2,4,8]

param4 = dict(neuron1 = neuron1, neuron2 = neuron2)
grid3 = GridSearchCV(estimator = model3,param_grid = param4,cv = KFold())
grid3result = grid3.fit(xstd,y)

print("Best{}, using{}".format(grid3result.best_score_, grid3result.best_params_))
means = grid3result.cv_results_['mean_test_score']
stds = grid3result.cv_results_['std_test_score']
params3 = grid3result.cv_results_['params']
for mean, stdev, param4 in zip(means, stds, params3):
  print('{},{} with: {}'.format(mean, stdev, param4))
  
def creatfinalmodel():
    model = Sequential()
    model.add(Dense(4, input_shape = (30,),kernel_initializer = "uniform", activation = "linear"))
    model.add(Dropout(0.2))
    model.add(Dense(8, kernel_initializer = "uniform", activation = "linear"))
    model.add(Dense(1, activation = "sigmoid"))
    adam = Adam(lr = 0.001)
    model.compile(loss = "binary_crossentropy", optimizer = adam, metrics = ["accuracy"])
    return model

finalmodel = KerasClassifier(build_fn = creatfinalmodel,batch_size = 30,epochs = 10)

finalmodel.fit(xstd,y)

predy = finalmodel.predict(xstd)

from sklearn.metrics import classification_report, accuracy_score
accuracy = accuracy_score(predy.round(),y.round())
accuracy

xf = pd.DataFrame(xstd, columns = x.columns)
predy1 = predy*100
xf["prediction_%"] = predy1
xf
