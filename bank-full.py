# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:35:13 2020

@author: UNME
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression as lr

bank = pd.read_csv("C:\\Users\\UNME\Downloads\\DS Course\\Logistic Reg Assignment\\bank-full.csv", header = 0, sep = ";")
bank.head()
bank.columns
bank.isna().sum()
bank.describe()
bank1 = bank.drop(["campaign","pdays", "previous"], axis=1)
bank1
bank1.dtypes
bank1.info()

mnth = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}

bank1["y"] = [0 if x=="no" else 1 for x in bank1["y"]]
bank1["default"] = [0 if x=="no" else 1 for x in bank1["default"]]
bank1["housing"] = [0 if x=="no" else 1 for x in bank1["housing"]]
bank1["loan"] = [0 if x=="no" else 1 for x in bank1["loan"]]

bank1.education.value_counts()
bank1.job.value_counts()
bank1.marital.value_counts()
bank1.default.value_counts()
bank1.housing.value_counts()
bank1.loan.value_counts()
bank1.contact.value_counts()
bank1.month.value_counts()
bank1.poutcome.value_counts()

bank1.month = [mnth[item] for item in bank1.month]
bank1

for col in bank1.columns:
    if bank1[col].dtypes =="object":
        print(bank1[col].value_counts())
        print()
        
dummylist = ["job", "marital", "education", "contact", "poutcome"]
def dummy_df(df, dummylist):
    for x1 in dummylist:
        dummies = pd.get_dummies(df[x1], prefix=x1, dummy_na=False)
        df = df.drop(x1, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

bank1 = dummy_df(bank1, dummylist)
print(bank1.shape)

from sklearn import preprocessing
x2 = bank1.values
mms = preprocessing.MinMaxScaler()
x2scaled = mms.fit_transform(x2)
bank2 = pd.DataFrame(x2scaled, columns = bank1.columns)
bank2

x = bank2.drop("y", axis = 1)
y = bank2.y
x.shape
y.shape

from sklearn.model_selection import train_test_split as tts
train_x,test_x= tts(x,test_size=0.3)
train_y,test_y= tts(y,test_size=0.3)

classifier = lr(solver='lbfgs',class_weight='balanced', max_iter=10000)
classifier.fit(x,y)
y_pred = classifier.predict(x)
y_pred
y_preddf = pd.DataFrame({"actual":y, "predicted":y_pred})
y_preddf

from sklearn.metrics import confusion_matrix as cm
print(cm(y, y_pred))

from sklearn.metrics import classification_report as cr
print(cr(y, y_pred))

print(cm(test_y, classifier.predict(test_x)))
print(cr(test_y, classifier.predict(test_x)))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba (x)[:,1])

df_new=pd.DataFrame({"fpr":fpr,"tpr":tpr,"cutoff":thresholds})
df_new[df_new["fpr"]>=0.22]

auc = roc_auc_score(y, y_pred)
auc
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')

prob=classifier.predict_proba(x)
prob=prob[:,1]

new_pred= pd.DataFrame({'actual': y,"pred":0})
new_pred.loc[prob>0.58,"pred"]=1
new_pred

cmnew = cm(new_pred.actual,new_pred.pred) 
cmnew
print(cr(new_pred.actual, new_pred.pred))
