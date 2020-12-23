# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 01:22:29 2020

@author: UNME
"""

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crime = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\CLustering Assignment\\crime_data.csv")

crime.head()
crime.info()

crime = crime.rename(columns = {"Unnamed: 0":"City"})
crime

crime1 = crime.drop(["City"], axis = 1)
crime1

x = crime1.values
xscaled = StandardScaler().fit_transform(x)

crime2 = pd.DataFrame(xscaled, columns = crime1.columns)
crime2

dbscan = DBSCAN(eps=0.5, min_samples = 5)
dbscan.fit(crime2)
dbscan.labels_

#Cannot use DBSCAN as all the data set are coming as outliers. Performing a KMeans clustering

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

kmeans = KMeans(n_clusters = 3).fit(crime2)
kmeans.labels_

k = list(range(2,10))

twss = []
for i in k:
    kmeans = KMeans(n_clusters = i).fit(crime2)
    wss = []
    for j in range(i):
        wss.append(sum(cdist(crime2.iloc[kmeans.labels_==j,:], kmeans.cluster_centers_[j].reshape(1, crime2.shape[1]), "euclidean")))
    twss.append(sum(wss))
twss

plt.plot(k, twss, "ro-");plt.xlabel("clusters");plt.ylabel("total within ss");    

#Considering kas 6 from elbow chart

kmeansf = KMeans(n_clusters = 6).fit(crime2)
kmeansf.labels_

pred = kmeansf.predict(crime2)

pred1 = pd.Series(pred)
pred1
crime2["clusters"] = pred1
crime2

crime = pd.concat([crime, crime2], axis = 1)
crime
crime = crime.groupby(["City"]).mean()
crime

