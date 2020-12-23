# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:54:58 2020

@author: UNME
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

airline = pd.read_excel("C:\\Users\\UNME\\Downloads\\DS Course\\CLustering Assignment\\EastWestAirlines.xlsx", sheet_name = "data")

airline.head()
airline.isnull().sum()
airline.info()

normair = normalize(airline)

airline1 = pd.DataFrame(normair, columns = airline.columns)
airline1

dend1 = sch.dendrogram(sch.linkage(airline1, method='single'))

dend2 = sch.dendrogram(sch.linkage(airline1, method='complete'))

hc = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "complete")

yhc = hc.fit_predict(airline1)
cl = pd.DataFrame(yhc, columns = ["clusters"])
cl

airline2 = pd.concat([airline1, cl], 1)
airline2

airline2 = airline2.groupby(["clusters"]).mean()
airline2

#Clustering with K-Means

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

kmeans = KMeans(n_clusters = 4).fit(airline1)
kmeans.labels_

k=list(range(2,10))

twss = []
for i in k:
    kmeans = KMeans(n_clusters = i).fit(airline1)
    wss = []
    for j in range(i):
        wss.append(sum(cdist(airline1.iloc[kmeans.labels_==j,:], kmeans.cluster_centers_[j].reshape(1, airline1.shape[1]), "euclidean")))
    twss.append(sum(wss))
twss

plt.plot(k, twss, "ro-");plt.xlabel("clusters");plt.ylabel("total within ss");

#as elbow smoothens at 4,5,6 clusters. I am assuming 5 as the number of optimal clusters

kmeansf = KMeans(n_clusters = 5).fit(airline1)
pred = kmeansf.predict(airline1)

airline3 = airline1
airline3["clusters"] = pred
airline3 = airline3.groupby(["clusters"]).mean()
airline3

