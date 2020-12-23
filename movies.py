# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:02:05 2020

@author: UNME
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import matplotlib.pyplot as plt

movies = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\association rules assignment\\my_movies.csv")
movies.info()
movies.head()

movies1 = pd.get_dummies(movies)
movies1

freqitem = apriori(movies1, min_support = 0.5, use_colnames = True)
freqitem

freqitem1 = apriori(movies1, min_support = 0.3, use_colnames = True)
freqitem1

rules = association_rules(freqitem1, metric = "confidence", min_threshold = 0.6)
rules

plt.scatter(rules["support"], rules["confidence"], alpha = 0.5); plt.xlabel("support");plt.ylabel("confidence");plt.title("support vs confidence"); plt.show()

plt.scatter(rules["support"], rules["lift"], alpha = 0.5); plt.xlabel("support");plt.ylabel("confidence");plt.title("support vs lift"); plt.show()

plt.scatter(rules["lift"], rules["confidence"]); plt.xlabel("lift");plt.ylabel("confidence"); plt.title("lift vs confidence"); plt.show()

z = np.polyfit(rules["lift"], rules["confidence"], 1)
zfn = np.poly1d(z)

plt.plot(rules["lift"], rules["confidence"], "yo", rules["lift"], zfn(rules["lift"]))
