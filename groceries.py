# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:48:15 2020

@author: UNME
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import matplotlib.pyplot as plt

groc = pd.read_excel("C:\\Users\\UNME\\Downloads\\DS Course\\association rules assignment\\groceries.xlsx", header = None)
groc.head()

groc = groc[~groc.isna()]
groc

items = groc[0].unique()
items

groc1 = pd.get_dummies(groc)
groc1

freqitem = apriori(groc1, min_support = 0.5, use_colnames = True)
freqitem

freqitem = apriori(groc1, min_support = 0.1, use_colnames = True)
freqitem

freqitem = apriori(groc1, min_support = 0.8, use_colnames = True)
freqitem

freqitem = apriori(groc1, min_support = 0.005, use_colnames = True)
freqitem

rules = association_rules(freqitem, metric = "confidence", min_threshold = 0.6)
rules

plt.scatter(rules["support"], rules["confidence"], alpha = 0.5); plt.xlabel("support");plt.ylabel("confidence");plt.title("support vs confidence"); plt.show()

plt.scatter(rules["support"], rules["lift"], alpha = 0.5); plt.xlabel("support");plt.ylabel("confidence");plt.title("support vs lift"); plt.show()

plt.scatter(rules["lift"], rules["confidence"]); plt.xlabel("lift");plt.ylabel("confidence"); plt.title("lift vs confidence"); plt.show()

z = np.polyfit(rules["lift"], rules["confidence"], 1)
zfn = np.poly1d(z)

plt.plot(rules["lift"], rules["confidence"], "yo", rules["lift"], zfn(rules["lift"]))
