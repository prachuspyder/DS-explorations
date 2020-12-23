# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:03:12 2020

@author: UNME
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np

book = pd.read_csv("C:\\Users\\UNME\\Downloads\\DS Course\\association rules assignment\\book.csv")
book.head()
book.info()
book
book.mean()
book.mode()

freqitem = apriori(book, min_support = 0.1, use_colnames = True)
freqitem

rules = association_rules(freqitem, metric="confidence", min_threshold = 0.4)
rules

rulesa = association_rules(freqitem, metric="confidence", min_threshold = 0.7)
rulesa

freqitem1 = apriori(book, min_support = 0.2, use_colnames = True)
freqitem1

rules1 = association_rules(freqitem, metric="confidence", min_threshold = 0.7)
rules1

rules.sort_values("conviction")[0:20]
rulesa.sort_values("lift", ascending = False)[0:10]
rules1.sort_values("confidence")[0:10]

rules[rules.lift>1]
rulesa[rulesa.lift>1]
rules1[rules1.lift>1]

import matplotlib.pyplot as plt

plt.scatter(rules["support"], rules["confidence"], alpha = 0.5); plt.xlabel("support");plt.ylabel("confidence");plt.title("support vs confidence"); plt.show()

plt.scatter(rules["support"], rules["lift"], alpha = 0.5); plt.xlabel("support");plt.ylabel("confidence");plt.title("support vs lift"); plt.show()

plt.scatter(rules["lift"], rules["confidence"]); plt.xlabel("lift");plt.ylabel("confidence"); plt.title("lift vs confidence"); plt.show()

z = np.polyfit(rules["lift"], rules["confidence"], 1)
zfn = np.poly1d(z)

plt.plot(rules["lift"], rules["confidence"], "yo", rules["lift"], zfn(rules["lift"]))
