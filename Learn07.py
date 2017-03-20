# -*- coding: utf-8 -*-
#Creation of a dateset to implement a classifier
"""
Created on Fri Mar 17 14:52:20 2017

@author: Umberto
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

iterations = 50

X1 = np.zeros((2*iterations,2))
y = []

loop = range(0,iterations)

for i in loop:
    X1[i] = np.random.normal(2,0.9,2)
    y.append(1)
    
for i in loop:
    X1[i + iterations] = np.random.normal(-2,0.9,2)
    y.append(-1)
    
plt.scatter(X1[:,0], X1[:,1], s=40, c=y, cmap=plt.cm.Spectral)
