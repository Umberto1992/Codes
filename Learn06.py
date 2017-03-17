# -*- coding: utf-8 -*-
# Use of random function and create a dataset and plot it
#

"""
Created on Fri Mar 17 14:38:00 2017

@author: Umberto
"""

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

np.random.seed(12)

X, y = sklearn.datasets.make_moons(50, noise=0.001)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

print(X)