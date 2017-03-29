# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:26:32 2017

@author: Umberto
"""

import numpy as np
import matplotlib.pyplot as plt
import Create_Network as myNN
import One_Cycle_NN as loopNN

from sklearn.datasets import load_iris

X = load_iris()


Net = myNN.Network(5,7,X)

UpdNet, MSE, Delta = loopNN.ForAndBack(Net, X)

Error = np.mean(MSE)

print(MSE, Error)


