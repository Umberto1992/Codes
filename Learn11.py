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


Net = myNN.Network(2,3,X)

UpdNet, MSE = loopNN.ForAndBack(Net, X)

Error = np.mean(MSE)

print(MSE, Error)

for i in range(0,15000):
     
     UpdNet, MSE = loopNN.ForAndBack(UpdNet, X)

     Error = np.mean(MSE)

     print(MSE, Error)
     
     
