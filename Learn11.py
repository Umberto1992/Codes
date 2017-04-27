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

UpdNet, MSE, final_out = loopNN.ForAndBack(Net, X)

Error = np.mean(MSE)

OUTPUT = final_out

print(MSE, Error)

tpr = 0;

for i in range(0,1500):
     
     UpdNet, MSE, final_out = loopNN.ForAndBack(UpdNet, X)

     Error = np.mean(MSE)
     
     OUTPUT = final_out
     
     if tpr == 500:

          print(MSE, Error)
          tpr = 0
     
     tpr += 1
