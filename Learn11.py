# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:26:32 2017

@author: Umberto
"""

import numpy as np
import matplotlib.pyplot as plt
import Create_Network as myNN

from sklearn.datasets import load_iris

X = load_iris()


Net = myNN.Network(2,4,X)
