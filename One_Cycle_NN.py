# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:09:09 2017

@author: Umberto
"""

import numpy as np

def ForAndBack(network_tuple, dataset):
     
     w_in = network_tuple(0)
     w_h = network_tuple(1)
     w_o = network_tuple(2)
     b_h = network_tuple(3)
     b_o = network_tuple(4)
     
     f = lambda x: 1.0/(1.0 + np.exp(-x))
     
     