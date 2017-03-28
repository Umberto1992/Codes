# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:49:45 2017

@author: Umberto
"""

import numpy as np

def Network(Layers, Neurons, dataset):
     
     to_input_w = 10*(np.random.rand(Neurons,np.size(dataset.data[:,0]))-0.5)
     
     hidden_w = 10*(np.random.rand(Neurons,Neurons,Layers)-0.5)
     
     output_w = 10*(np.random.rand(np.size(dataset.target_names),Neurons)-0.5)
     
     return to_input_w, hidden_w, output_w