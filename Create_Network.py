# -*- coding: utf-8 -*-

#This script creates a Neural Network
# (input_layer, hidden_layer, output_layer) = Network(Number of Layers, Number of Neurons, Dataset in a Bunch file)
"""
Created on Tue Mar 28 17:49:45 2017

@author: Umberto
"""

import numpy as np

def Network(Layers, Neurons, dataset):
     
     to_input_w = 10*(np.random.rand(Neurons,np.size(dataset.data[:,0]))-0.5)
     
     hidden_w = 10*(np.random.rand(Neurons,Neurons,Layers)-0.5)
     hidden_b = 10*(np.random.rand(Neurons,Layers)-0.5)
     
     output_w = 10*(np.random.rand(np.size(dataset.target_names),Neurons)-0.5)
     out_b = 10*(np.random.rand(np.size(dataset.target_names)))
     
     return to_input_w, hidden_w, output_w, hidden_b, out_b