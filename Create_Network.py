# -*- coding: utf-8 -*-

#This script creates a Neural Network
# (input_layer, hidden_layer, output_layer) = Network(Number of Layers, Number of Neurons, Dataset in a Bunch file)
"""
Created on Tue Mar 28 17:49:45 2017

@author: Umberto
"""

import numpy as np

np.random.seed(3)

def Network(Layers, Neurons, dataset):
     
     var = 0.2
     mean = 0.5
     
     to_input_w = var*(np.random.rand(Neurons,np.size(dataset.data[0,:]))+ (mean-0.5))
     
     hidden_w = var*(np.random.rand(Neurons,Neurons,Layers-1)+ (mean-0.5))
     hidden_b = var*(np.random.rand(Neurons,Layers)+ (mean-0.5))
     
     output_w = var*(np.random.rand(np.size(dataset.target_names),Neurons)+ (mean-0.5))
     out_b = var*(np.random.rand(np.size(dataset.target_names))+ (mean-0.5))
     
     return to_input_w, hidden_w, output_w, hidden_b, out_b