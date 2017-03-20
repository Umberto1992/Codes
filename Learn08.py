# -*- coding: utf-8 -*-
# I used the dataset created in Learn07 and then implemented the feed-forward step
# with random parameters.
"""
Created on Mon Mar 20 10:19:38 2017

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

###############################################################

W1 = np.random.rand(3,2)
W2 = np.random.rand(2,3)
b1 = np.random.rand(1,3)
b2 = np.random.rand(2,1)
out = np.zeros((2*iterations,2))

# forward-pass of a 2-layer neural network:
f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
for i in loop:
     x = X1[i] # random input vector of two numbers (2x1)
     h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (2x1)
     out[i] = np.transpose(np.dot(W2, np.transpose(h1)) + b2) # output neuron (2x1)
     
for i in loop:
     x = X1[iterations + i] # random input vector of two numbers (2x1)
     h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (2x1)
     out[iterations + i] = np.transpose(np.dot(W2, np.transpose(h1)) + b2) # output neuron (2x1)

