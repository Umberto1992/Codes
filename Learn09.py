# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:18:04 2017

@author: Umberto
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

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
    y.append(0)
    
plt.scatter(X1[:,0], X1[:,1], s=40, c=y, cmap=plt.cm.Spectral)

######################### Feedforward Step ######################################

W1 = np.random.rand(3,2) # weights from inout to hidden layer
W2 = np.random.rand(1,3) # weights from hidden layer to output layer
b1 = np.random.rand(3,1) # bias to the hidden layer
b2 = np.random.rand() # bias to the output layer
out = np.zeros((2*iterations)) 


# forward-pass of a 2-layer neural network:
f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)

x = X1 # random input vector
h1 = f(np.dot(W1, np.transpose(x)) + b1) # calculate first hidden layer activations
x = h1
out = f(np.dot(W2, x) + b2) # output neuron, it's a single neuron

########################  Error Estimation - Output Layer ###################################
 
error = out - y # calculate error of every single input at the output
derivative_out = scipy.misc.derivative(f,np.dot(W2, x) + b2) #gradient of the sigmoid function applied in every single value stored at the output node
delta_out = derivative_out*error # error weighted with gradient
gradient_out = np.zeros((3,2*iterations))

for i in range(0,3):
     
    gradient_out[i] =h1[i,:]*delta_out
########################  Error Estimation - Hidden Layer ###################################

delta_h = np.zeros((3,2*iterations)) # delta for the hidden neurons
for i in range(0,3):
     delta_h[i] = scipy.misc.derivative(f,np.dot(W1[i,:], np.transpose(X1)) + b1[i])*W2[0,i]*delta_out #delta in the hidden layer

########################## Backpropagation #####################################

eps = 0.3
alpha = 0.01

previous_delta = np.zeros((3,1))