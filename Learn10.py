# -*- coding: utf-8 -*-
# Neural Network, 1 hidden layer 2 neurons, input with 2 features, output two classes
"""
Created on Wed Mar 22 11:22:51 2017

@author: Umberto
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)


############################ Initialization #####################################

iterations = 100 # How many couples of data we want
X1 = np.zeros((2*iterations,2)) # Dataset
y = []                          # Labels
loop = range(0,iterations)
W1 = np.random.normal(0,6,(3,2)) # weights from inout to hidden layer
W2 = np.random.normal(0,6,(1,3)) # weights from hidden layer to output layer
b1 = np.random.rand(3,1) # bias to the hidden layer
b2 = np.random.rand() # bias to the output layer
out = np.zeros((2*iterations)) 


########################### Dataset Creation ###################################

for i in loop:
    X1[i] = np.random.normal(2,0.9,2)
    y.append(1)
    
for i in loop:
    X1[i + iterations] = np.random.normal(-2,0.9,2)
    y.append(0)
    
plt.scatter(X1[:,0], X1[:,1], s=40, c=y, cmap=plt.cm.Spectral) # Plot of the dataset

for steps in range(0,3000):

     ######################### Feedforward Step ######################################
     
     # forward-pass of a 2-layer neural network:
     f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)

     out_h1 = f(np.dot(W1, np.transpose(X1)) + b1) # calculate first hidden layer activations
     out = f(np.dot(W2, out_h1) + b2) # output neuron, it's a single neuron
     
     
     ########################  Gradient Estimation - Output Layer ###################################
      
     error = out - y # calculate total error summing every single squared error
     derivative_out = out*(1-out) #gradient of the sigmoid function applied in every single value stored at the output node
     delta_out = derivative_out*error # error weighted with gradient 
     print("The error mean is: ", np.mean(error))
#     print("Weight 1 is: ", W1)
#     print("Weight 2 is: ", W2)
         
     ########################  Gradient Estimation and Backpropagation - Hidden Layer ###################################
     
     delta_h1 = np.zeros((3,2*iterations)) # delta for the hidden neurons
     derivative_h1 = out_h1*(1-out_h1)
     for i in range(0,3):
          delta_h1[i] = derivative_h1[i]*W2[0,i]*delta_out #delta in the hidden layer
     
     
     ########################## Weights update #####################################
     
     eps = 0.7
     alpha = 0.3 #to escape local minima
     
     previous_DELTA_out = np.zeros((3,1))
     current_DELTA_out = np.zeros((3,1))
     
     for i in range(0,3):
          current_DELTA_out[i] = -eps*np.sum(delta_out*out_h1[i,:]) + alpha*previous_DELTA_out[i]
          W2[0,i] = W2[0,i] + current_DELTA_out[i]
     
     
     b2 = b2 - np.sum(delta_out)     
     previous_DELTA_out = current_DELTA_out
     
     previous_DELTA_l1 = np.zeros((3,2))
     current_DELTA_l1 = np.zeros((3,2))
     
     for i in range(0,3):
          
         for j in range(0,2):
               
             current_DELTA_l1[i,j] = -eps*np.sum(delta_h1[i,:]*X1[:,j]) + alpha*previous_DELTA_l1[i,j]
             W1[i,j] = W1[i,j] + current_DELTA_l1[i,j]
               
         b1[i] = b1[i] - np.sum(delta_h1[i,:])     
     
     
####################### Plot the Classifier ##############################     
     
X_plots = 10*(np.random.rand(9000,2)-0.5)
Yplots = np.zeros(9000)

outplot_h1 = f(np.dot(W1, np.transpose(X_plots)) + b1) # calculate first hidden layer activations
Yplots = f(np.dot(W2, outplot_h1) + b2) # output neuron, it's a single neuron
plt.scatter(X_plots[:,0], X_plots[:,1], s=1, c=Yplots, cmap=plt.cm.Spectral) # Plot of the dataset