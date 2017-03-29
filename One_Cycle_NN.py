# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:09:09 2017

@author: Umberto
"""

import numpy as np

def ForAndBack(network_tuple, dataset):
     
     w_in =network_tuple[0]
     w_h = network_tuple[1]
     w_o = network_tuple[2]
     b_h = network_tuple[3]
     b_o = network_tuple[4]
     X_input = np.transpose(dataset.data)
     Y_target = dataset.target
     n_layers = np.size(w_h[0,0,:])
     n_neurons = np.size(w_h[0,:,0])
     n_sample = np.size(X_input[0,:])
     n_labels = np.size(b_o)
     partial_output = np.zeros((n_neurons, n_layers, n_sample))
     final_out = np.zeros((n_labels,n_sample))
     
     f = lambda x: 1.0/(1.0 + np.exp(-x))
     
###################### FeedForward Step ####################################################à
     
     partial_output[:,0,:] = f(np.dot(w_in, X_input).T + b_h[:,0]).T
     
     for i in range(1,n_layers):
          partial_output[:,i,:] = f(np.dot(w_h[:,:,i-1], partial_output[:,i-1]).T + b_h[:,i]).T
     
     final_out = f(np.dot(w_o, partial_output[:,i]).T + b_o)
     
 ######################### Error Estimate #######################################
     
     accum = np.zeros(n_labels)
     for i in range(0,n_labels):
        for j in range(0,n_sample):
             if Y_target[j] == i:
                  accum[i] += (1-final_out[j,i])**2
             else:
                  accum[i] += (final_out[j,i])**2
          
     MSE = accum/n_sample     
########################### Backpropagation ########################################

     deriv_out = final_out*(1-final_out)
     delta_out = deriv_out*MSE
     
     deriv_h = partial_output*(1-partial_output)
     delta_h = np.zeros((n_neurons, n_layers, n_sample))
     
     for i in range(0,n_neurons):
          delta_h[i,n_layers-1,:] = deriv_h[i,n_layers-1,:]*np.sum(np.dot(w_o[:,i],delta_out.T))
     
     
     for i in reversed(range(n_layers-1)):
          for j in range (0,n_neurons):
               delta_h[j,i,:] = deriv_h[j,i,:]*np.sum(np.dot(w_h[j,:,i],delta_h[:,i+1,:]))

     
     return final_out, MSE, delta_h


