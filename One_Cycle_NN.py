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
     n_layers = np.size(w_h[0,0,:])+1
     n_neurons = np.size(w_h[0,:,0])
     n_sample = np.size(X_input[0,:])
     n_features = np.size(X_input[:,0])
     n_labels = np.size(b_o)
     partial_output = np.zeros((n_neurons, n_layers, n_sample))
     final_out = np.zeros((n_labels,n_sample))
     
     f = lambda x: 1.0/(1.0 + np.exp(-x))
     
###################### FeedForward Step ####################################################à
     
     partial_output[:,0,:] = f(np.dot(w_in, X_input).T + b_h[:,0]).T
     
     for i in range(1,n_layers):
          partial_output[:,i,:] = f(np.dot(w_h[:,:,i-1], partial_output[:,i-1]).T + b_h[:,i]).T
     
     final_out = f(np.dot(w_o, partial_output[:,i]).T + b_o).T
     
     
 ######################### Error Estimate #######################################
     
     error = np.zeros((n_labels,n_sample))
     
     for j in range(0,n_sample):
        for i in range(0,n_labels):
             if Y_target[j] == i:
 #                 accum[i] += (1-final_out[i,j])**2
 #                 accum[i] += final_out[i,j] - 1
                 error[i,j] = final_out[i,j]-1
             else:
 #                 accum[i] += (final_out[i,j])**2
#                  accum[i] += final_out[i,j]
                error[i,j] = final_out[i,j]
     MSE = np.sum(error**2/2,1)/n_sample
     
     
########################### Backpropagation ########################################

     deriv_out = final_out*(1 - final_out)
     delta_out = deriv_out*error
     
     deriv_h = partial_output*(1 - partial_output)
     delta_h = np.zeros((n_neurons, n_layers, n_sample))
 
     delta_h[:,n_layers-1,:] = deriv_h[:,n_layers-1,:]*np.sum(np.dot(w_o.T,delta_out),0)  
     
     for i in reversed(range(n_layers-1)):
          delta_h[:,i,:] = deriv_h[:,i,:]*np.sum(np.dot(w_h[:,:,i],delta_h[:,i+1,:]),0)
 
    # delta_in = deri

###################### Weights Update ############################################

     eps = 0.7
     alpha = 0 #to escape local minima
     
     
###################### Out Update #################################################     
     
     previous_DELTA_out = np.zeros((n_labels, n_neurons))
     current_DELTA_out = np.zeros((n_labels, n_neurons))
     
     for i in range(0,n_labels):
          current_DELTA_out[i,:] = -eps*np.sum(partial_output[:,n_layers-1,:]*delta_out[i,:],1) + alpha*previous_DELTA_out[i,:]
    
     w_o = w_o + current_DELTA_out
     b_o = b_o - eps*np.sum(delta_out,1)  
     previous_DELTA_out = current_DELTA_out
     
   
###################### Hidden Layers Update ############################################     
     
     previous_DELTA_hl = np.zeros((n_neurons, n_neurons, n_layers-1))
     current_DELTA_hl = np.zeros((n_neurons, n_neurons, n_layers-1))
     
     for i in reversed(range(n_layers-2)):
          current_DELTA_hl[:,:,i] = -eps*np.sum(delta_h[:,i+1,:]*partial_output[:,i,:],1) + alpha*previous_DELTA_hl[:,:,i]
     
     w_h = w_h + current_DELTA_hl
     b_h = b_h - np.sum(delta_h,2)
     previous_DELTA_hl = current_DELTA_hl

########################## Input Update ################################################

     previous_DELTA_inl = np.zeros((n_neurons, n_features))
     current_DELTA_inl = np.zeros((n_neurons, n_features))
     
     for i in range(0,n_features):
         current_DELTA_inl[:,i] = -eps*np.sum(delta_h[:,0,:]*X_input[i,:],1) + alpha*previous_DELTA_inl[:,i] 
     
     w_in = w_in + current_DELTA_inl
     previous_DELTA_inl = current_DELTA_inl
     
     UpdatedNet = (w_in, w_h, w_o, b_h, b_o)
     
     return UpdatedNet, MSE, final_out