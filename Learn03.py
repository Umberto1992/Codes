# -*- coding: utf-8 -*-
# This code is performing a bubble sort.
# Here I learn how to use random numbers and variable 
# iterations.
"""
Created on Wed Mar 15 16:39:44 2017

@author: Umberto
"""

from random import randrange

iteration = 14;
loop = range(0,iteration+1);
storage = []
flag = 1;
temp = 0;


for i in loop:
     storage.append(randrange(0, 100))
     print (storage[i])
    

loop = range(0,iteration)
while flag != iteration:
     flag = 0
     for i in loop:
          if storage[i] < storage[i+1]:
               temp = storage [i]
               storage[i] = storage[i+1]
               storage[i+1] = temp
          else:
               flag = flag + 1         
     print(storage)   
               
     