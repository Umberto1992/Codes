# -*- coding: utf-8 -*-
#Testing 2D - matrices and product between them
"""
Created on Thu Mar 16 10:42:55 2017

@author: Umberto
"""

import numpy
matrix_alpha = numpy.zeros((3,2))
matrix_alpha[0][1] = 1
matrix_alpha[2][0] = 5
print(matrix_alpha) 

x = numpy.array( ((3,5), (2,9), (3,7)) )
y = numpy.array( ((0,2,1), (2,0,0)))

print(x)
print(y)

z = numpy.dot(x,y)

print(z)

w = numpy.dot(y,x)

print (w)