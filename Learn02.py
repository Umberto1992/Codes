# -*- coding: utf-8 -*-
# This script is to learn how to use for cycle
#
#
#
"""
Created on Wed Mar 15 15:09:24 2017

@author: Umberto
"""

xs = [0, 0, 0]

count = range(1,5)

for integer in count:
     xs[0] = xs[0] + 1
     xs[1] = xs[1] + 2
     xs[2] = xs[2] + 3
     print (xs)

print ("Result is ",xs)