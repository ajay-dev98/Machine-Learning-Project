#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:24:46 2020

@author: dev
"""

import numpy as np

# here u define the numebr of datset u wish to create and the range
x=np.arange(np.deg2rad(0.01),np.deg2rad(180),np.deg2rad(0.05))

#x = np.random.normal(0,np.pi,500)
z = x,',',np.cos(x)
# opens a file named '---.txt'
with open('label_cos.txt', 'w') as file:
    for i in range(len(x)):
        # writes the value of x and f(x). currently f is set as cos
        file.write('{},{}\n'.format(x[i],np.cos(x[i])))


    
    
    
    
