# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:17:25 2019

@author: jaide
"""

import pandas as pd
import numpy as np
""" Sigmoid layer"""
class Sigmoid:
    
    def __init__(self,input_size,output_size):
        
        self.W=np.random.randn(output_size,input_size)
        self.gradient=np.zeros((output_size,input_size))
        self.output=np.zeros((output_size,1))
        
    def forward(self,X):
        s= np.inner(self.W,X)
        
        e=np.exp(-s)
        e=1+e
        self.output=1/e
        
        return self.output
    
    def backward(self,incoming_gradient,X):
        local_gradient= ((1.0 - self.output) * self.output) 
        self.gradient=  local_gradient * np.sum(incoming_gradient,axis=0)
        self.gradient = np.outer(self.gradient,X)
        return self.gradient
        
