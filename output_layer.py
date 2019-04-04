# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:32:34 2019

@author: jaide
"""

import numpy as np
""" Softmax layer"""
class Softmax:
    def __init__(self,input_size,output_size):
        print("Hi")
        self.W=np.random.randn(output_size,input_size)
        self.gradient=np.zeros((output_size,input_size))
        self.output=np.zeros((output_size,1))
        
    def forward(self,X):
       
        exp_score=np.exp(np.inner(self.W,X)) 
        self.output = exp_score / np.sum(exp_score, axis=-1, keepdims=True)
       
        return self.output
    
    def backward(self,incoming_gradient,X):
        
        g1= np.dot(np.diag(np.ones(self.output.shape[0])) , self.output)
        g2=self.output * self.output
        
        local_gradient= g1 - g2
        self.gradient = local_gradient * incoming_gradient
        
        self.gradient = np.outer(self.gradient , X)
         
        return self.gradient
        
    
