# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:35:27 2019

@author: jaide
"""

import pandas as pd
import numpy as np
from layer import Sigmoid
from output_layer import Softmax
""" File to build NN model, forward and backward computation (training),  and evaluation """
class Model:
    
    def __init__(self):
        self.layers=[]
        #self.output_layer=[]
        self.input_size= 0
        self.output_size=0
        self.fit_results = []
    
    def build(self, input_size, output_size, hidden_layer_units):
        
        np.random.seed(7)
        self.input_size=input_size
        self.output_size=output_size
        
        self.layers.append(Sigmoid(input_size,hidden_layer_units[0]))
        for i in range(1,len(hidden_layer_units)):
            self.layers.append(Sigmoid(hidden_layer_units[i-1],hidden_layer_units[i]))
        
        self.layers.append(Softmax(hidden_layer_units[i],output_size))
        
        
    def forward(self,X):
        
        for layer in self.layers:
            #print(X.shape)
            X = layer.forward(X)
        
        yhat= X
        #print(yhat.shape)
        return yhat
    
    def categorical_loss_gradient(self,y_pred,y):
        return y_pred - y
    
    def categorical_loss(self, y_pred,y):
        cor_logprobs = - np.log(y_pred[y == 1])
        loss = np.sum(cor_logprobs) 
        return (loss / len(y_pred))
    
    def accuracy(self, y_pred,y):
        return np.round(np.float((np.sum(y_pred==y))/len(y_pred))*100,2)
    
    def backward(self, y_pred, y , X):
        
        loss_gradient = self. categorical_loss_gradient(y_pred, y)
        #print(loss_gradient.shape)
        for i in reversed(range(len(self.layers))):
            if (i>0):
                #print(i)
                ip= self.layers[i-1].output
                loss_gradient = self.layers[i].backward(loss_gradient,ip)
                #print(loss_gradient.shape)
        
        loss_gradient = self.layers[0].backward(loss_gradient,X) 
        
    def update_gradient(self, learning_rate):

        for layer in reversed(self.layers):
            #print(layer,layer.gradient.shape)
            layer.W = layer.W - (learning_rate * layer.gradient)
            
    def model_evaluation(self, X, Y):
        
        y_pred=np.zeros(Y.shape)
        for i in range(len(X)):
            y_pred[i] = self.forward(X[i])
        
        loss = self.categorical_loss (y_pred, Y)
        accuracy = self.accuracy(y_pred,Y)
        
        return loss, accuracy
            
    def fit(self, x_train, y_train, x_val = 0, y_val = 0, learning_rate=0.01, epochs=10):
        
        for epoch in range(epochs):
            print("Epoch:" + str(epoch))
            for i in range(len(x_train)):
                #print(i)
                y_pred= self.forward(x_train[i])
                self.backward(y_pred,y_train[i],x_train[i])
                self.update_gradient(learning_rate)
                
            training_loss , training_accuracy = self.model_evaluation(x_train, y_train)
            val_loss , val_accuracy = self.model_evaluation(x_val, y_val)
            print({'training_loss': training_loss , 
                                     'training_acc': training_accuracy , 
                                     'val_loss': val_loss, 
                                     'val_acc': val_accuracy})
            self.fit_results.append({'training_loss': training_loss , 
                                     'training_acc': training_accuracy , 
                                     'val_loss': val_loss, 
                                     'val_acc': val_accuracy})
            
        return self.fit_results
        
        