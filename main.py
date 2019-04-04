# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:12:39 2019

@author: jaide
"""

import os
import pickle
import pandas as pd
import numpy as np
from keras.datasets import cifar10
from Model import Model

if __name__== "__main__":
    """ Cifar - 10 dataset (data pre processing same as Assignment 1)"""    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    labels= {0:'airplane',
             1:'automobile',
             2:'bird',
             3:'cat',
             4:'deer',
             5:'dog',
             6:'frog',
             7:'horse',
             8:'ship',
             9:'truck',
            }
    
    
    
    idx=np.where(y_train[:] == [0,1,2])[0]
    X_train=X_train[idx,::]
    y_train=y_train[idx]
    
    idx_test=np.where(y_test[:] == [0,1,2])[0]
    X_test=X_test[idx_test,::]
    y_test=y_test[idx_test]
    
    
    
    X_train=X_train.astype('float32')/255
    X_test=X_test.astype('float32')/255
    
    X_train=X_train.reshape((15000,32*32*3))
    X_test=X_test.reshape((3000,32*32*3))
    
    from keras.utils import to_categorical
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    
    X_val=X_train[:5000]
    partial_X_train=X_train[5000:]
    y_val=y_train[:5000]
    partial_y_train=y_train[5000:]
    
    """ Build Model and plot evaluation graph """    
    model = Model()
    
    model.build(partial_X_train.shape[1],partial_y_train.shape[1],[255,16])
    
    history=model.fit(partial_X_train,partial_y_train,X_val,y_val,epochs=5)
    
    import matplotlib.pyplot as plt
    
    acc = [d['training_acc'] for d in history]
    val_acc = [d['val_acc'] for d in history]
    loss = [d['training_loss'] for d in history]
    val_loss = [d['val_loss'] for d in history]
    
    epochs = range(1, len(acc) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    plt.savefig("Figure.png")
    
