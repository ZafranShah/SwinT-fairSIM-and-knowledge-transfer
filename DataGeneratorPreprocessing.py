#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:31:22 2022

@author: shah
"""

import numpy as np 
from tensorflow import keras
from skimage import io
import ImageHelpers    
    





class CustomDataGenerator(keras.utils.Sequence):
   
    """This class is the implementation of custom Generators to load the training and test data by using custom generator to avoid any memory issue during the training process.
    
    Parameters:
    ----------    
 
    inputIDs: path of the folder contain input images
    labelsIDs: path of the folder contain output images
    batch_size: batch of images loaded
    to_fit: while fitting the model should be set to True and during the evaluation set to False default=True 
    dim: tuple of dimension of input and output images default=(1024,1024) 
    n_channels: channel of the images default=1
    shuffle: shuffle the batch of samples before propagation default= False
 
    Returns: 
    ----------    
    
    tensors of input and output samples  """
    
    def __init__(self, inputIDs, labelsIDs, batch_size=2,to_fit=True, dim=(1024, 1024), n_channels=1,  shuffle=False):
        
        'Initialization'

        self.batch_size = batch_size
        self.labelsIDs = labelsIDs
        self.inputIDs = inputIDs
        self.to_fit = to_fit

        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
 
        
    def __len__(self):
        
        """Denotes the number of batches per epoch
        
        return: number of batches per epoch"""
        return int(np.floor(len(self.inputIDs) / self.batch_size))

    def __getitem__(self, index):
        
        """Generate one batch of data
        
        parameter index: index of the batch

        return: X and Y when called in the fit function otherwise only X is returned"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        inputIDs_temp = [self.inputIDs[k] for k in indexes]
        
        labelsIDs_temp = [self.labelsIDs[k] for k in indexes]


        # Generate data
        X = self._generate_X(inputIDs_temp)

        if self.to_fit:
            y = self._generate_y(labelsIDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
       
       """Updates indexes after each epoch """
       
       self.indexes = np.arange(len(self.inputIDs))
       if self.shuffle == True:
           np.random.shuffle(self.indexes)

    def _generate_X(self, inputIDs_temp):
        
        """Generates data containing batch_size images

        parameter list_IDs_temp: list of label ids to load

        return: batch of images"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # X=[]
        # Generate data
        for i, ID in enumerate(inputIDs_temp):
            imgx= np.asarray(io.imread(ID))
            imgx= ImageHelpers.rescaleSingleImgs(imgx)
            X[i,] = imgx.reshape(*self.dim,self.n_channels)
        return X

    def _generate_y(self, labelsIDs_temp):
        
        """Generates data containing batch_size masks

        parameter list_IDs_temp: list of label ids to load

        return: batch if masks"""

        y = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(labelsIDs_temp):
            # Store sample
            imgy= np.asarray(io.imread(ID))
            imgy= ImageHelpers.rescaleSingleImgs(imgy)
            y[i,] =imgy.reshape(*self.dim,self.n_channels)

        return y


