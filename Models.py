#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:29:39 2022

@author: shah
"""
from tensorflow import keras
from Utils import psnr, SSim


class UNetfairSIM:
    
    """This class is the implement UNetwork architecture for the image denoising task
    
       Parameters
       ----------
       
       height, width, channel : Dimension of the input image (type=uint8).
       kernelsize : dimension of the kernel (default = 3 & type=uint8).
       padding :   padding of Conv2d layers (default = 'same' & type = string).
       stride : Stride of Conv2d layers (default = 1 & type = uint8)
       activation : activation function (default='relu' & type=string).
       Max pooling factor: default = 2 type =uint8 
        
       Returns
       -------
        
       model : Architecture of UNetwork model based on compression & decompression blocks"""
    
    
    def __init__(self, height, width, channel, kernelsize=3, padding='same', stride=1, activation='relu', maxpoolingfactor=2):

        self.height=height
        self.width = width
        self.channel= channel
        self.kernelsize = kernelsize
        self.padding = padding
        self.stride = stride
        self.activation=activation
        self.maxpooling= maxpoolingfactor
        
    def Contraction(self, image, filters):
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(image)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(c)
        p = keras.layers.MaxPool2D((self.maxpooling, self.maxpooling), (self.maxpooling, self.maxpooling))(c)
        return c, p

    def Expansion(self,feature, skip, filters):
        us = keras.layers.UpSampling2D((self.maxpooling, self.maxpooling))(feature)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(concat)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(c)
        return c

    def BottleNeck(self, feature, filters):
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(feature)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, strides=self.stride, activation=self.activation)(c)
        return c



    def buildUNetwork(self):
        f =  [16,  32,  64,  128,  256, 512]
        inputs =  keras.layers.Input((self.height , self.width, self.channel))
    
        Input = inputs
        c1, p1 = self.Contraction(Input, f[0]) 
        c2, p2 = self.Contraction(p1, f[1]) 
        c3, p3 = self.Contraction(p2, f[2]) 
        c4, p4 = self.Contraction(p3, f[3]) 
        c5, p5= self.Contraction(p4, f[4]) 
        
        bn = self.BottleNeck(p5, f[5])
          
        u1 = self.Expansion(bn, c5 ,f[4]) 
        u2 = self.Expansion(u1, c4, f[3]) 
        u3 = self.Expansion(u2, c3, f[2]) 
        u4 = self.Expansion(u3, c2, f[1]) 
        u5 = self.Expansion(u4, c1, f[0]) 

        
        outputs = keras.layers.Conv2D(1, (1, 1), padding=self.padding, activation="relu")(u5)
        model = keras.models.Model(Input, outputs)
        model.summary()

        return model
    
    
class RedNetfairSIM:
    
    """This call contains the RedNetwork architecture for image denoising problem 
        
        Parameters
        ----------
        
        height, width, channel : Dimension of the input image (type=uint8).
        kernelsize : dimension of the kernel (default = 3 & type=uint8).
        padding :   padding of Conv2d layers (default = 'same' & type = string).
        stride : Stride of Conv2d layers (default = 1 & type = uint8)
        activation : activation function (default='relu' & type=string).
        
        Returns
        -------
        
        model : Architecture of Red Network model based on encoder & decoder blocks"""
    
    def __init__(self, height, width, channel,filters, kernelsize=3, padding='same', stride=1, activation='relu'):
        self.height=height
        self.width = width
        self.channel= channel
        self.filters= filters
        self.kernelsize = kernelsize
        self.padding = padding
        self.stride = stride
        self.activation=activation
        

    def EncodingBlock(self, x, filters):
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding, data_format='channels_first', strides=self.stride, activation=self.activation)(c)
        return c

    def SecondaryDecodingBlock(self, x, filters):
    
        c = keras.layers.Conv2D(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        return c
    

    def DecodingBlock(self, x, skip, filters):
    
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        concat = keras.layers.Add()([c, skip])
        active= keras.layers.Activation(self.activation)(concat)
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding,data_format='channels_first', strides=self.stride, activation=self.activation)(active)
        return c

    
    def OutputDecodingBlock(self,x, filters):
        c = keras.layers.Conv2DTranspose(filters, kernel_size=self.kernelsize, padding=self.padding, data_format='channels_first', strides=self.stride, activation=self.activation)(x)
        return c

    def buildRedNet(self):
        f= [64, 1]
        inputs =  keras.layers.Input((self.channel, self.height , self.width))

    
        p1 = self.EncodingBlock(inputs, f[0])
        p2 = self.EncodingBlock(p1, f[0])
        p3 = self.EncodingBlock(p2, f[0])
        p4 = self.EncodingBlock(p3, f[0])
        p5 = self.EncodingBlock(p4, f[0])
        p6 = self.EncodingBlock(p5, f[0])
        p7 = self.EncodingBlock(p6, f[0])
        p8 = self.SecondaryDecodingBlock(p7, f[0])
        
        u1 = self.DecodingBlock(p8, p7 ,f[0])
    
        u2 = self.DecodingBlock(u1, p6 ,f[0])
        u3 = self.DecodingBlock(u2, p5 ,f[0])
        u4 = self.DecodingBlock(u3, p4 ,f[0])
        u5 = self.DecodingBlock(u4, p3 ,f[0])
        u6 = self.DecodingBlock(u5, p2 ,f[0])
        u7=  self.DecodingBlock(u6,p1, f[0])
        ouputlayer= self.OutputDecodingBlock(u7, f[1])

    
        model = keras.models.Model(inputs, ouputlayer)
        model.summary()
        return model


class ModelFineTunning:
        
        """This class can be used to to train the pretrained model
        
        Parameters
        ----------
        
        PretrainedmodelFile : path of the pretrained model (type=string).
        startingpoint : strating point of the layers you want to freeze (type=uint8).
        endingpoint : ending pont of the freeze layer (type=uint8).
        
        Returns
        -------
        
        model : Architecture of the pretrained model with freeze & unfreeze blocks"""
    
        def __init__(self, PretrainedmodelFile, startingpoint, endingpoint):
            self.Pretrainedmodel=PretrainedmodelFile
            self.freezeLayerstartPoint = startingpoint
            self.freezeLayerendPoint= endingpoint



        def PretrainedModel(self):
            model = keras.models.load_model(self.Pretrainedmodel, custom_objects ={'psnr':psnr, 'SSim':SSim})
            newModel=keras.models.Model(inputs=model.input, outputs=model.output)
            for layer in newModel.layers[self.freezeLayerstartPoint:self.freezeLayerendPoint]:
                layer.trainable = False
                print ("{0}:\t{1}".format(layer.trainable, layer.name)) 
            newModel.summary()
            return newModel


    
