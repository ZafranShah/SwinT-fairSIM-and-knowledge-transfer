#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:24:19 2020

@author: shah
"""
import tf_psnr
from tensorflow import keras
import pickle, os, numpy 
import tensorflow as tf
import skimage

def GaussianFilter(size, sigma):
    
    """
    This function defines the gaussian filzter w.r.t size and sigma to compute the SSIM of images 
    Parameters
    ----------
    size : size of Gaussian filter window
    sigma: sigma of the Gaussian filter 
    
    return
    ----------
    Gaussain kernel



"""
    x_data, y_data = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = numpy.expand_dims(x_data, axis=-1)
    x_data = numpy.expand_dims(x_data, axis=-1)

    y_data = numpy.expand_dims(y_data, axis=-1)
    y_data = numpy.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def SSim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    """
    
     SSim actuallly calculate the ssim value of predicted and true image
     
    Parameters
    ----------
    img1, img2 : are the true and predicted images 
    cs_map: default= False
    mean_metric: default= True
    size : size of Gaussian filter window default = 11
    sigma: sigma of the Gaussian filter default = 1.5
    
    return
    ----------
    SSIM value of predicted and true image 
    
    """
    img1= tf.expand_dims(img1, 0)
    img2= tf.expand_dims(img2,0)
    img1=tf.cast(img1, 'float32')
    img2=tf.cast(img2, 'float32')
    kernel = GaussianFilter(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, kernel, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, kernel, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, kernel, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, kernel, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, kernel, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value




# Wrapper function for the tensorflow psnr function. Needed for own metrics in compiling the keras model.
def psnr(y_true, y_pred):
    
    """
    This function calculate the PSNR of predicted and true image 
    
    Parameters
    ----------
    
    y_true : array of true image
    y_pred:  array of predicted image 
    
    return
    ----------
    
    psnr of images"""        
    
    return tf_psnr.psnr(y_true, y_pred, 1.0)

def normalizeImage(imgs):
    image=numpy.clip(imgs, 0, 1.0)
    return image

def numpyToPIL(numpyImage):
    return skimage.img_as_uint(numpyImage)

def AveragePSNR(goodImages, noisyImages):
    psnrValues = [] # List of psnr values per step number of images.
    for i in range(len(goodImages)):
            psnrValues += [psnr(numpy.array(goodImages[i]), numpy.array(noisyImages[i])),]
    return psnrValues


def AverageSSIM(goodImages, noisyImages):
    ssimValues = []
    for i in range(len(goodImages)):
            ssimValues += [SSim(numpy.array(goodImages[i]), numpy.array(noisyImages[i])),]
    return  ssimValues

def compileDNN(dnn, LOSS_TYPE):
    dnn.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None)
                , loss=LOSS_TYPE
                , metrics=[psnr, SSim, LOSS_TYPE])   # ToDo Make psnr function from tf available

class SaveHistory(keras.callbacks.Callback):
    def __init__(self, outFilepath):
        self.outFilepath = outFilepath
        self.history = {}

    def on_train_begin(self, outFilepath, logs={}):
        # Dictionary of lists with metric values in order of the epochs. Keys are
        self.history = {}
        # Create output file if not already existing
        if not os.path.isfile(self.outFilepath):
            f = open(self.outFilepath, 'w')
            f.close()

    def on_epoch_end(self, epoch, logs={}):
        for key in logs:
            if key in self.history:
                self.history[key].append(logs[key])
            else:
                self.history[key] = [logs[key],]
        # Pickle losses
        with open(self.outFilepath, 'wb') as f:
            pickle.dump(self.history, f)
            
            
def safelyCreateNewDir(dirPath):
    potentialDirPath = dirPath
    dirSuffix = 1
    while True:
        try:
            os.makedirs(potentialDirPath, exist_ok=False)
            return potentialDirPath
        except OSError:
            pass
        except: # Catch any other unexpected error and re-raise it
            raise
        potentialDirPath = dirPath+'_'+str(dirSuffix)
        dirSuffix += 1
        
