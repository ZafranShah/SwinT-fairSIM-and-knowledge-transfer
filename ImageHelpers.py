# Description: Helper functions for handling, modifying and filtering images.

import numpy as np
import math, os
import json
import time
import copy
import random
import numpy 
from PIL import Image
from skimage import io







def RescaleImgs(imgs):
    """
    Linear rescale over all images in imgs to the range [0,1] so the values of 0.0 and 1.0 exist in the image-list.
    Parameters:
    ----------
    imgs: List of images as numpy arrays. This list is going to be emptied.
 
    Returns: 
    ----------
    New list of rescaled images. Same ordering as the argument imgs.

"""    
    
    newImgs = []
    min = np.amin(imgs)
    max = np.amax(imgs)
    while len(imgs) != 0:
        img = (imgs[0] - min) * (1/(max-min))
        newImgs.append(img)
        del imgs[0]
    return newImgs

def rescaleSingleImgs(imgs):
    min = np.amin(imgs)
    max = np.amax(imgs)
    newImgs = (imgs - min) * (1/(max-min))
    return newImgs

def Flip(imgs):
    
    """
    This function generate the flipped version of the images inserted after their respective unflipped version.
    Parameters:
    ----------    
    imgs: List of images as numpy arrays. This list is going to be emptied.
    
    
    Returns: 
    ----------    
    New image list with normal and flipped images. Same ordering as the argument imgs.    
    """
    newImgs = []
    while len(imgs) != 0:
        newImgs.append(imgs[0])
        newImgs.append(np.fliplr(imgs[0]))
        del imgs[0]

    return newImgs



def NormalizeImage(array, maxValue=255):
    """
    This function generate the normalize version of images with pixel values [0,255] 
    Parameters:
    ----------    
    array: image as numpy arrays
    
    
    Returns: 
    ----------    
    New normalized image.    
    """
    
    array = np.array(array, dtype=np.float64)
    array /= maxValue
    return array


# Reshape of the images into tensors

def ReshapeData(noisyImages, goodImages):
    
    """
    This function reshape the reshape the input and output images into tensors 
    Parameters:
    ----------    
    noisyImages, goodImages: list of input and output images    
    Returns: 
    ----------    
    tensor of input and output images    
    """
    IMG_RES = tuple(noisyImages[0].shape[:2])

    noisyImages, goodImages= numpy.array(noisyImages), numpy.array(goodImages)
    noisyImages = numpy.reshape(noisyImages, (noisyImages.shape[0],) + IMG_RES + (1,))
    goodImages = numpy.reshape(goodImages, (goodImages.shape[0],) + IMG_RES + (1,))
    return noisyImages, goodImages
    
    
    
def LoadImages(listofImagepath):
    """
    This function load the images from the disk and this function is called in Load data function
    Parameters:
    ----------    
    listofImagepath: list of images path
    
    Returns: 
    ----------    
    images    
    """
    images=[]
    for img in listofImagepath:
        img_arr=numpy.array(io.imread(img))
        images.append(img_arr)
    return images




def ExtractPath(pathofFile):
    """
    This function Extract and sort the path of the images and this function is called in Load data function
    Parameters:
    ----------    
    pathoffile: path of the folder contain images
    
    Returns: 
    ----------    
    the sorted path of all the images in that folder    
    """
    listofimages=[]
    for img in os.listdir(pathofFile):
        pathofimage= os.path.join(pathofFile, img)
        listofimages.append(pathofimage)
        listofimages.sort()
    return listofimages



def LoadData(imageDirectory, traindata=True):
    
    """Extract and sort path of data
    Parameters:
    ----------    
    imageDirectory: path of the folder contain subfolders of training and test data
    traindata: default = True 
    Returns: 
    ----------    
    lists of either training & test samples or list of only test samples if Traindata != True  
    """
    

    print('Loading data path')
    if traindata:
        print('Loading train and test data ')
        trainxPath= os.path.join(imageDirectory,'trainx')    
        trainyPath= os.path.join(imageDirectory,'trainy')
        testxPath= os.path.join(imageDirectory,'testx')    
        testyPath= os.path.join(imageDirectory,'testy')
        trainXpath= ExtractPath(trainxPath)
        trainYpath= ExtractPath(trainyPath)
        testXpath= ExtractPath(testxPath)
        testYpath= ExtractPath(testyPath)
        trainDataX= LoadImages(trainXpath)
        trainDataY= LoadImages(trainYpath)
        testDataX= LoadImages(testXpath)
        testDataY= LoadImages(testYpath)
        trainDataX = RescaleImgs(trainDataX)
        trainDataY = RescaleImgs(trainDataY)
        testDataX = RescaleImgs(testDataX)
        testDataY = RescaleImgs(testDataY)
        trainDataX, trainDataY= ReshapeData(trainDataX, trainDataX)   
        testDataX, testDataY= ReshapeData(testDataX, testDataY)   
        return trainDataX, trainDataY, testDataX, testDataY
    else:
        print('Loading test data only')
        testxPath= os.path.join(imageDirectory,'testx')    
        testyPath= os.path.join(imageDirectory,'testy')
        testXpath= ExtractPath(testxPath)
        testYpath= ExtractPath(testyPath)
        testDataX= LoadImages(testXpath)
        testDataY= LoadImages(testYpath)
        testDataX = RescaleImgs(testDataX)
        testDataY = RescaleImgs(testDataY)
        testDataX, testDataY= ReshapeData(testDataX, testDataY)   
        print ('Length of test_x: ', len(testDataX),' and test_y: ', len(testDataY))
        return testDataX, testDataY
        


# This function load the test data only for evalzuation purpose 


def LoadDataForGenerators(imageDirectory):
    
    """Extract and sort path of training and test data for the custom generator to train a model
    Parameters:
    ----------    
    imageDirectory: path of the folder contain subfolders of training and test data
    
    Returns: 
    ----------    
    lists of paths of training & test data  
    """



    print('Loading data path')

    
    trainxPath= os.path.join( imageDirectory,'trainx')    
    trainyPath= os.path.join( imageDirectory,'trainy')
    testxPath= os.path.join( imageDirectory,'testx')    
    testyPath= os.path.join( imageDirectory,'testy')	



    trainXpath= ExtractPath(trainxPath)
    trainYpath= ExtractPath(trainyPath)
    testXpath= ExtractPath(testxPath)
    testYpath= ExtractPath(testyPath)
    print ('Length of trainx: ', len(trainXpath),' and length of trainy: ', len(trainYpath))

    return trainXpath[:], trainYpath[:], testXpath[:], testYpath[:]



