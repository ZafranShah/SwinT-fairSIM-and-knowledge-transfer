#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:15:47 2022

@author: shah
"""

import os
from PIL import Image
import numpy as np
from Utils import normalizeImage, AveragePSNR,psnr,SSim, AverageSSIM, numpyToPIL
import ImageHelpers
import tensorflow as tf


def PropagateTestImages(modelPath,outputDir, imageDirectory):
    """This function can be used to denoise the test samples
    

    Parameters:
    ----------    
    model: model to evaluate
    outputdir: The path of folder where the results will be stored
    imageDirectory: Path of input and output test samples

    Returns: 
    ----------    
    save the denoised images along with input & output images as well as the text files that contains the PSNR/SSIM values on the disk.
        """
       
    
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)
    outputdir1 =outputDir +'/noisy'
    outputdir2 =outputDir +'/reconstructed'
    outputdir3 =outputDir +'/original'
    if not os.path.isdir(outputdir1):
        os.mkdir(outputdir1)
    if not os.path.isdir(outputdir2):
        os.mkdir(outputdir2)
    if not os.path.isdir(outputdir3):
        os.mkdir(outputdir3)
        
        
    testx, testy = ImageHelpers.LoadData(imageDirectory, False)

    model = tf.keras.models.load_model(modelPath, custom_objects ={'psnr':psnr, 'SSim':SSim})         
    print ('Summary of the model which is loaded for the test purpose')
    model.summary()


    metrics = []  # Array of every training metrics for the corresponding image in propagatedImgs.
    noisyPsnr=[]
    reconstructedPsnr=[]
    reconstructedssim=[]
    psnrImprovement=[]
    noisyssim=[]
    ssimImprovement=[]
    ssimvalue=[]
    width, height= tuple(testx[0].shape[:2])
    metrics = [] 
    for i in range(len(testx)):

        input_image = testx[i].reshape(1,width, height,1)
        output_image = testy[i].reshape(1,width, height,1)
        
        
        denoised_image = np.array(model.predict(input_image, batch_size=1))
        denoised_image = denoised_image.reshape(1,width, height,1)
        output_image = output_image.reshape(1,width, height,1)
        #metrics += [model.evaluate(input_image, output_image, batch_size=1),]
        #metrics +=[psnr(output_image,denoised_image)]

        denoised_image = normalizeImage(denoised_image)
        output_image = normalizeImage(output_image)
        input_image = normalizeImage(input_image)
#

        noisyPsnrvalue=AveragePSNR( output_image, input_image)
        noisyPsnr += [noisyPsnrvalue,]
        psnrValues= AveragePSNR( output_image, denoised_image)
        reconstructedPsnr += [psnrValues,]
        ssimvalues=AverageSSIM( output_image, denoised_image)
        reconstructedssim += [ssimvalues,]
        noisyssimvalue=AverageSSIM( output_image, input_image)
        noisyssim +=[noisyssimvalue,]
        ssimdiff= np.array(ssimvalues) - np.array(noisyssimvalue)
        ssimvalue += [ssimvalue,]
        psnrdiff= np.array(psnrValues) - np.array(noisyPsnrvalue)
        psnrImprovement += [psnrdiff,]
        ssimImprovement += [ssimdiff,]
        
        # Convert np.array images to PIL images
        input_image = numpyToPIL(np.reshape(input_image, (width, height)))
        denoised_image = numpyToPIL(np.reshape(denoised_image, (width, height)))
        output_image = numpyToPIL(np.reshape(output_image, (width, height)))
        
        input_image=Image.fromarray(input_image.astype(np.uint16))
        denoised_image=Image.fromarray(denoised_image.astype(np.uint16))
        output_image=Image.fromarray(output_image.astype(np.uint16))
        input_image.save(outputdir1 +'/Noisy_Image_'+str(i)+'.tif')
        denoised_image.save(outputdir2 +'/Reconstructed_Image_'+str(i)+'.tif')
        output_image.save(outputdir3 +'/Original_Imag_'+str(i)+'.tif')



    with open(os.path.join(outputDir, 'metrics.txt'), 'w') as f:
        for i, e in enumerate(metrics):
            f.write('Image'+str(i+1)+': '+str(e)+'\n')  
                
    with open(os.path.join(outputDir, 'metrics_2.txt'), 'w') as f:
        for i, (d, e, e1, f1, f2, f3) in enumerate(zip(psnrImprovement, reconstructedPsnr, noisyPsnr,ssimImprovement, reconstructedssim, noisyssim)):
            f.write('Image'+str(i+1)+': '+': '+ str(d)+': '+str(e)+': '+str(e1)+': '+str(f1)+': '+str(f2)+': '+str(f3)+'\n')
            if (i+1) == len(reconstructedPsnr):
                meanSSIM = np.sum(np.array(reconstructedssim))/len(reconstructedssim)
                meanPSNR = np.sum(np.array(reconstructedPsnr))/len(reconstructedPsnr)
                meanPSNRnoisy = np.sum(np.array(noisyPsnr))/len(noisyPsnr)
                meanSSIMnoisy = np.sum(np.array(noisyssim))/len(noisyssim)
                f.write('Average_Psnr_denoised_images_ : ' + str(meanPSNR)+' \n' +'Average_PSNR_noisy_Images_ : '+ str(meanPSNRnoisy) + '\n' + 'Average_SSim_denoised_images_ : ' +str(meanSSIM) + '\n'+'Average_SSim_Noisy_images_ : ' +str(meanSSIMnoisy))
                f.write('-'*10+'\n')
            
    print ('The results are stored onto the disk and the code is finished without errors')


