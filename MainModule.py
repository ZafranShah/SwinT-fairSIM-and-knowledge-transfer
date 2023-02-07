#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:37:02 2022

@author: shah
"""

import tensorflow as tf
from Models import UNetfairSIM, ModelFineTunning
from DataGeneratorPreprocessing import CustomDataGenerator
from Utils import compileDNN, safelyCreateNewDir
import os, datetime,argparse, json
from EvaluationModule import PropagateTestImages
import ImageHelpers
from tensorflow import keras



def ModelTraining(imageDirectory, n_GPUS, n_Dim,n_Channel, lossType, modelName, Epochs, trainingfromScratch=True, pretrainedModel='', finetuneStartingpoint=0, finetuneEndingpoint=0):
    """This is the main function to train the models from scratch or fine-tune model

    Parameters:
    ----------  
    imageDirectory: Path of the folder contain images
    n_GPUS: number of GPUs
    n_Dim, n_Channel: dimension & channel of input & ouput images
    lossType: type of loss function you want to use during training process
    modelName: Assign the name of the model you want to save
    Epochs: number of epochs
    trainingfromScratch: Whether you want to start the training from scratch or fine-tune the pretrained model default= True
    pretrainedModel: path of the pretrained model
    finetuneStartingpoint, finetuneEndingpoint: layers you want freeze during the fine-tunning process

    Returns: 
    ----------  
    return: stored the final results on the disk
        """
    train_x_dir, train_y_dir, test_x_dir, test_y_dir = ImageHelpers.LoadDataForGenerators(imageDirectory)
    BATCH_SIZE= 2* n_GPUS
    training_data_generator=CustomDataGenerator(train_x_dir, train_y_dir,BATCH_SIZE)
    validation_data_generator=CustomDataGenerator(test_x_dir, test_y_dir,BATCH_SIZE)
    
    print('Build model')
    
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        if trainingfromScratch:
            print ('Training a model from scratch')
            Createmodel=UNetfairSIM(n_Dim,n_Dim,n_Channel)
            model=Createmodel.buildUNetwork()

        else:
            print ('Loading a pretrained model from the directory')
            Loadmodel=ModelFineTunning(pretrainedModel, finetuneStartingpoint, finetuneEndingpoint)
            model=Loadmodel.PretrainedModel()

        compileDNN(model, lossType)

    if not os.path.exists('models'):
        os.makedirs('models')
    
       
    trainingRunDirPathTemp = os.path.join('models', modelName)
    trainingRunDirPathTemp = safelyCreateNewDir(trainingRunDirPathTemp)
    callbackPath = os.path.join(trainingRunDirPathTemp, 'callbacks')
    os.mkdir(callbackPath) # Create weight directory inside the temporary directory
    bestModel=os.path.join(callbackPath, 'best_model.h5')
    bestCheck = keras.callbacks.ModelCheckpoint(bestModel,
                                         monitor='val_SSim', mode='max',
                                         save_best_only=True, save_weights_only=True,
                                         period=1)  
 

    trainingHistory =model.fit(training_data_generator, validation_data=(validation_data_generator), batch_size=BATCH_SIZE,epochs=Epochs, callbacks=[bestCheck])


    trainingMetrics = model.evaluate(training_data_generator,batch_size=BATCH_SIZE,verbose=1)

    testingMetrics = model.evaluate(validation_data_generator,batch_size=BATCH_SIZE,verbose=1)
    
    
    
    # Generate name-prefix for saving files.
    now = datetime.datetime.now()
    dateStr = '{0:02d}-{1:02d}-{2:04d}_{3:02d}-{4:02d}'.format(now.day, now.month, now.year, now.hour, now.minute)
    fileNamePrefix = modelName +'_'+dateStr
    
    trainingRunDirPath = os.path.join(trainingRunDirPathTemp +'/models', modelName)
    trainingRunDirPath = safelyCreateNewDir(trainingRunDirPath)  
    
    
    
    # Save model information
    with open(os.path.join(trainingRunDirPath, fileNamePrefix + '-modelInfo-'), 'w') as file:
        info = {}

        info['title'] = modelName
        info['epochs'] = Epochs
        info['batch_size'] = BATCH_SIZE
        info['lossType'] = lossType
        info['trainingMetrics'] = {str(lossType) : trainingMetrics[0], 'psnr': trainingMetrics[1], 'ssim': trainingMetrics[2],
                                str(lossType) : trainingMetrics[3]}
        info['testingMetrics'] = {str(lossType) : testingMetrics[0], 'psnr': testingMetrics[1], 'ssim': testingMetrics[2],
                              str(lossType): testingMetrics[3]}

        json.dump(str(info), file)
        
            # Save trained model
    modelFilePath = os.path.join(trainingRunDirPath, fileNamePrefix + '-model')
    model.load_weights(bestModel) # Save model
    model.save(modelFilePath)
    newDir = safelyCreateNewDir(os.path.join('Results', fileNamePrefix))
    propagatedImagesDir = os.path.join(newDir, 'propagated_test_images' )
    os.mkdir(propagatedImagesDir)

    
    PropagateTestImages(modelFilePath,propagatedImagesDir,imageDirectory)
    
    print ('Code is finished without any errors')

    

def EvaluateModel(modelFilePath,imageDirectory):
    """This function can be used to evaluate the pretrained model and to reproduce the results of test samples
    Parameters:
    ----------  
    imageDirectory: Path of the folder contain images
    modelFilePath: path of the pretrained model

    Returns: 
    ----------  
    stored the final results on the disk
        """
    
    print ('Creating a result folder')    
    if not os.path.exists('Results'):
        os.makedirs('Results')
    
    PropagateTestImages(modelFilePath,'Results',imageDirectory)
    print ('The code is executed without any Error for test module')
    



def LoadParameters(PathofJsonFile):
    with open(PathofJsonFile) as paramfile:
        data=json.load(paramfile)
        return data



def Main():
    parser= argparse.ArgumentParser()

    parser.add_argument('PathofJsonFile', help= 'Enter the of the configuration file')
    args = parser.parse_args()
    return args

arguements=Main()

if len(arguements.PathofJsonFile) !=0:
    JsonParameters=LoadParameters(arguements.PathofJsonFile)
    Image_dir=JsonParameters['ImageDirectory']
    Image_dimension=JsonParameters['imageSize']
    Image_channel=JsonParameters['imageChannel']
    LOSS_type =JsonParameters['TypeofLossFunction']    
    MODEL_name =JsonParameters['ModelName']
    N_gpus=JsonParameters['NumberofGPUs']
    Epochs=JsonParameters['Epochs']
    training_from_Scratch= JsonParameters['TrainNetworkfromScratch']
    Activate_train_and_test_module_only= JsonParameters['ActivateTrainandTestModule']
    Pretrained_model= JsonParameters['PretrainedmodelPath']
    FineTune_startingpoint= JsonParameters['FineTuneStartingpoint']
    FineTune_endingpoint= JsonParameters['FineTuneEndingpoint']

    if Activate_train_and_test_module_only:
        ModelTraining(Image_dir,N_gpus,Image_dimension,Image_channel, LOSS_type,MODEL_name,Epochs,training_from_Scratch,Pretrained_model,FineTune_startingpoint,FineTune_endingpoint)
        print ('All results are computed and stored')
    else:
        print ('Only to regenerate the results')
        EvaluateModel(Pretrained_model,Image_dir)
        
else:
    print ('path to parameter json file is missing and code contains Error')
    




if __name__ == "__main__":
    
    Main()

    
    
