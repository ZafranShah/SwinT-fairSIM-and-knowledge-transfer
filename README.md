# Evaluation of Swin Transformer and knowledge transfer for denoising of super-resolution structured illumination microscopy data

In recent years, convolutional neural network (CNN)-based methods have shown remarkable performance in the denoising and reconstruction of super-resolved structured illumination microscopy (SR-SIM) data. Therefore, CNN-based architectures have been the main focus of existing studies. Recently, however, an alternative and highly
competitive deep learning architecture, Swin Transformer, has been proposed for image restoration tasks. In this work, we present SwinT-fairSIM, a novel method for restoring SR-SIM images with low signal-to-noise ratio (SNR) based on Swin Transformer. The experimental results show that SwinT-fairSIM outperforms previous CNN-based denoising methods. Furthermore, the generalization capabilities of deep learning methods for image restoration tasks on real fluorescence microscopy data have not been fully explored yet, i.e., the extent to which trained artificial neural networks are limited to specific types of cell structures and noise. Therefore, as a second contribution, we benchmark two types of transfer learning, i.e., direct transfer and fine-tuning, in combination with SwinT-fairSIM and two CNN-based methods for denoising SR-SIM data. Direct transfer does not prove to be a viable strategy, but fine-tuning achieves results comparable to conventional training from scratch while saving computational time and potentially reducing the amount of required training data. As a third contribution, we published four datasets of raw SIM images and already reconstructed SR-SIM images. These datasets cover two types of cell structures, tubulin filaments and vesicle structures. Different noise levels are available for the tubulin filaments. These datasets are structured in such a way that they can be easily used by the research community for research on denoising, super-resolution, and transfer learning strategies.

The SIM microscopy datasets that were used during this work can be downloaded through this link: http://dx.doi.org/10.5524/102461  

## Installation:

This implementation requires the Tensorflow-GPU2.5 version. To avoid package conflicts, we recommend you create a new environment by using our provided environment.yml file. To create a new environment please run the following script:

>  conda env create -f environment.yml

## How to use this code:

This code can be used to train a denoising model from scratch or to fine-tune a pretrained model. After the installation of the Python environment from the yml file, the next step is to set the input parameters in the JSON parameter file (i.e., ParameterFile.json). Most of the input parameters are self-explanatory but below we will discuss some of the important input parameters from the JSON file:

- TrainNetworkfromScratch: This input parameter will train the model from scratch If set to True, otherwise, for fine-tuning, It should be False.
- ActivateTrainandTestModel: This parameter will be set to False If you want to use this code for evaluation of the trained model or the reproducibility of the results by using pretrained models.
- PretrainedmodelPath: This parameter is mandatory in case of fine-tuning or evaluation of a pretrained model.
- FineTuneStartingpoint and FineTuneEndingpoint: These two input parameters are essential in the fine-tuning of a pretrained model. All the layers between the starting and ending points will be frozen during the fine-tuning of the pretrained model.

After the assignment of the input parameters. You can run the following script from the command line to start training the model:

> python MainModule.py 'ParameterFile.json'

## Reproducibility and evaluation:

To reproduce the results of the paper all the trained models used in this work are available in the 'Models' directory at [zenodo](https://doi.org/10.5281/zenodo.7626173). This code is capable of performing all the necessary steps for the training and test phases. It will automatically evaluate the model and generate a result directory to write all the results. Similarly, during the training process, It will also create a model directory and save the trained model along with the best checkpoints in the model directory.   

## Important Note:

This code will work with at least one GPU.

## Reference:

Zafran Hussain Shah, Marcel Müller, Wolfgang Hübner, Tung-Cheng Wang, Daniel Telman, Thomas Huser, Wolfram Schenck, Evaluation of Swin Transformer and knowledge transfer for denoising of super-resolution structured illumination microscopy data, GigaScience, Volume 13, 2024, giad109, https://doi.org/10.1093/gigascience/giad109



