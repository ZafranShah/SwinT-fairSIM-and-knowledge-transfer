# Evaluation of knowledge transfer for the denoising of super-resolution structured illumination microscopy data

In recent years, convolutional neural network (CNN) based methods have shown remarkable performance in the denoising and reconstruction of super-resolved structured illumination microscopy (SR-SIM) data. The potential for the generalization of these deep-learning models to different real-world fluorescence microscopy data has, however, not yet been completely explored. The question arises whether such CNN-based denoising methods are structure- and noise specific. Here, we apply transfer learning and fine-tuning strategies to assess the generalization capability of these methods to different domains, in particular super-resolved microscopy images of different biological structures, before and after knowledge transfer. We extensively investigate the performance of CNN-based denoising networks on data from different domains (i.e. different biological structures) by applying transfer learning strategies, such as direct transfer and fine-tuning. In the last step, we demonstrate that the fine-tuning approach is more advantageous than the conventional training of CNN-based denoising methods to avoid computational overhead.

The SIM microscopy datasets which were used during this work can be downloaded through this link: [Link will be uploaded soon]  

## Installation:

This implementation requires the Tensorflow-GPU2.5 version. To avoid package conflicts, we recommend you to create a new environment by using our provided environment.yml file. To create a new environment please run the following script:

>  conda env create -f environment.yml

## How to use this code:

This code can be used to train a denoising model from scratch or to fine-tune a pretrained model. After the installation of the Python environment from the yml file, the next step is to set the input parameters in the JSON parameter file (i.e., ParameterFile.json). Most of the input parameters are self-explanatory but below we will discuss some of the important input parameters from the JSON file:

- TrainNetworkfromScratch : This input parameter will train the model from scratch if set to True, otherwise, for fine-tuning, it should be False.
- ActivateTrainandTestModel : This parameter will be set to False if you want to use this code for evaluation of pretrained model.
- PretrainedmodelPath : 'Path of pretrained model', this parameter is mandatory in case of fine-tuning or evaluation of a pretrained model.
- FineTuneStartingpoint, FineTuneEndingpoint : These two input parameters are essential in the case of the fine-tuning process. All the layers between the starting and ending points will be frozen during the fine-tuning of the pretrained model.

After the assignment of the input parameters. You can run the following script from the command line to start training the model:

> python MainModule.py 'ParameterFile.json'

## Reproducibility and evaluation:

To reproduce the results of the paper all the pretrained models are available in the model directory. This code is capable to perform all the necessary steps for the training and test phases. It will automatically evaluate the model and generate a result directory to write all the results. Similarly, during the training process, It will also create a model directory and save the trained model along with the best checkpoints in the model directory.   

## Important Note:

This code will work with at least one GPU.

## Reference:

Please cite our paper in case you use this code for any scientific publication. We will soon upload the citation index!




