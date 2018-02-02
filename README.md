## Fully convolutional Encoder-Decoder Semantic Segmentation network for detection and localisation
---
### Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [How to use it](#how-to-use-it)
4. [Sources](#sources)


### Overview

This is a Keras implementation of a generic convolutional encoder-decoder model architecture. 

The basic principles are based on the paper "Simple Does It: Weakly Supervised Instance and Semantic Segmentation" (https://arxiv.org/pdf/1603.07485.pdf). Primarily this is a similar application of an encoder-decoder network to train from the bounding boxes of object classes (i.e. weakly supervised) as opposed to fully segmented training data. 

The original structure and code is based on the SSD detection network introduced by Wei Liu at al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) and implemented by Pier Luigi Ferarri, and found at Github (https://github.com/pierluigiferrari/ssd_keras).

Though the SSD7 implementation served as the basis of the first (failed) detector attempt, and an invaluable template for implementing a Keras model, the code otherwise has largely been re-purposed for a different approach and model type. 

The core of the implementation is contained in 3 files: 
ssd8_batch_generator - which parses input data, and generates batches for training, validation and prediction. 
keras_ssd8 - which contains the architecture of the network, and any changes to layers of the CNN are made here.
train_ssd8 - this is the runnable file and, briefly, performs the following: Imports, initial parameters, initialise model, load weights (optional), initialise batch generators, train model (optional), plot prediction output.


### Dependencies
These are the versions that were used to build and test the model (GP). Backwards compatibility is uncertain. 
* Python 3.6
* Numpy
* TensorFlow r1.4
* Keras 2.1.1
* OpenCV (for data augmentation)


### How to use it

The model is executed by running the 'train_ssd8.py' file. Most parameters should be left as default, though most hyperparameters can be edited here if needed. The only things to check/change:

* If the model is to be trained from scratch, the 'load weights' line should be commented out. Otherwise a valid weight save file should be passed.
* If the aim is to perform predictions, section 7 of the code should not be executed (or commented out entirely). 

To change the structure of the model, edit the 'keras_ssd8.py' file. This should hopefully be self explanatory, but key things to note:

* On each layer, there is a term at the end in brackets. This defines what the input to this layer is (i.e. one of the previous layers). If any layers are added, these need to be edited as well, to ensure the flow of the network does not bypass the new layers. 
* Any added layers need to have the 'name=' parameter changes, or else the compiler will fail. The error message will explain the problem, but it is easily to overlook. 
* The number of filters on any convolution layer can be changed at will, but any saved weights will no longer be useable, and the network will have to be trained from scratch (as above). 
* Changing the padding options (from 'same' to 'valid') is not advised, as this will change the image size of each layer. This might cause the final upsampled image to deviate from the current 51 by 51 size, and the output will no longer match the size of the "ground truth" training data, and training will be dubious. For some reason the training algorithm does not warn you that output doesnt match input, and continues training regardless, so this needs to be tracked. Running one pass of the 'train_ssd8' file (without training) will produce a printed statement at the end with the size of both the network output and training data arrays, so this can be checked to make sure they match before embraking on any lengthy training. 
* Currently the upsampling method is very simple, and this will likely be tweaked, perhaps with another upsampling step to get a roughly 100 by 100 prediction. 

The batch generator file doesn't have any parameters that can be edited to change anything to the functioning of the model, unless:
* More significant changes need to be made to how the model reads in data (e.g. downsampling ground truth data to a different size. Currently this is 51 by 51, but can be changed in parse_pixel_map()
* This currently ignores and skips greyscale images, and needs to be extended to be able to cope with these, on the assumption that the test data will contain greyscale images (This should really be checked before we start work on editing this). 

Training and validation data structure:
* Training data (.jpg and .txt files) should be in a sub-folder ./Data/, inside the folder where this code is contained. Currently the model will use all the images in that folder to train on, and there is no way to change which images are used without actually removing them from the folder (might need to address this). 
* Validation data should be in another subfolder from training data i.e. ./Data/val. Similar to above, it uses all images found inside this folder to validate. 
* The above are the default paths to the data that I've set. These can be changed in the 'train_ssd8.py' file, as these paths are passed as arguments when intialising the batch generators. 

Output is currently only for visual validation purposes, and is just a 51 by 51 array, where each pixel is labelled with the corresponding number of the class which the network thinks it belongs to. No other output or saving is currently implemented. 


#### Training and prediction

This is currently all done in the 'train_ssd8.py' file, in a single process. 

To make multiple predictions, an interactive python console (or Jupyter) should be used to re-run the last prediction section (5.2) of code repeatedly.


### Sources

Here are some papers, etc, that I used to get my head around what to do, that might be useful to read, or at least scan. There are many more about convolutional networks and such, but I either didn't save them, or they just aren't as relevant. I should probably add the original SSD, YOLO and Various R-CNN papers (these are all networks I considered at the start)
* http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
* https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
* http://mi.eng.cam.ac.uk/~cipolla/publications/article/2016-PAMI-SegNet.pdf
* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf
* https://arxiv.org/pdf/1705.09052.pdf
* https://arxiv.org/pdf/1603.07485.pdf
* https://github.com/pierluigiferrari/ssd_keras
* https://pdfs.semanticscholar.org/2844/0d41a558011c322ffbe3feb398d4ee807751.pdf
* https://www.youtube.com/watch?v=nDPWywWRIRo&t=6s


