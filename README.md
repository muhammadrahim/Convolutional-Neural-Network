## Fully convolutional Encoder-Decoder Semantic Segmentation network for detection and localisation
---
### Contents

1. [Overview](#overview)
2. [Examples](#examples)
3. [Dependencies](#dependencies)
4. [How to use it](#how-to-use-it)
5. [ToDo](#todo)
6. [Sources](#sources)


### Overview

This is a Keras implementation of a generic convolutional encoder-decoder model architecture. 

The basic principles are based on the paper "Simple Does It: Weakly Supervised Instance and Semantic Segmentation" (https://arxiv.org/pdf/1603.07485.pdf). Primarily this is a similar application of an encoder-decoder network to train from the bounding boxes of object classes (i.e. weakly supervised) as opposed to fully segmented training data. 

The original structure and code is based on the SSD detection network introduced by Wei Liu at al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) and implemented by Pier Luigi Ferarri, and found at Github (https://github.com/pierluigiferrari/ssd_keras).

Though the SSD7 implementation served as the basis of the first (failed) detector attempt, and an invaluable template for implementing a Keras model, the code otherwise has largely been re-purposed for a different approach and model type. 

The core of the implementation is contained in 3 files: 
ssd8_batch_generator - which parses input data, and generates batches for training, validation and prediction. 
keras_ssd8 - which contains the architecture of the network, and any changes to layers of the CNN are made here.
train_ssd8 - this is the runnable file and, briefly, performs the following: Imports, initial parameters, initialise model, load weights (optional), initialise batch generators, train model (optional), plot prediction output.


### Examples

TBC - For now I think this should be filled with interesting cases (succeses, epic failures, general weirdness or anything amusing)


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


### ToDo

The following things still need to be done. 

Critical to allow us to upload results:

* Convert the 51 by 51 output from the network, into the format Per Kristian has specified (https://www.kaggle.com/c/uob-nc2017/data). As a first step it should just parse the output directly to the format required. (I think Cindy has agreed to start looking at this initially, though I think I will too as it's pretty critical). 
* It seems we upload the entire list of predictions in a single csv file, so it would also help to have some kind of loop that can perform predictions on the entire test data set, and output each result into a single continuous csv file. (This follows on from the work above). 
* If the test data does contain greyscale images, then the batch generator code needs to be fixed to be able to cope with these, as currently in will throw an error, unless these images are ignored. The key issue here is that some elements of the batch generator code (specifically some of the image augmentation methods) assume that the image is a standard 3 channel RGB image. More importantly however, I think the tensorflow model itself expects the input to the network to be batch_size by height by width by channel, so currently batch_size by 400 by 400 by 3. So even if batch generator can handle a single channel image, the network still expects 3 channels. So perhaps the single channel can be copied into the other two? (Muhammad said he'd start looking at this). 

Important to hopefully get better results:

* Prior to parsing the output of the network to the csv, I think we need to do some filtering on it, and to turn the predicted "blobs", into "boxes". How exactly we do this is very much up for debate right now. I think this needs to be context specific as well, but sort of depends on the quality of the predictions we get out of the final network. But by context specific I mean things like: For big objects like airplanes, boats, busses, trains, we can often assume there is only one of these in the image, so if there are multiple blobs of pixels, the resulting box should surround all of them. For rarely identified objects like tv monitors or chairs, if there is even a single pixel of that type in the image, its very likely it is there, and should probably be expanded and included. Other small areas or sections for more common objects should probably be filtered out. Similar networks for this purpose already use filtering to improve results, so we need to look at how this is done. Aside from the network itself, this will be the next biggest factor that will determine our accuracy. If we are going to do context specific filtering it will also involve a lot of studying of the resulting images, to see what kind of patterns it produces, where it fails, etc. 
* Network architecture: The current network I've built has been stripped down to a fairly small one, with only single convolutional layers, and a relatively small number of filters in each, so that it can be trained relatively quickly (45 min per epoch, on my machine). However it is significantly smaller than the state of the art models in literature. So we should do some experimentation to determine what the upper bound of the size of the network is, to balance training time against any potential improvement a large model will produce. 
* The upsampling layers at the moment are extremely basic, and we should see if they can do the upsampling in a cleverer way. That said, at least one of the papers that describe semantic segmentation networks like this (cant recall which) claimed they got little gain from this, and kept the basic upsampling layers. 
* We need to do some analysis on the training images. At the moment, there are at least 5 times as many images depicting people, as there are of any other category. I believe this may be counter-productive, especially at the initial training runs, and we should perhaps split the data, and remove some of the images that only depict people for initial training. Again, some of the papers about this mention dealing with class-imbalance, and should be studied a bit more. 

Optional things, if we have time (we probably wont):

* Further to the filtering above, it would be nice to be able to get accuracy ratings for each individual class. This would really help us determine where the network is performing the worst and try to account for it. Im not sure how exactly this could be done right now, but some kind of class-wise comparison of output and ground truth arrays would seem logical. Also this kind of data would look good in the report I think. 
* Currently the batch generator performs a flattening of the training "ground truth" class data, so that every pixel is only associated with a single class label. This is necessary if we're using a softmax activation on the final layer. I've defined some initial priorities based on some simple logic (e.g. large objects should have low priority, small objects should be kept at the front (high priority), rare objects should have high priority if they do not appear often in the training images (or if they are rarely correctly identified) while common objects (especially people) have low priority. However, for images with multiple overlapping classes, we could potentially shuffle the priorities between training epochs. This could be seen as an additional image augmentation (along with increasing contrast, random flipping, etc). 


Things for the report:

* We need some kind of standardised test procedure that we can use to compare different versions of the network. For example the baseline standard could be something like: Train for 5 epochs (or a set time, if using the same machine) using the full unaugmented training set. Predict accuracy on a standardised subset of validation data (preferably containing a good distribution of different classes). This prediction could be done after each training epoch, to see how it improves (or doesnt). Classwise accuracy scores would be even more ideal here. Any changes we make to the network can then be validated against this baseline. We can of course use the calculated "loss" as a benchmark as well, however this isn't very informative on its own, in my opinion. It should definitely be recorded at all times though. Also we need to present this in a "statistically rigorous way" to quote the task description from Per Kristian. I have only the vaguest idea of what this means in this context, so we need to discuss it. 
* Various things I think we can experiment with are: Network size/architecture, Non-augmented vs augmented training, full training set vs reduced set to address class imbalance, different gradient descent algorithms or loss models, learning rate (though this sort of goes hand in hand with gradient descent type). I'm sure there are other possibilities, but we probably don't have time to do them all, so we need to decide which to focus on. 
* Keep a good record of results, and remember to save instances of images that show something interesting! I have unfortunately forgotten to do this for all my experimentation with the previous SSD model, coz I am daft. Except for one image set that shows how the network thought all animals were cats, which is amusing, but its inclusion in the report is ... up to you guys. 

Experimentation on the SSD implementation:

Before this current encoder-decoder type network, I was working on trying to get the SSD (Single shot detector) architecture to work. I'm not going to write up what I found in detail yet (coz it's late), but in summary:

* The SSD requires bounding boxes as training input (in four coordinates), which is not what we have been given. Though I coded a fairly elaborate algorithm that processed the data we have into bounding boxes, it still struggled to seperate overlapping boxes from the same class (e.g. there are many instances where 4-5 people are overlapping, with no distinction in the raw data between them). So results we're varied, and I ended up having to cull a reasonable chunk of training images, due to the bounding boxes being suspect. I know other teams are using bounding box based detector networks, and I am genuinely curious how they are generating their training bounding boxes, or if they found a way around it, coz I didn't. 
* With the data that did succesfully convert to bounding boxes my training results were pretty dissappointing. Admitedly I trained for only about 10 epochs maximum, but often because the training algorithm would stop early, because the loss would stop improving. Initially I thought this was due to the first design (7 layers) being too small to succesfully predict 20 classes, however I also tried the 16 layer (VGG16) network, and the results were not really any better. I am not 100% sure what was wrong, but I believe most of it is from the rather small training set, dubious input bounding boxes, and a generally complicated network implementation that I can't pretend I fully understood it or succesfully ironed out the bugs in it. 
* As the network design is quite complicated, with a custom loss function, and a great deal of custom and complicated code on encoding and decoding "default bounding boxes" I felt there was very little that could be experimented with in this model, nor much that we could really customise. This may be a bias on my part, but apart from the bounding box creator, I was simply using someone elses code, and despite an awful lot of time working on it, I didn't feel like there was any achievement or results to write about. 
* If I kept at it, and tweaked it, the results may have been better than what we have now, but there was no gaurantee. Initial results from the segmentation model (that took a fraction of the time to implement) were already better, so I made the executive decision to abandon SSD and work on this version. It's simpler, uses all training data directly, is a more suitable selection to suit the training data format (rather than shoehorning the training data into the format the network accepts), and it generally allows us to experiment more deeply and hopefully have some interesting things to say in the report. 

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


