import IPython
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('/Users/User/Downloads/group17UoBINC-fcn-e65948b6ad20')

#get_ipython().run_line_magic('matplotlib', 'inline') # DOESNT WORK IN ECLIPSE. UNCOMMENT IN JUPYTER.
import h5py

from keras_ssd8 import build_model
from ssd8_batch_generator import BatchGenerator
from tensorflow.python import debug as tf_debug


''' NOTE(GP) This is the original SSD introduction. Kept here for posterity?
# ### 1. Introduction and building the model
# 
# The cell below sets a number of parameters that define the model architecture and then calls the function `build_model()` to build the model. Read the comments and the documentation, but here are some further explanations for a few parameters:
# 
# * Set the height, width, and number of color channels to whatever you want the model to accept as image input. This does not have to be the actual size of your input images! However, if your input images have a different size than you define as the model input here, you must use the `crop`, `resize` and/or `random_crop` features of the batch generator to convert your images to the model input size during training. If your dataset contains images of varying size, like the Pascal VOC datasets for example, use the `random_crop` feature of the batch generator to cope with that (see the documentation).
# * The number of classes includes the background class, i.e. if you have `n` positive classes in your dataset, set `n_classes = n + 1`. Class ID 0 must always be reserved for the background class, i.e. your positive classes must have positive integers as IDs.
# * The reason why the list of scaling factors has 5 elements even though there are only 4 predictor layers in this model is that the last scaling factor is used for the second aspect-ratio-1 box of the last predictor layer. See the documentation for details.
# * Alternatively to passing an explicit list of scaling factors, you could also just define a mimimum and a maximum scale, in which case the other scaling factors would be linearly interpolated. If you pass both min/max scaling factors and an explicit list, the explicit list will be used.
# * `build_model()` and `SSDBoxEncoder` have two arguments for the anchor box aspect ratios: `aspect_ratios_global` and `aspect_ratios_per_layer`. You can use either of the two. If you use `aspect_ratios_global`, then you just pass a list containing all aspect ratios for which you would like to create anchor boxes. Every aspect ratio you want to include must be listed once and only once. If you use `aspect_ratios_per_layer`, then you pass a list containing lists of aspect ratios for each individual predictor layer. In the example below, the model has four predictor layers, so you would pass a list containing four lists.
# * If `two_boxes_for_ar1 == True`, then two boxes of different size will be created for aspect ratio 1 for each predictor layer. See the documentation for details.
# * If `limit_boxes == True`, then the generated anchor boxes will be limited so that they lie entirely within the image boundaries. This feature is called 'clip' in the original Caffe implementation. Even though it may seem counterintuitive, it is recommended **not** to clip the anchor boxes. According to Wei Liu, the model performs slightly better when the anchors are not clipped.
# * The variances are scaling factors for the target coordinates. Leaving them at 1.0 for each of the four box coordinates means that they have no effect whatsoever. Decreasing them to below 1.0 **upscales** the gradient for the respective target box coordinate.
# * The `coords` argument lets you choose what coordinate format the model should learn. If you choose the 'centroids' format, the targets will be converted to the (cx, cy, w, h) coordinate format used in the original implementation. If you choose the 'minmax' format, the targets will be converted to the coordinate format (xmin, xmax, ymin, ymax). The model, of course, will learn whatever the targets tell it to.
# * `normalize_coords` converts all absolute ground truth and anchor box coordinates to relative coordinates, i.e. to coordinates that lie within [0,1] relative to the image height and width. Whether you use absolute or relative coordinates has no effect on the training - the targets end up being the same in both cases. The main reason why the original implementation uses relative coordinates is because it makes coding some box operations more convenient. This defaults to `False`.
# 
# These paramters might be a bit much at first, but they allow you to configure many things easily.
# 
# The parameters set below are not only needed to build the model, but are also passed to the `SSDBoxEncoder` constructor in the subsequent cell, which is responsible for matching and encoding ground truth boxes and anchor boxes during training. In order to do that, it needs to know the anchor box specifications. It is for the same reason that `build_model()` does not only return the model itself, but also `predictor_sizes`, a list of the spatial sizes of the convolutional predictor layers - `SSDBoxEncoder` needs this information to know where the anchor boxes must be placed spatially.
# 
# The original Caffe implementation does pretty much everything inside a model layer: The ground truth boxes are matched and encoded inside [MultiBoxLossLayer](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/multibox_loss_layer.cpp), and box decoding, confidence thresholding and non-maximum suppression is performed in [DetectionOutputLayer](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/detection_output_layer.cpp). In contrast to that, in the current form of this implementation, ground truth box matching and encoding happens as part of the mini batch generation (i.e. outside of the model itself). To be specific, the `generate()` method of `BatchGenerator` calls the `encode_y()` method of `SSDBoxEncoder` to encode the ground truth labels, and then yields the matched and encoded target tensor to be passed to the loss function. Similarly, the model here outputs the raw prediction tensor. The decoding, confidence thresholding, and non-maximum suppression (NMS) is then performed by `decode_y2()`, i.e. also outside the model. It's (almost) the same process as in the original Caffe implmentation, it's just that the code is organized differently here, which likely has performance implications, but I haven't measured it yet. I might look into incorporating all processing steps inside the model itself, but for now it was just easier to take the non-learning-relevant steps outside of Keras/Tensorflow. This is one advantage of Caffe: It's more convenient to write complex custom layers in plain C++ than to grapple with the Keras/Tensorflow API.

# ### Note:
# 
# The example setup below was used to train SSD7 on two street traffic datasets released by [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) with around 20,000 images in total and 5 object classes (car, truck, pedestrian, bicyclist, traffic light), although the vast majority of the objects are cars. The original datasets have a constant image size of 1200x1920 RGB. I consolidated the two datasets, removed a few bad samples (although there are probably many more), and resized the images to 300x480 RGB, i.e. to one sixteenth of the original image size. In case you'd like to train a model on the same dataset, you can find the consolidated and resized dataset I used [here](https://drive.google.com/file/d/0B0WbA4IemlxlT1IzQ0U1S2xHYVU/view?usp=sharing).
'''


### Set up the model

# 1: Set some necessary parameters

img_height = 400 # Height of the input images
img_width = 400 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 21 # Number of classes including the background class

# 2: Build the Keras model (and possibly load some trained weights)

K.clear_session() # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
model = build_model(image_size=(img_height, img_width, img_channels),
                                      n_classes=n_classes)

''' NOTE(GP) Comment this out if you wish to train a new model. Change filename if you want to switch to alternate, or old, weights
'''
#model.load_weights('./saves/ssd8_1_ssd81.h5')
#gmodel = load_model('./saves/ssd7_1_ssd1.h5') #


# ### 2. Set up the training
# 
# The cell below sets up everything necessary to train the model. The only things you have to set are the correct file paths to the images and labels in your dataset, and in case your labels do not come in a CSV file, you might have to switch from the CSV parser to the XML parser or you might have to write a new parser method in the `BatchGenerator` class that can handle whatever the format of your labels is. The README of this project provides an overview of the design of the batch generator class, which should help you in case you need to write a new parser or adapt one of the existing parsers to your needs.
# 
# For everything in this cell that does not concern loading your data: You don't have to change anything (but you can change everything of course).
# 
# Set the batch size to whatever value you like (and one that makes the model fit inside your GPU memory), it's not the most important hyperparameter - 32 works well, but so do most other batch sizes.
# 
# I'm using an Adam optimizer with the standard initial learning rate of 0.001 and a small decay, nothing special.


### Set up training

batch_size = 32

# 3: Instantiate an Adam optimizer and compile the model

sgd = SGD(lr=0.0003, momentum=0.9, decay=5e-05, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy')

# 4: NOTE(GP) This step has been removed entirely, as it related to SSD box encoding

# 5: Create the training set batch generator

train_dataset = BatchGenerator(images_path='./data/',shuffle=True, shuffleLength=640) # 

train_dataset.parse_data(labels_path='./data/',filenamecsv='LabelsMaster.npy') # 

# Change the online data augmentation settings as you like NOTE(GP) we should experiment with data augmentation here
train_generator = train_dataset.generate(batch_size=batch_size,
                                         train=True,
                                         equalize=False,
                                         brightness=False,#(0.5, 2, 0.5), # Randomly change brightness between 0.5 and 2 with probability 0.5
                                         flip=False,#0.5, # Randomly flip horizontally with probability 0.5
                                         translate=False,#((5, 50), (3, 30), 0.5), # Randomly translate by 5-50 pixels horizontally and 3-30 pixels vertically with probability 0.5
                                         scale=False,#(0.75, 1.3, 0.5), # Randomly scale between 0.75 and 1.3 with probability 0.5
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         diagnostics=False)

n_train_samples = train_dataset.get_n_samples()

# 6: Create the validation set batch generator (if you want to use a validation dataset)

val_dataset = BatchGenerator(images_path='./data/', shuffle=True, shuffleLength=2058)

val_dataset.parse_data(labels_path='./data/',filenamecsv='LabelsMaster.npy')

val_generator = val_dataset.generate(batch_size=batch_size,
                                     train=True,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     diagnostics=False)

n_val_samples = val_dataset.get_n_samples()
print('Number of training samples: ',n_train_samples)
print('Number of validation samples: ',n_val_samples)


# ### 3. Run the training
# 
# Now that everything is set up, we're ready to start training. Set the number of epochs and the model name, the weights name in `ModelCheckpoint` and the filepaths to wherever you'd like to save the model. There isn't much more to say here, just execute the cell. If you get "out of memory" errors during training, reduce the batch size.
# 
# Training currently only monitors the validation loss, not the mAP.
runtraining = 5
## Run training
if runtraining == 5:
## 6: Run training
#NOTE(GP) UNCOMMENT THIS BLOCK TO TRAIN THE MODEL!! It is currently commented to test predictions, and the code for that is below
    epochs = 5
     
    history = model.fit_generator(generator = train_generator,
                                  steps_per_epoch = ceil(n_train_samples/batch_size),
                                  epochs = epochs,
                                  callbacks = [ModelCheckpoint('./saves/ssd8_2_weights_epoch{epoch:02d}_loss{loss:.4f}.h5',
                                                               monitor='val_loss',
                                                               verbose=1,
                                                               save_best_only=True,
                                                               save_weights_only=True,
                                                               mode='auto',
                                                               period=1),
                                               EarlyStopping(monitor='val_loss',
                                                             min_delta=0.001,
                                                             patience=2),
                                               ReduceLROnPlateau(monitor='val_loss',
                                                                 factor=0.5,
                                                                 patience=0,
                                                                 epsilon=0.001,
                                                                 cooldown=0)],
                                  validation_data = val_generator,
                                  validation_steps = ceil(n_val_samples/batch_size),
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False,)
     
     
    model_name = 'ssd8_2'
    model.save('./saves/{}_ssd81.h5'.format(model_name))
    model.save_weights('./saves/{}_weights.h5'.format(model_name))
     
    print()
    print("Model saved as {}.h5".format(model_name))
    print("Weights also saved separately as {}_weights.h5".format(model_name))
    print()

# ### 4. Make predictions
# 
# Now let's make some predictions on the validation dataset with the trained model. We'll use the validation generator which we've already set up above. If you did not use a validation dataset, change "val_dataset" to "train_dataset" below (or whatever you called the `BatchGenerator` instance you used for training above). Feel free to change the batch size.


### Make predictions

# 1: Set the generator

predict_generator = val_dataset.generate(batch_size=2,
                                         train=False,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         diagnostics=False)



# 2: Generate samples
X, y_true, filenames = next(predict_generator)

i = 0 # Which batch item to look at

print("Image:", filenames[i])
#     print()
print("Ground truth boxes:\n")
print("y_true"[i])

# 3: Make a prediction

y_pred = model.predict(X)
y_pred2 = np.argmax(y_pred[i,:,:,0:21],-1)
y_true2 = np.argmax(y_true,-1)
print(y_pred.shape, np.array(y_true).shape)
y_pred_decoded = y_pred2

#from numpy import genfromtxt
#my_data = genfromtxt('./good predictions/y_pred_2008_006213.csv', delimiter=',')
  # X is an array
# np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays

# Finally, let's draw the predicted segment map and original image. NOTE(GP) This currently shows these separately. Would be good to have a way to overlay prediction onto
# the original image
classes = ['0 background', '1 aeroplane', '2 bike', '3 birb', '4 boat', '5 bottle', '6 bus', '7 car', '8 meow','9 chair','10mooo','11 table','12 woof','13 neigh','14 broom','15 meh','16 plant','17 baaaa','18 sofa','19 train','20 tv'] # Just so we can print class names onto the image instead of IDs 

f, axarr = plt.subplots(1,3)
f.suptitle(classes)

# 5: Draw the prediction onto the image NOTE(GP) I've changed the class names when I was sleep deprived. We should probably change them back at some point
print(classes[y_pred2[25,25]]) # This prints the predicted class at the centre of the image
#plt.figure(figsize=(10,10))
axarr[0].imshow(y_pred2)
#plt.figure(figsize=(10,10))
axarr[1].imshow(y_true2[0,:,:])
axarr[2].imshow(X[i])
plt.show()

#np.savetxt(('y_pred_{}.csv').format(filenames[i]), y_pred2, delimiter=',') 
# NOTE(GP) I've kept the code below as it may be reusable later
# current_axis = plt.gca()
# 
#classes = ['background', 'aeroplane', 'bike', 'birb', 'boat', 'bottle', 'bus', 'car', 'meow','chair','mooo','table','woof','neigh','broom','meh','plant','baaaa','sofa','train','tv'] # Just so we can print class names onto the image instead of IDs
# 
# # Draw the predicted boxes in blue
# 
# # Draw the ground truth boxes in green (omit the label for more clarity)
# for box in y_true[i]:
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(plt.Rectangle((box[1], box[3]), box[2]-box[1], box[4]-box[3], color='green', fill=False, linewidth=2))  
#     current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
# 
# for box in y_pred_decoded[i]:
#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#     current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
#     current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

