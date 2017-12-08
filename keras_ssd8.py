''' NOTE(GP) Original description of 7 layer SSD model. 
A small 7-layer Keras model with SSD architecture. Also serves as a template to build arbitrary network architectures.

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

''' NOTE(GP) 
The new model is built on the backbone of the SSD7 version. The number, order and size of layers has been changed, but largely it is the same 
CNN architecture and code as standard, until the upsampling layers. Most references to SSD encoders or anchor boxes have been removed, but any left over
should be ignored.
'''

import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, Conv2DTranspose, AtrousConv2D, AtrousConvolution2D

from keras.activations import softmax
from keras.layers.convolutional import UpSampling2D

def build_model(image_size,
                n_classes):
    ''' NOTE(GP) original description. Unused argument descriptions have been removed
    Build a Keras model with SSD architecture, see references.

    The model consists of convolutional feature layers and a number of convolutional
    predictor layers that take their input from different feature layers.
    The model is fully convolutional.

    The implementation found here is a smaller version of the original architecture
    used in the paper (where the base network consists of a modified VGG-16 extended
    by a few convolutional feature layers), but of course it could easily be changed to
    an arbitrarily large SSD architecture by following the general design pattern used here.
    This implementation has 7 convolutional layers and 4 convolutional predictor
    layers that take their input from layers 4, 5, 6, and 7, respectively.

    In case you're wondering why this function has so many arguments: All arguments except
    the first two (`image_size` and `n_classes`) are only needed so that the anchor box
    layers can produce the correct anchor boxes. In case you're training the network, the
    parameters passed here must be the same as the ones used to set up `SSDBoxEncoder`.
    In case you're loading trained weights, the parameters passed here must be the same
    as the ones used to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Training currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of categories for classification including
            the background class (i.e. the number of positive classes +1 for
            the background calss).

    Returns:
        model: The Keras SSD model.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    # Design the actual network
    x = Input(shape=(img_height, img_width, img_channels))
    normed = Lambda(lambda z: z/127.5 - 1., # Convert input feature range to [-1,1]
                    output_shape=(img_height, img_width, img_channels),
                    name='lambda1')(x)

    conv1 = Conv2D(16, (5, 5), name='conv1', strides=(1, 1), padding="same")(normed)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
#    200 * 200
    conv2 = Conv2D(32, (3, 3), name='conv2', strides=(1, 1), padding="same")(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
#    100 * 100
    conv3 = Conv2D(64, (3, 3), name='conv3', strides=(1, 1), padding="same")(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
#     LAYER REMOVED FOR TIME BEING
#     conv3b = Conv2D(32, (3, 3), name='conv3b', strides=(1, 1), padding="same")(conv3)
#     conv3b = BatchNormalization(axis=3, momentum=0.99, name='bn3b')(conv3b)
#     conv3b = ELU(name='elu3b')(conv3b)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
#     LAYER REMOVED FOR TIME BEING
#     conv4 = Conv2D(64, (3, 3), name='conv4', strides=(1, 1), padding="same")(pool3)
#     conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
#     conv4 = ELU(name='elu4')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)
#    50 * 50
    conv5 = Conv2D(128, (3, 3), name='conv5', strides=(1, 1), padding="same")(pool3)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)
# 25 * 25
    conv6 = Conv2D(128, (3, 3), name='conv6', strides=(1, 1), padding="valid")(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)
#    12*12
    conv6b = Conv2D(128, (3, 3), name='conv6b', strides=(1, 1), padding="same")(pool6)
    conv6b = BatchNormalization(axis=3, momentum=0.99, name='bn6b')(conv6b)
    conv6b = ELU(name='elu6b')(conv6b)
    pool6b = MaxPooling2D(pool_size=(2, 2), name='pool6b')(conv6b)
#    6*6
#    NOTE (GP) These layers upsample the image back to 51*51
    deconv1 = Conv2DTranspose(128, (3,3),name='deconv1', strides=(1, 1), padding='same')(pool6b)
    deconv1 = BatchNormalization(axis=3, momentum=0.99, name='bndc1')(deconv1)
    deconv1 = ELU(name='eludc1')(deconv1)    
    uppool1 = UpSampling2D(3)(deconv1) 
   
    deconv2 = Conv2DTranspose(64, (3,3),name='deconv2', strides=(1, 1), padding='valid')(uppool1)
    deconv2 = BatchNormalization(axis=3, momentum=0.99, name='bndc2')(deconv2)
    deconv2 = ELU(name='eludc2')(deconv2)  
    uppool2 = UpSampling2D(3)(deconv2) 
#     51 * 51

#    NOTE(GP) This final layer produces the softmax predictions of each class in a 51 * 51 area
    out = Conv2DTranspose(21, (3,3),name='out', strides=(1, 1), activation='softmax', padding='same')(uppool2)
    
    predictions = out

    model = Model(inputs=x, outputs=predictions)

    return model
