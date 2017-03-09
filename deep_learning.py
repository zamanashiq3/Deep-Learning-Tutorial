from __future__ import print_function,division,absolute_import

"""
project simion deep learning model applied in the simion dataset.

Akm Ashiquzzaman
13101002@uap-bd.edu
fall 2016


A simple convnet for digit classifing. 

See the semeion handwritten digit database for details  

"""

import numpy as np
np.random.seed(1337) #for reproducibility

#loading data
data = np.load('../semeion.npz')
#X and Y for train and feet
dataX, dataY = data['arr_0'],data['arr_1']

dataX = dataX.reshape(dataX.shape[0],1,16,16).astype('float32')
dataY = dataY.astype('float32')
 
print('X tensor sample size: ',dataX.shape)
print('Y tensor sample size: ',dataY.shape)

#keras import
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Convolution2D(32,3,3,
                        border_mode='valid',
                        input_shape=(1,16,16)))
model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Convolution2D(16,3,3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

#ConvNET weight flepath saving, we're usung the callback loss to save the best weight model.

from os.path import isfile
  
weight_path="../semeion_weights.hdf5"

if isfile(weight_path):
	model.load_weights(weight_path)

model.compile(optimizer='adadelta',loss='categorical_crossentropy'
	      ,metrics=['accuracy'])

checkpoint = ModelCheckpoint(weight_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#Data/Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())

datagen.fit(dataX)

model.fit_generator(datagen.flow(dataX,dataY,batch_size=59),samples_per_epoch=len(dataX),nb_epoch=30,callbacks=callbacks_list)




