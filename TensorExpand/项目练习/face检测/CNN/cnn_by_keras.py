#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import glob
import pandas as pd

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
display_step = 10
EPOCH=3

# Network Parameters
img_h=19
img_w=19
img_c=4
n_input = img_h*img_w*img_c
n_classes = 2 #
dropout = 0.75 #

# 加载数据
filepaths=glob.glob('./data_*.pkl')
for i,filepath in enumerate(filepaths):
    if i==0:
        data=pd.read_pickle(filepath)
    else:
        data=np.vstack((data,pd.read_pickle(filepath)))


X_train =np.reshape(data[:,:-1],[-1,img_h,img_w,img_c]).astype(np.float32) # [-1,19,19,4]
y_train =data[:,-1].astype(np.int32)

y_train = np_utils.to_categorical(y_train, num_classes=n_classes) # 转成one_hot

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape ( 19, 19,32)
model.add(Convolution2D(
    batch_input_shape=(None, img_h, img_w,img_c), # theano  batch_input_shape=(None, 4, 19, 19),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    # data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (10, 10, 32)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    # data_format='channels_first',
))

# Conv layer 2 output shape (10, 10, 64)
model.add(Convolution2D(64, 5, strides=1, padding='same',)) #data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (5, 5, 64)
model.add(MaxPooling2D(2, 2, 'same',))# data_format='channels_first'))


# Fully connected layer 1 input shape (5 * 5 * 64) = (1600), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(n_classes))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=learning_rate)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_train[:100], y_train[:100])

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
