#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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
dropout_ = 0.75 #

# 加载数据
filepaths=glob.glob('./data_*.pkl')
for i,filepath in enumerate(filepaths):
    if i==0:
        data=pd.read_pickle(filepath)
    else:
        data=np.vstack((data,pd.read_pickle(filepath)))


# X_train =np.reshape(data[:,:-1],[-1,img_h,img_w,img_c]).astype(np.float32) # [-1,19,19,4]
X_train=data[:,:-1].astype(np.float32) # [-1,19*19*4]
y_train =data[:,-1].astype(np.int32)

y_train = np_utils.to_categorical(y_train, num_classes=n_classes) # 转成one_hot

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (-1,256)
model.add(Dense(
    batch_input_shape=(None, img_h*img_w*img_c), # [-1,19*19*4]
    units=256,
    #activation='relu'
))  # [-1,256]
model.add(Activation('relu'))

model.add(Dropout(dropout_))

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
