#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
np.random.seed(1337)  # for reproducibility
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import glob
import pandas as pd
import tensorflow as tf
# from keras.utils import np_utils


# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
display_step = 10
EPOCH=3
hiddle_layes_1=256
hiddle_layes=128
hiddle_layes_2=256
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
X_train =np.reshape(data[:,:-1],[-1,img_h,img_w*img_c]).astype(np.float32) # [-1,19,19*4]
y_train =data[:,-1].astype(np.int64)
# y_train=np_utils.to_categorical(y_train,n_classes) # 转成one_hot


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # 从标量类标签转换为一个one-hot向量
    num_labels = labels_dense.shape[0]        #label的行数
    index_offset = np.arange(num_labels) * num_classes
    # print index_offset
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# y_train=dense_to_one_hot(y_train,n_classes) # 转成one_hot
y_train=to_categorical(y_train,n_classes)


# # Building convolutional network
network = tflearn.input_data(shape=[None, img_h,img_w*img_c], name='input') # [-1,19,19*4]
# network=tflearn.fully_connected(network,hiddle_layes_1) # [-1,256] 默认flatten（）
network=tf.layers.dense(network,hiddle_layes_1,activation=tf.nn.relu) # [-1,19,256]

# network=tflearn.embedding(network, input_dim=img_h*img_w*img_c, output_dim=hiddle_layes)  # [-1,256,128]

network = tflearn.lstm(network, hiddle_layes_2, dropout=dropout_) # 256个[-1,256]序列

network = tflearn.fully_connected(network, 512, activation='tanh') # [-1,512] 默认flatten（）
network = tflearn.dropout(network, dropout_)

network = tflearn.fully_connected(network, n_classes, activation='softmax') # [-1,2]

network = tflearn.regression(network, optimizer='adam', learning_rate=learning_rate,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X_train}, {'target': y_train}, n_epoch=2,
           validation_set=({'input': X_train[:200]}, {'target': y_train[:200]}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
