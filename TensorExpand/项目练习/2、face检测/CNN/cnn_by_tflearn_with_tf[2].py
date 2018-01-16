#! /usr/bin/python
# -*- coding: utf8 -*-
'''
tflearn作为tf的API，快速搭建网络（相当于tf-slim）
'''


import numpy as np
np.random.seed(1337)  # for reproducibility
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import glob
import pandas as pd
# import tensorflow as tf
# from keras.utils import np_utils
import tensorflow as tf
import tensorlayer as tl

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
display_step = 10
n_epoch=3

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
# y_train =data[:,-1].astype(np.int64)
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

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, img_c])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[None,])

# # Building convolutional network
# network = input_data(shape=[None, img_h, img_w, img_c], name='input')
network = conv_2d(x, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 128, activation='tanh')
network = dropout(network, dropout_)

network = fully_connected(network, 256, activation='tanh')
network = dropout(network, dropout_)

network = fully_connected(network, n_classes, activation='softmax')

y=network
# --------------------------------------------------------------------#
cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y,name='cost'))
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost)#, var_list=train_params)

# tl.layers.initialize_global_variables(sess) # 或 tf.global_variables_initializer().run()
tf.global_variables_initializer().run()

for epoch in range(n_epoch):
    # for step,(X_train_a, y_train_a) in \
    #         enumerate(tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True)):

    np.random.shuffle(data)
    start = 0
    end = 0
    for step in range(1000):
        end = min(len(data), start + batch_size)
        train_data = data[start:end]

        if end == len(data):
            start = 0
        else:
            start = end

        X_train_a = np.reshape(train_data[:, :-1], [-1, img_h, img_w, img_c]).astype(np.float32)  # [-1,19,19,4]
        y_train_a = train_data[:, -1].astype(np.int64)

        feed_dict = {x: X_train_a, y_: y_train_a}
        sess.run(train_op, feed_dict=feed_dict)

        if step % display_step==0:
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            print('epoch:',epoch,'|','step:',step,'|','loss:',err,'|','acc',ac)
