#! /usr/bin/python
# -*- coding: utf8 -*-

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

mnist = read_data_sets('./MNIST_data', one_hot=True)

# train_x, train_y, test_x, test_y = tflearn.datasets.mnist.load_data(one_hot=True)

train_x=mnist.train.images
train_y=mnist.train.labels
test_x=mnist.test.images
test_y=mnist.test.labels

train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)

# 定义神经网络模型
conv_net = input_data(shape=[None, 28, 28, 1], name='input')
conv_net = conv_2d(conv_net, 32, 2, activation='relu')
conv_net = max_pool_2d(conv_net, 2)
conv_net = conv_2d(conv_net, 64, 2, activation='relu')
conv_net = max_pool_2d(conv_net, 2)
conv_net = fully_connected(conv_net, 1024, activation='relu')
conv_net = dropout(conv_net, 0.8)
conv_net = fully_connected(conv_net, 10, activation='softmax')
conv_net = regression(conv_net, optimizer='adam', loss='categorical_crossentropy', name='output')

model = tflearn.DNN(conv_net)

# 训练
model.fit({'input': train_x}, {'output': train_y}, n_epoch=13,
          validation_set=({'input': test_x}, {'output': test_y}),
          snapshot_step=300, show_metric=True, run_id='mnist')

model.save('mnist.model')  # 保存模型

"""
model.load('mnist.model')   # 加载模型
model.predict([test_x[1]])  # 预测
"""