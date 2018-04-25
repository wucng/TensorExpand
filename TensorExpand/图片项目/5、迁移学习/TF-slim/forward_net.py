#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
inception_resnet_v2  输入shape 299x299x3
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v2
import os
import argparse
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

img_size=299
img_channel=3
num_classes=10
lr=1e-4

x=tf.placeholder(tf.float32,[None,img_size,img_size,img_channel],name='x')
y_ = tf.placeholder(tf.float32, [None, num_classes], 'y_')
is_training=tf.placeholder(tf.bool, name='MODE')
keep = tf.placeholder(tf.float32)

with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    y, _ = inception_resnet_v2.inception_resnet_v2(inputs=x,num_classes=num_classes,is_training=is_training,dropout_keep_prob=keep)

cost=tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=y)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Launch the graph
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用GPU70%的显存

sess=tf.InteractiveSession(config=config)
sess.run(init)

for step in range(100):
    feed_dict={}
    sess.run(train_op,feed_dict=feed_dict)

sess.close()
