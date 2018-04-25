#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
inception_resnet_v2  输入shape 299x299x3
inception_resnet_v2.py 下载地址：https://github.com/tensorflow/models/tree/master/research/slim/nets
inception_resnet_v2 model: https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v2
import os
from tensorflow.python.platform import gfile
import argparse
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

with tf.Graph().as_default() as graph:
    img_size=299
    img_channel=3
    num_classes=10
    lr=1e-4

    x=tf.placeholder(tf.float32,[None,img_size,img_size,img_channel],name='x')
    y_ = tf.placeholder(tf.float32, [None, num_classes], 'y_')
    is_training=tf.placeholder(tf.bool, name='MODE')
    keep = tf.placeholder(tf.float32)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        end_points= inception_resnet_v2.inception_resnet_v2(inputs=x,num_classes=1001,is_training=is_training,dropout_keep_prob=keep)

    net=end_points[1]['PreLogitsFlatten']
    net=tf.stop_gradient(net) # 这层与之前的层都不进行梯度更新
    
    print(net.shape)

    fc = slim.fully_connected(net, 128, activation_fn=tf.nn.sigmoid,
                                  scope='coding_layer')

    y = slim.fully_connected(fc, num_classes, activation_fn=tf.nn.softmax,
                                  scope='output')

    cost=tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=y)
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    correct_prediction = tf.equal(y, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用GPU70%的显存

    sess=tf.InteractiveSession(config=config)
    sess.run(init)

    # wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    # tar -zxf inception_resnet_v2_2016_08_30.tar.gz  -C output
    # 加载模型参数
    var_list = tf.global_variables()
    var_list_1 = []
    for var in var_list:  # 不加载 最后两层的参数，即重新训练
        if 'coding_layer' in var.name or 'output' in var.name:
            # var_list_1.remove(var)
            continue
        var_list_1.append(var)
    var_list=None
    saver=tf.train.Saver(var_list=var_list_1)
    saver.restore(sess,'./output/inception_resnet_v2_2016_08_30.ckpt')

    exit(0)
    # for step in range(100):
    #     feed_dict={}
    #     sess.run(train_op,feed_dict=feed_dict)
    sess.close()
