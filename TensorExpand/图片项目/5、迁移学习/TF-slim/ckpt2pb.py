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
    num_classes=1001
    lr=1e-4

    x=tf.placeholder(tf.float32,[None,img_size,img_size,img_channel],name='x')
    y_ = tf.placeholder(tf.float32, [None, num_classes], 'y_')
    is_training=tf.placeholder(tf.bool, name='MODE')
    keep = tf.placeholder(tf.float32)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        y, _ = inception_resnet_v2.inception_resnet_v2(inputs=x,num_classes=num_classes,is_training=is_training,dropout_keep_prob=keep)

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
    saver=tf.train.Saver()
    saver.restore(sess,'./output/inception_resnet_v2_2016_08_30.ckpt')

    graph_def = graph.as_graph_def()

    # with gfile.GFile('./output/export.pb', 'wb') as f:
    #     f.write(graph_def.SerializeToString())

    # 或者
    tf.train.write_graph(graph_def, './output', 'expert-graph.pb', as_text=False)

    exit(0)
    # for step in range(100):
    #     feed_dict={}
    #     sess.run(train_op,feed_dict=feed_dict)
    sess.close()

  
# ---------------------------------------------------------------------

```python
# 保存图表并保存变量参数

from tensorflow.python.framework import graph_util
var_list=tf.global_variables()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=[var_list[i].name for i in range(len(var_list))]) # 保存图表并保存变量参数
tf.train.write_graph(constant_graph, './output', 'expert-graph.pb', as_text=False)

# -----方式2-------------------
from tensorflow.python.framework import graph_util
var_list=tf.global_variables()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=[var_list[i].name for i in range(len(var_list))])
with tf.gfile.FastGFile(logdir+'expert-graph.pb', mode='wb') as f:
    f.write(constant_graph.SerializeToString())
```

```
# 只保留图表
graph_def = tf.get_default_graph().as_graph_def()
with gfile.GFile('./output/export.pb', 'wb') as f:
    f.write(graph_def.SerializeToString())

# 或者
tf.train.write_graph(graph_def, './output', 'expert-graph.pb', as_text=False)
```
    
    
