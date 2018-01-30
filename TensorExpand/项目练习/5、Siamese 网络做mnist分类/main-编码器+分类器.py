#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import os
import sys
import argparse

# 超参数
lr=0.5
batch_size=128
epochs=1
displayer=5
FLAGS=None

parser = argparse.ArgumentParser()
#获得buckets路径
parser.add_argument('--buckets', type=str, default='./MNIST_data',
                    help='input data path')

#获得checkpoint路径
parser.add_argument('--checkpointDir', type=str, default='model',
                    help='output model path')
FLAGS, _ = parser.parse_known_args()

# 建立神经网络编码器
encode_x=tf.placeholder(tf.float32,[None,28*28*1],name='x')
class_x=tf.placeholder(tf.float32,[None,128])
y_=tf.placeholder(tf.int64) # 这里使用非one_hot 标签


# 编码器
def encode(x,reuse):
    x=tf.reshape(x,[-1,28,28,1])

    conv1=tf.layers.conv2d(x,32,3,padding='SAME',activation=tf.nn.relu,reuse=reuse,name='conv1') # [-1,28,28,32]
    pool1=tf.layers.max_pooling2d(conv1,2,2,'SAME',name='pool1') # [-1,14,14,32]

    conv2 = tf.layers.conv2d(pool1, 64, 3, padding='SAME', activation=tf.nn.relu,reuse=reuse, name='conv2')  # [-1,14,14,64]
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME', name='pool2')  # [-1,7,7,64]

    conv3 = tf.layers.conv2d(pool2, 64, 3, padding='SAME', activation=tf.nn.relu,reuse=reuse, name='conv3')  # [-1,7,7,64]
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'SAME', name='pool3')  # [-1,4,4,64]

    fc1=tf.reshape(pool3,[-1,4*4*64],name='flatten') # [-1,4*4*64]
    fc1=tf.layers.dense(fc1,1024,activation=tf.nn.relu,reuse=reuse,name='fc1') # [-1,1024]

    out = tf.layers.dense(fc1, 128, activation=None,reuse=reuse,name='fc2') # [-1,128]

    return out

# 分类器
def classify(x,reuse):
    x=tf.reshape(x,[-1,128])
    fc1=tf.layers.dense(x,64,activation=tf.nn.leaky_relu,reuse=reuse) # [-1,64]
    fc2=tf.layers.dense(fc1,2,activation=tf.nn.sigmoid,reuse=reuse) # [-1,2]
    return fc2

global reuse
reuse=None

encode_=encode(encode_x,reuse) # [-1,128]

classify_=classify(class_x,reuse) # [-1,2]

loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=classify_))

train_op=tf.train.AdamOptimizer(lr).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(classify_, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 加载数据
mnist=read_data_sets(FLAGS.buckets,one_hot=False)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    steps=mnist.train.images.shape[0]//batch_size
    for epoch in range(epochs):
        for step in range(steps):
            if epoch==0 and step ==0:
                reuse=False
            else:
                reuse=True

            batch_x,batch_y=mnist.train.next_batch(batch_size)

            encode_out=sess.run(encode_,feed_dict={encode_x:batch_x}) # [-1,128]

            encode_out_=[np.fabs(encode_out[i]-encode_out[i+1]) for i in range(batch_size-1)]
            encode_out_=np.array(encode_out_,dtype=np.float32)
            label_y=batch_y[0:batch_size-1]-batch_y[1:batch_size]
            label_y_=(label_y==0).astype(np.int64)

            sess.run(train_op,{class_x:encode_out_,y_:label_y_})

            if step%50==0:
                Loss,acc=sess.run([loss,accuracy],{class_x:encode_out_,y_:label_y_})
                print('epoch',epoch,'|','loss',Loss,'|','acc',acc)
