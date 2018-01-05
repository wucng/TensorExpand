
```python
# -*- coding: utf-8 -*-

import tensorlayer as tl
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys,keep_prob:1})
    return result

with tf.device('/gpu:0'):
    xs=tf.placeholder(tf.float32,[None,784])
    ys=tf.placeholder(tf.float32,[None,10])
    keep_prob = tf.placeholder(tf.float32)

net=tf.reshape(xs,[-1,28,28,1]) # Tensor

# Using tensorlayer convolution layer.
net = tl.layers.InputLayer(net, name='input_layer') # Tensor 转成 TL
net=tl.layers.Conv2dLayer(net,act=tf.nn.relu,shape=[3,3,1,32])

# Using Tensorflow's max pooling op.
net = tf.nn.max_pool(net.outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 类型 Tensor
# net=tl.layers.MaxPool2d(net)

net = tl.layers.InputLayer(net, name='input_layer2') # Tensor 转成 TL
net=tf.nn.dropout(net.outputs,keep_prob) # net.outputs TL 转成Tensor
net = tl.layers.InputLayer(net, name='input_layer3') # Tensor 转成 TL

# 全连接层
net=tl.layers.FlattenLayer(net) # 数据形状转成[batch_size,-1]
pred=tl.layers.DenseLayer(net,10,act=tf.nn.softmax)

prediction=pred.outputs # 数据类型 TL 转成Tensor

# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=prediction))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
```
