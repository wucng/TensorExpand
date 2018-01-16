#! /usr/bin/python
# -*- coding: utf8 -*-
'''
tensorlayer作为tf的API，快速搭建网络（相当于tf-slim）
'''
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import glob
import pandas as pd

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
print_freq = 1
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


X_train =np.reshape(data[:,:-1],[-1,img_h,img_w,img_c]).astype(np.float32) # [-1,19,19,4]
y_train =data[:,-1].astype(np.int64) # 非one_hot 标签
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

# ------------------------------------------------------------#
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[batch_size, img_h, img_w, img_c])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[batch_size,])

# Building convolutional network
network = tl.layers.InputLayer(x, name='input') # Tensor-->TL

network = tl.layers.Conv2d(network, 32, (5, 5), (1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn1')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2),
        padding='SAME', name='pool1')

network = tl.layers.Conv2d(network, 64, (5, 5), (1, 1),
        act=tf.nn.relu, padding='SAME', name='cnn2')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2),
        padding='SAME', name='pool2')

## end of conv
network = tl.layers.FlattenLayer(network, name='flatten')
network = tl.layers.DropoutLayer(network, keep=dropout_, name='drop1')
network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=dropout_, name='drop2')
network = tl.layers.DenseLayer(network, n_classes, act=tf.identity, name='output')

y = network.outputs # TL-->Tensor

# ---------------------------------------------------------------#

# cost = tl.cost.cross_entropy(y, y_, 'cost')
cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y,name='cost'))
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost)#, var_list=train_params)

# tl.layers.initialize_global_variables(sess) # 或 tf.global_variables_initializer().run()
tf.global_variables_initializer().run()

for epoch in range(n_epoch):
    # for step in range(1000):
    for step,(X_train_a, y_train_a) in \
            enumerate(tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True)):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update(network.all_drop)  # enable noise layers
        sess.run(train_op, feed_dict=feed_dict)

        if step % display_step==0:
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            print('epoch:',epoch,'|','step:',step,'|','loss:',err,'|','acc',ac)
