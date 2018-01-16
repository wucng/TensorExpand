#! /usr/bin/python
# -*- coding: utf8 -*-

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
n_epoch=3
hidden_size=128
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

x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, img_c])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[None,])

# Building convolutional network
network = tl.layers.InputLayer(x, name='input') # Tensor-->TL

network = tl.layers.Conv2d(network, 32, (5, 5), (1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn1') # [-1,19,19,32]
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2),
        padding='SAME', name='pool1') # [-1,10,10,32]

network=network.outputs # TL-->Tensor
network=tf.reshape(network,[-1,10,10*32]) # [-1,10,10*32] Tensor
network=tl.layers.InputLayer(network, name='input2') # Tensor-->TL
'''
network=tl.layers.RNNLayer(
    network,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,  #tf.nn.rnn_cell.BasicLSTMCell,)
    cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
    n_hidden=hidden_size,
    initializer=tf.random_uniform_initializer,
    n_steps=1,
    return_last=False,
    return_seq_2d=True,
    name='basic_lstm2') # 10个[batch,128]序列  ；如果n_steps=10 shape [10*batch,128]
'''
'''
network=tl.layers.DynamicRNNLayer(
    network,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
    n_hidden=hidden_size,n_layer=2,return_last=True,return_seq_2d=True,
    name='basic_lstm2'
)
'''
network=tl.layers.BiRNNLayer(
    network,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
    n_hidden=hidden_size,
    n_steps=1,
    n_layer=2,
    return_last=True,
    return_seq_2d=True,
    name='basic_lstm2'
)

## end of conv
network = tl.layers.DropoutLayer(network, keep=dropout_, name='drop1')
network = tl.layers.DenseLayer(network, n_classes, act=tf.identity, name='output')

y = network.outputs # TL-->Tensor

cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess) # 或 tf.global_variables_initializer().run()

for epoch in range(n_epoch):
    for X_train_a, y_train_a in tl.iterate.minibatches(
            X_train, y_train, batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update(network.all_drop)  # enable noise layers
        sess.run(train_op, feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            train_loss += err;
            train_acc += ac;
            n_batch += 1
        print(" train loss: %f" % (train_loss / n_batch),'train acc:',train_acc/n_batch)
