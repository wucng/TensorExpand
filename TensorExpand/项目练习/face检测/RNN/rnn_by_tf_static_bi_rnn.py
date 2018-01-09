#! /usr/bin/python
# -*- coding: utf8 -*-

'''
static_rnn
'''

import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import glob

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
display_step = 10

# Network Parameters
img_h=19
img_w=19
img_c=4
n_input = img_h*img_w*img_c
n_classes = 2 #
dropout = 0.75 #
n_hidden_units=128

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int64, [None,])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([img_w * img_c, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units*2, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def rnn_net(x,weights,biases):
    x = tf.reshape(x, shape=[-1, img_h, img_w*img_c])  # [batch,h,w*c] 相当于 [batch,序列数,每个序列长度]
    x = tf.reshape(x, shape=[-1, img_w * img_c])  # [batch*h,w*c]

    # into hidden
    X_in = tf.matmul(x, weights['in']) + biases['in'] # [batch*h,n_hidden_units]
    X_in = tf.reshape(X_in, [-1, img_h, n_hidden_units]) # [batch,h,n_hidden_units]

    X_in = tf.unstack(X_in, img_h, 1) # img_h个[batch,n_hidden_units]序列

    # basic LSTM Cell.
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units*2)
    # 加入多层rnn核
    # lstm_cell = rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
    # lstm_cell = rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
    # lstm_cell = rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)

    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)

    # init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    # outputs, final_state = tf.nn.static_rnn(lstm_cell, X_in, dtype=tf.float32)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X_in,
                                                     dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X_in,
                                               dtype=tf.float32)

    # # outputs shape h个[batch,n_hidden_units*2]序列
    results = tf.matmul(outputs[-1], weights['out']) + biases['out'] # outputs[-1]取最后一个序列 shape [batch,n_hidden_units*2]

    return results

# Construct model
pred = rnn_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# 加载数据
filepaths=glob.glob('./data_*.pkl')
for i,filepath in enumerate(filepaths):
    if i==0:
        data=pd.read_pickle(filepath)
    else:
        data=np.vstack((data,pd.read_pickle(filepath)))
np.random.shuffle(data)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    start=0
    end=0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # print('start:',step)
        end = min(len(data), start + batch_size)
        train_data=data[start:end]
        batch_x, batch_y = train_data[:,0:-1],train_data[:,-1]
        if end == len(data):
            start = 0
        else:
            start = end

        # if step==1:
        feed_dict={x: batch_x, y: batch_y,keep_prob: dropout}
        # else:
        #     feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout,final_state:state}

        # Run optimization op (backprop)
        # _,state=sess.run([optimizer,final_state], feed_dict=feed_dict)
        sess.run(optimizer, feed_dict=feed_dict)
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    pred_y=pred.eval(feed_dict={x: data[:batch_size,0:-1], keep_prob: 1.})
    print('pred:',np.argmax(pred_y,1)[:10],'real:',data[:10,-1])
