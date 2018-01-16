#! /usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
import tensorflow as tf
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

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int64, [None, ])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def conv_net(x,dropout,n_classes):
    x = tf.reshape(x, shape=[batch_size, img_h, img_w, img_c])
    conv1=tf.layers.conv2d(x,32,5,1,padding='SAME',activation=tf.nn.relu) # [-1,19,19,32]
    conv1=tf.layers.max_pooling2d(conv1,2,2,'SAME') # [-1,10,10,32]

    conv2 = tf.layers.conv2d(conv1, 64, 5, 1, padding='SAME',activation=tf.nn.relu)  # [-1,10,10,64]
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME')  # [-1,5,5,64]

    # conv3 = tf.layers.conv2d(conv2, 64, 5, 2, padding='valid',activation=tf.nn.relu)  # [-1,9,9,64]
    # conv3 = tf.layers.max_pooling2d(conv3, 2, 2, 'SAME')  # [-1,5,5,64]

    fc1=tf.reshape(conv2,[batch_size,-1]) # [batch,5*5*64]
    fc1=tf.layers.dense(fc1,128,activation=tf.nn.relu) # [batch,128]
    # fc1=tf.layers.dropout(fc1,dropout) # layers中的dropout不能使用占位符，会报错
    fc1=tf.nn.dropout(fc1,dropout)

    out=tf.layers.dense(fc1,n_classes) # [batch,20]

    return out

# Construct model
pred = conv_net(x, keep_prob,n_classes)

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

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    pred_y=pred.eval(feed_dict={x: data[:batch_size,0:-1], keep_prob: 1.})
    print('pred:',np.argmax(pred_y,1)[:10],'real:',data[:10,-1])
