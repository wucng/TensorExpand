#! /usr/bin/python
# -*- coding: utf8 -*-

"""
使用1d卷积
输入[-1,h,w*c]

融合 1维卷积和反向卷积

假设输入x shape[1,19,19,4]

正向卷积：tf.layers.conv_2d(x,32,5,strides=(1,1),'valid') # shape [1,15,15,32]

15=((19+2p-f)/s)+1  其中p为padding大小 valid模式默认padding为0，f为卷积核大小，s为步数
15=((19+2*0-5)/1+1)

反向卷积：tf.layers.conv2d_transpose(x,3,6,strides=(1,1),padding='valid') # shape [1,24,24,3]
是正向卷积反过来计算，即假设 输出的大小为h

19=((h+2p-f)/s)+1 ==> h=(19-1)*s+f-2p=(19-1)*1+6=24

注：如果padding模式为SAME，通过调整padding大小，会保证卷积（无论正向还是反向）后的图像的大小始终不改变
如：tf.layers.conv2d_transpose(x,3,6,strides=(1,1),padding='SAME') # shape [1,19,19,3]

"""

import pandas as pd
import tensorflow as tf
import numpy as np
import glob

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
display_step = 10
n_hidden_units=128
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

def cnn_net(x,dropout,n_classes):
    x = tf.reshape(x, shape=[-1, img_h, img_w*img_c]) # [batch,19,19*4]
    branch1 =tf.layers.conv1d(x,img_w*img_c,3,1,padding='valid',activation=tf.nn.relu) # [batch,17,19*4]
    branch2 = tf.layers.conv1d(x, img_w * img_c, 4, 1, padding='valid', activation=tf.nn.relu) # [batch,16,19*4]
    branch3 = tf.layers.conv1d(x, img_w * img_c, 5, 1, padding='valid', activation=tf.nn.relu) # [batch,15,19*4]

    # conv1 = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=1)
    network=tf.concat([branch1, branch2, branch3],axis=1) # [batch,48,19*4]

    network=tf.layers.max_pooling1d(network,2,2,'SAME') # [batch,24,19*4]

    network=tf.reshape(network,[-1,24,19,4]) # # [batch,24,19,4]
    # '''
    network = tf.layers.conv2d_transpose(
        network, 4,[1,6], padding='valid'
    ) # [batch,24,24,4]
    '''
    network=tf.nn.conv2d_transpose(
        network,
        tf.get_variable("up_sample_1", shape=[1, 6, 4, 4],
                        initializer=tf.random_uniform_initializer()),
        output_shape=[-1, 24, 24,4],
        strides=[1, 1, 1, 1],
        padding="SAME"
    ) # [-1, 24, 24,4]
    # '''
    network=tf.layers.conv2d(network,
                             32,5,padding='SAME',activation=tf.nn.relu) # [-1,24,24,32]
    network=tf.layers.max_pooling2d(network,2,2,'SAME') # [-1,12,12,32]

    network = tf.layers.conv2d(network,
                               64, 5, padding='SAME', activation=tf.nn.relu)  # [-1,12,12,64]
    network = tf.layers.max_pooling2d(network, 2, 2, 'SAME')  # [-1,6,6,64]

    network=tf.reshape(network,[-1,6*6*64])

    network=tf.layers.dense(network,256,activation=tf.nn.relu)

    network = tf.nn.dropout(network, dropout)

    network = tf.layers.dense(network, n_classes, activation=tf.nn.softmax)

    return network

# Construct model
pred = cnn_net(x, keep_prob,n_classes)

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

        feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout}

        # Run optimization op (backprop)
        # sess.run(optimizer, feed_dict=feed_dict)
        try:
            sess.run(optimizer, feed_dict=feed_dict)
        except:
            continue

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
