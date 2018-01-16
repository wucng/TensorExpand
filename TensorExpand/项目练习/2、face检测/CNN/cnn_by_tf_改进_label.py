#! /usr/bin/python
# -*- coding: utf8 -*-

'''
改进地方：将输入x与第一层卷积合并一起作为下一层的输入

img:[h,w,c]
label:[h,w]

'''
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
x = tf.placeholder(tf.float32, [None, img_h,img_w,img_c])
y = tf.placeholder(tf.float32, [None, img_h,img_w])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def conv_net(x,dropout,n_classes):
    # x = tf.reshape(x, shape=[batch_size, img_h, img_w, img_c]) # [-1,19,19,4]
    conv1=tf.layers.conv2d(x,32,5,1,padding='SAME',activation=tf.nn.relu) # [-1,19,19,32]

    conv1=tf.concat([x,conv1],axis=-1) # [-1,19,19,36] 改进地方：将输入x与第一层卷积合并一起作为下一层的输入

    conv1=tf.layers.max_pooling2d(conv1,2,2,'SAME') # [-1,10,10,36]

    conv2 = tf.layers.conv2d(conv1, 64, 5, 1, padding='SAME',activation=tf.nn.relu)  # [-1,10,10,64]
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME')  # [-1,5,5,64]

    # conv3 = tf.layers.conv2d(conv2, 64, 5, 2, padding='valid',activation=tf.nn.relu)  # [-1,9,9,64]
    # conv3 = tf.layers.max_pooling2d(conv3, 2, 2, 'SAME')  # [-1,5,5,64]

    fc1=tf.reshape(conv2,[-1,5*5*64]) # [batch,5*5*64]
    fc1=tf.layers.dense(fc1,256,activation=tf.nn.relu) # [batch,128]
    # fc1=tf.layers.dropout(fc1,dropout) # layers中的dropout不能使用占位符，会报错
    fc1=tf.nn.dropout(fc1,dropout)

    out=tf.layers.dense(fc1,img_h*img_w) # [batch,19*19]

    out=tf.reshape(out,[-1,img_h,img_w]) # [batch,19,19]

    return out

# Construct model
pred = conv_net(x, keep_prob,n_classes)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
'''
# Calculate accuracy
def compute_acc(xs, ys, img_h,img_w):
    global pred
    y1 = sess.run(pred, {x: xs, y_: ys, keep: 1., is_training: False})
    prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in y1 for x2 in x1 for x3 in x2]
    prediction = np.reshape(prediction, [-1, img_h,img_w]).astype(np.uint8)
    accuracy = np.mean(np.equal(prediction, ys).astype(np.float32))
    return accuracy
'''
# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), y)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
        # batch_x, batch_y = train_data[:,0:-1],train_data[:,-1]
        batch_x = train_data[:, 0:-1]
        batch_x=np.reshape(batch_x,[-1,img_h,img_w,img_c])
        batch_y=np.mean(batch_x,axis=-1).astype(np.uint8) # 按波段求平均 转成int 做标签
        if end == len(data):
            start = 0
        else:
            start = end

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x,
                                               y: batch_y,
                                               keep_prob: 1.})

            y1 = sess.run(pred, {x: batch_x, y: batch_y, keep_prob: 1.})
            prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in y1 for x2 in x1 for x3 in x2]
            prediction = np.reshape(prediction, [-1, img_h, img_w]).astype(np.uint8)
            acc = np.mean(np.equal(prediction, batch_y).astype(np.float32))

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    test_x = data[:, 0:-1]
    test_x = np.reshape(test_x, [-1, img_h, img_w, img_c])
    test_y = np.mean(test_x, axis=-1).astype(np.uint8)  # 按波段求平均 转成int 做标签

    pred_y=pred.eval(feed_dict={x: test_x, keep_prob: 1.})
    prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in pred_y for x2 in x1 for x3 in x2]
    prediction = np.reshape(prediction, [-1, img_h, img_w]).astype(np.uint8)
    print('pred:',np.argmax(prediction,1)[:10],'real:',test_y[:10])
