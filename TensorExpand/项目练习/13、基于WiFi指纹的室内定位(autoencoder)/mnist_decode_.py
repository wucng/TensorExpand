#! /usr/bin/python
# -*- coding: utf8 -*-

'''
参考模型流程图 使用卷积网络 对mnist进行分类

分开训练非监督与监督模型

包含非监督方法（自编码）
监督分类
'''

import tensorflow as tf
import numpy as np

# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

n_output_layer = 10


# 定义待训练的神经网络
def convolutional_neural_network(X):
    X=tf.reshape(X,[-1,28,28,1])

    # encode
    conv1=tf.layers.conv2d(X,32,3,padding='SAME',activation=tf.nn.relu) # [-1,28,28,32]
    pool1=tf.layers.max_pooling2d(conv1,2,2,'SAME') # [-1,14,14,32]

    conv2=tf.layers.conv2d(pool1,64,3,padding='SAME',activation=tf.nn.relu) # [-1,14,14,64]
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME') # [-1,7,7,64]

    # decode
    conv3=tf.layers.conv2d_transpose(pool2,32,8,1,padding='valid',activation=tf.nn.relu) # [-1,14,14,32]
    decode=tf.layers.conv2d_transpose(conv3,1,15,1,padding='valid',activation=tf.nn.relu) # [-1,28,28,1]


    # 监督分类
    fc = tf.reshape(pool2, [-1, 7 * 7 * 64])  # [-1,7*7*64]
    fc=tf.layers.dense(fc,1024,activation=tf.nn.relu)

    output=tf.layers.dense(fc,n_output_layer,activation=None)

    return decode,output


# 每次使用100条数据进行训练
batch_size = 128
epochs = 1

X = tf.placeholder('float', [None, 28 * 28*1])
Y = tf.placeholder('float')


# 使用数据训练神经网络
def train_neural_network(X, Y):
    decode,predict = convolutional_neural_network(X)
    # 非监督模型
    un_cost = tf.losses.mean_squared_error(labels=tf.reshape(X, [-1, 28, 28, 1]), predictions=decode)
    un_optimizer = tf.train.AdamOptimizer().minimize(un_cost)
    # 监督模型
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)  # learning rate 默认 0.001

    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    total_batches = mnist.train.images.shape[0] // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 训练非监督模型
        for epoch in range(epochs):
            epoch_costs = np.empty(0) # 空数组
            for step in range(total_batches):
                x, _ = mnist.train.next_batch(batch_size)
                _, c = sess.run([un_optimizer, un_cost], feed_dict={X: x})

                epoch_costs = np.append(epoch_costs, c)

            print("Epoch: ", epoch,'|', " Loss: ", np.mean(epoch_costs))

        print("------------------------------------------------------------------")


        # 训练监督模型
        for epoch in range(epochs):
            epoch_costs = np.empty(0)
            for step in range(total_batches):
                x, y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: x,Y:y})

                epoch_costs = np.append(epoch_costs, c)

            accuracy_in_train_set = sess.run(accuracy, feed_dict={X: mnist.train.images[:500], Y: mnist.train.labels[:500]})
            accuracy_in_test_set = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            print("Epoch: ", epoch,'|', " Loss: ", np.mean(epoch_costs),'|', " Accuracy: ", accuracy_in_train_set, '||',
                  accuracy_in_test_set)

train_neural_network(X, Y)

"""
Accuracy:  0.998   0.9919
"""
