#! /usr/bin/python
# -*- coding: utf8 -*-

'''
参考模型流程图 使用卷积网络 对mnist进行分类

并同时训练非监督+监督模型

包含非监督方法（自编码）
监督分类
'''

import tensorflow as tf
import os
import gzip
import numpy as np
import argparse

'''
Loading data with Python
def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_mnist('./', kind='train')
X_test, y_test = load_mnist('./', kind='t10k')
'''

from tensorflow.examples.tutorials.mnist import input_data

FLAGS=None

# mnist.train.next_batch(BATCH_SIZE)
# print(mnist.train.images.shape) # (55000, 784)
# print(mnist.train.labels.shape) # (55000,10)
# print(mnist.test.images.shape) # (10000, 784)
# print(mnist.test.labels.shape) # (10000,10)


# 定义待训练的神经网络
def convolutional_neural_network(X,keep_prob,training):
    X=tf.reshape(X,[-1,28,28,1])
    X=tf.layers.batch_normalization(X,training=training)
    # encode
    conv1=tf.layers.conv2d(X,32,3,padding='SAME',activation=tf.nn.relu) # [-1,28,28,32]
    conv1 = tf.layers.batch_normalization(conv1, training=training)
    pool1=tf.layers.max_pooling2d(conv1,2,2,'SAME') # [-1,14,14,32]
    pool1 = tf.layers.batch_normalization(pool1, training=training)

    conv2=tf.layers.conv2d(pool1,64,3,padding='SAME',activation=tf.nn.relu) # [-1,14,14,64]
    conv2 = tf.layers.batch_normalization(conv2, training=training)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME') # [-1,7,7,64]
    pool2 = tf.layers.batch_normalization(pool2, training=training)

    # decode
    conv3=tf.layers.conv2d_transpose(pool2,32,8,1,padding='valid',activation=tf.nn.relu) # [-1,14,14,32]
    conv3 = tf.layers.batch_normalization(conv3, training=training)
    decode=tf.layers.conv2d_transpose(conv3,1,15,1,padding='valid',activation=tf.nn.relu) # [-1,28,28,1]
    decode = tf.layers.batch_normalization(decode, training=training)


    # 监督分类
    fc = tf.reshape(pool2, [-1, 7 * 7 * 64])  # [-1,7*7*64]
    fc=tf.layers.dense(fc,1024,activation=tf.nn.relu)

    fc=tf.nn.dropout(fc,keep_prob=keep_prob)

    output=tf.layers.dense(fc,FLAGS.n_class,activation=None)

    return decode,output


# 使用数据训练神经网络
def train_neural_network(X, Y,keep_prob,training,mnist):
    decode,predict = convolutional_neural_network(X,keep_prob,training)
    # 非监督模型
    un_cost = tf.losses.mean_squared_error(labels=tf.reshape(X, [-1, 28, 28, 1]), predictions=decode)
    un_optimizer = tf.train.AdamOptimizer().minimize(un_cost)
    # 监督模型
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)  # learning rate 默认 0.001

    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    total_batches = mnist.train.images.shape[0] // FLAGS.batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 同时训练非监督+监督模型
        for epoch in range(FLAGS.epochs):
            un_epoch_costs = np.empty(0) # 空数组
            epoch_costs = np.empty(0)
            for step in range(total_batches):
                x, y = mnist.train.next_batch(FLAGS.batch_size)
                _, un_c = sess.run([un_optimizer, un_cost], feed_dict={X: x,training:FLAGS.training})
                un_epoch_costs = np.append(un_epoch_costs, un_c)

                _, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y,keep_prob:FLAGS.keep_hold,training:FLAGS.training})
                epoch_costs = np.append(epoch_costs, c)

            print("Epoch: ", epoch,'|', " un_Loss: ", np.mean(un_epoch_costs))

            accuracy_in_train_set = sess.run(accuracy,
                                             feed_dict={X: mnist.train.images[:500], Y: mnist.train.labels[:500],keep_prob:1.,training:True})
            accuracy_in_test_set = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob:1.,training:True})
            print("Epoch: ", epoch, '|', " Loss: ", np.mean(epoch_costs), '|', " Accuracy: ", accuracy_in_train_set,
                  '||',accuracy_in_test_set)

        print("------------------------------------------------------------------")

        # accuracy_in_test_set = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.,
        #                                                      training: False})
        # print("Accuracy: ", accuracy_in_test_set)

def main(_):
    # n_output_layer = FLAGS.n_class
    # batch_size = FLAGS.batch_size
    # epochs = FLAGS.epochs
    mnist = input_data.read_data_sets(FLAGS.buckets, one_hot=True)
    X = tf.placeholder('float', [None, 28 * 28 * 1])
    Y = tf.placeholder('float')
    keep_prob=tf.placeholder('float')
    training=tf.placeholder('bool')

    train_neural_network(X, Y,keep_prob,training,mnist)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # 获得buckets路径
    parser.add_argument('--buckets', type=str, default=r'C:\Users\Administrator\Desktop\test\fashion_mnist',
                        help='input data path')
    # 获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='model',
                        help='output model path')

    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')

    parser.add_argument('--keep_hold', type=float, default=0.8,
                        help='droup out')

    parser.add_argument('--training', type=bool, default=True,
                        help='training')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')

    parser.add_argument('--n_class', type=int, default=10,
                        help='num class')

    FLAGS=parser.parse_args()
    # FLAGS, _ = parser.parse_known_args()

    tf.app.run(main=main)