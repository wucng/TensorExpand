#! /usr/bin/python
# -*- coding: utf8 -*-

'''
Python：print显示颜色  https://www.cnblogs.com/ping-y/p/5897018.html 
'''

import tensorflow as tf
import numpy as np
import prettytensor as pt


# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

n_output_layer = 10


# 定义待训练的神经网络
def convolutional_neural_network(data):
    weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'w_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_output_layer]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_output_layer]))}

    data = tf.reshape(data, [-1, 28, 28, 1])

    conv1 = tf.nn.relu(
        tf.add(tf.nn.conv2d(data, weights['w_conv1'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv1'])) # [-1,28,28,32]
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [-1,14,14,32]

    conv2 = tf.nn.relu(
        tf.add(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv2'])) # [-1,14,14,64]
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [-1,7,7,64]

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64]) # [-1,7*7*64]
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc'])) # [-1,1024]

    # dropout剔除一些"神经元"
    # fc = tf.nn.dropout(fc, 0.8)

    output = tf.add(tf.matmul(fc, weights['out']), biases['out']) # [-1,10]
    return output


# 每次使用100条数据进行训练
batch_size = 100

X = tf.placeholder('float', [None, 28 * 28*1])
Y = tf.placeholder('float')


# 使用数据训练神经网络
def train_neural_network(X, Y):
    '''
    predict = convolutional_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    '''

    # 使用prettytensor
    input_tensor = tf.reshape(X,[-1, 28, 28, 1])
    pretty_input = pt.wrap(input_tensor)
    predict, cost_func = (pretty_input. \
                          conv2d(kernel=5, depth=32, name='layer_conv1'). \
                          max_pool(kernel=2, stride=2). \
                          conv2d(kernel=5, depth=64, name='layer_conv2'). \
                          max_pool(kernel=2, stride=2). \
                          flatten(). \
                          fully_connected(size=1024, name='layer_fc1'). \
                          softmax_classifier(num_classes=n_output_layer, labels=Y)
                          )
    accuracy = predict.evaluate_classifier(Y)
    optimizer_ = tf.train.GradientDescentOptimizer(0.1)  # learning rate
    optimizer = pt.apply_optimizer(optimizer_, losses=[cost_func])


    epochs = 1
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for step in range(mnist.train.num_examples // batch_size):
                x, y = mnist.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                # epoch_loss += c
                # print(epoch, ' : ', epoch_loss)
                if step %20==0:
                    print('epoch',epoch,'\t|','step',step,'\t|','\033[1;35m train acc \033[0m',accuracy.eval({X:x,Y:y}),'\t|'
                          '\033[1;35m test acc \033[0m','\033[1;34m '+ str(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))+'\033[0m')

        print('准确率: ', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

train_neural_network(X, Y)
