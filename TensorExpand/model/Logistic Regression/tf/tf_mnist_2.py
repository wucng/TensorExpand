# -*- coding: utf-8 -*-
# mnist使用one_hot 编码与不使用的情况
'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

包含 使用one_hot 编码与不使用的情况

'''

from __future__ import print_function

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-oht", "--one_hot", help="Whether to use one_hot encoding", type=bool, default=False) # 是否采用one_hot编码
args = parser.parse_args()
print("args:",args)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=args.one_hot)

# Parameters
learning_rate = 0.1
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
if args.one_hot: # 使用one_hot编码
    y = tf.placeholder(tf.float32, [None,10])  # 0-9 digits recognition => 10 classes
    # 或
    # y = tf.placeholder(tf.int64, [None, 10])  # 0-9 digits recognition => 10 classes
else:
    y = tf.placeholder(tf.int64, [None,]) # 0-9 digits recognition => 10 classes
# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x, W) + b) # shape [N,10]


if args.one_hot: # 使用one_hot编码
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
    # 或
    # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
else:
    cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    if args.one_hot: # 使用one_hot编码
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    else:
        correct_prediction = tf.equal(tf.argmax(pred, 1), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

# python3 test2.py --one_hot False
