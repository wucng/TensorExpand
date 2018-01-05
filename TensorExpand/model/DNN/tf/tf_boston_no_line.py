#! /usr/bin/python
# -*- coding: utf8 -*-

# boston 回归

from sklearn import datasets
import tensorflow as tf
import numpy as np
# import tflearn

boston=datasets.load_boston()
tr_x=boston.data
tr_y=boston.target
tr_y=np.reshape(tr_y,[-1,1])
print('tr_x',tr_x.shape,'tr_y',tr_y.shape) # tr_x (506, 13) tr_y (506,)

# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 20
display_step = 10

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 13])
y = tf.placeholder(tf.float32, [None,1])

# Set model weights
W = tf.Variable(tf.zeros([13, 1]))
b = tf.Variable(tf.zeros([1])) # 回归问题

pred = tf.nn.relu(tf.matmul(x, W) + b) # shape [N,1]

# cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
cost = tf.reduce_mean(tf.reduce_sum(tf.square(pred-y),reduction_indices=1))
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # total_batch = int(150/batch_size)
        # Loop over all batches
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: tr_x,
                                                      y: tr_y})
        # Compute average loss
        avg_cost += c
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost),
                  "Accuracy:", accuracy.eval({x: tr_x, y: tr_y}))

    print("Optimization Finished!")

    # Test model
