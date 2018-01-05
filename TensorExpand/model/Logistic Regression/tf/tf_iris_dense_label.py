#! /usr/bin/python
# -*- coding: utf8 -*-

# iris （非one_hot编码）

from sklearn import datasets
import tensorflow as tf

iris=datasets.load_iris()
tr_x=iris.data
tr_y=iris.target

print('tr_x',tr_x.shape,'tr_y',tr_y.shape) # tr_x (150, 4) tr_y (150,)

# Parameters
learning_rate = 0.1
training_epochs = 1000
batch_size = 20
display_step = 10

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.int64, [None, ])

# Set model weights
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3])) # 3个类别

pred = tf.nn.softmax(tf.matmul(x, W) + b) # shape [N,3]

cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), y)
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
