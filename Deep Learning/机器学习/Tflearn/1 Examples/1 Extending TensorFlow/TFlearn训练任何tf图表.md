
# 1、TFlearn训练TF搭建的图表
```python
"""
This tutorial will introduce how to combine TFLearn and Tensorflow, using
TFLearn wrappers regular Tensorflow expressions.
"""

import tensorflow as tf
import tflearn

# ----------------------------
# Utils: Using TFLearn Trainer
# ----------------------------

# Loading MNIST complete dataset
import tflearn.datasets.mnist as mnist
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

# Define a dnn using Tensorflow
with tf.Graph().as_default():

    # Model variables
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    W1 = tf.Variable(tf.random_normal([784, 256]))
    W2 = tf.Variable(tf.random_normal([256, 256]))
    W3 = tf.Variable(tf.random_normal([256, 10]))
    b1 = tf.Variable(tf.random_normal([256]))
    b2 = tf.Variable(tf.random_normal([256]))
    b3 = tf.Variable(tf.random_normal([10]))

    # Multilayer perceptron
    def dnn(x):
        x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))
        x = tf.nn.tanh(tf.add(tf.matmul(x, W2), b2))
        x = tf.add(tf.matmul(x, W3), b3)
        return x

    net = dnn(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32),
        name='acc')
    
    # 使用TFlearn训练TF搭建的图表
    # Using TFLearn Trainer
    # Define a training op (op for backprop, only need 1 in this model)
    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                              metric=accuracy, batch_size=128)

    # Create Trainer, providing all training ops. Tensorboard logs stored
    # in /tmp/tflearn_logs/. It is possible to change verbose level for more
    # details logs about gradients, variables etc...
    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0)
    # Training for 10 epochs.
    trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY},
                n_epoch=10, show_metric=True)
```
# 2、纯TF

```python
"""
This tutorial will introduce how to combine TFLearn and Tensorflow, using
TFLearn wrappers regular Tensorflow expressions.
"""

import tensorflow as tf
import tflearn

# ----------------------------
# Utils: Using TFLearn Trainer
# ----------------------------

# Loading MNIST complete dataset
import tflearn.datasets.mnist as mnist
# trainX, trainY, testX, testY = mnist.load_data(one_hot=True)
mnist_data = mnist.read_data_sets(one_hot=True)

# Define a dnn using Tensorflow
with tf.Graph().as_default():

    # Model variables
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    W1 = tf.Variable(tf.random_normal([784, 256]))
    W2 = tf.Variable(tf.random_normal([256, 256]))
    W3 = tf.Variable(tf.random_normal([256, 10]))
    b1 = tf.Variable(tf.random_normal([256]))
    b2 = tf.Variable(tf.random_normal([256]))
    b3 = tf.Variable(tf.random_normal([10]))

    # Multilayer perceptron
    def dnn(x):
        x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))
        x = tf.nn.tanh(tf.add(tf.matmul(x, W2), b2))
        x = tf.add(tf.matmul(x, W3), b3)
        return x

    net = dnn(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32),
        name='acc')

    '''
    # Using TFLearn Trainer
    # Define a training op (op for backprop, only need 1 in this model)
    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                              metric=accuracy, batch_size=128)

    # Create Trainer, providing all training ops. Tensorboard logs stored
    # in /tmp/tflearn_logs/. It is possible to change verbose level for more
    # details logs about gradients, variables etc...
    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0)
    # Training for 10 epochs.
    trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY},
                n_epoch=10, show_metric=True)
    '''

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        batch_size = 128
        for epoch in range(2):  # 2 epochs
            avg_cost = 0.
            total_batch = int(mnist_data.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
                cost = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += cost / total_batch
                if i % 20 == 0:
                    print("Epoch:", '%03d' % (epoch + 1), "Step:", '%03d' % i,
                          "Loss:", str(cost))



```

