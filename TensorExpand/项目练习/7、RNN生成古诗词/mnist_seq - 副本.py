#! /usr/bin/python
# -*- coding: utf8 -*-

'''
batch_size固定，
loss 函数 legacy_seq2seq.sequence_loss_by_example

'''

import collections
import numpy as np
import tensorflow as tf
import codecs
from tensorflow.contrib import legacy_seq2seq
from tensorflow.examples.tutorials.mnist import input_data
import cv2

mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)

#---------------------------------------RNN--------------------------------------#

# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
        # cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        # cell_fun = tf.nn.rnn_cell.BasicLSTMCell
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    # tf.contrib.rnn.BasicRNNCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)

    # ------------------#
    # inputs=tf.reshape(input_data,[batch_size,28,28,1])
    inputs=input_data
    inputs=tf.layers.conv2d(inputs,128,3,padding='SAME',activation=tf.nn.relu)
    inputs=tf.reshape(inputs,[batch_size,28*28,128])
    # ------------------------#

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    # outputs shape [batch_size,28*28,rnn_size]
    outputs = tf.reshape(outputs, [-1, rnn_size]) # [batch_size*28*28,rnn_size]

    # logits = tf.matmul(output, softmax_w) + softmax_b # [-1,len(words) + 1]
    # probs = tf.nn.softmax(logits)
    logits=tf.layers.dense(outputs,256,activation=None) # [batch_size*28*28,256]
    probs = tf.nn.softmax(logits)

    return logits, last_state, probs, cell, initial_state

#训练
def train_neural_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1]) # [batch_size*28*28,]
    # loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],
    #                                               len(words))
    loss=legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)])

    # loss = legacy_seq2seq.sequence_loss([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)])

    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        n_chunk=len(mnist.train.images)//batch_size
        for epoch in range(epochs):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            # n = 0
            for batche in range(21):# range(n_chunk):
                x_batches,_=mnist.train.next_batch(batch_size)
                # y_batches=np.round(x_batches).astype(np.uint8)
                train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: x_batches.reshape([-1,28,28,1]), output_targets: x_batches})
                # n += 1
                if batche %20==0:
                    print(epoch, batche, train_loss)
            if epoch % 1 == 0:
                saver.save(sess, logdir+'model.ckpt', global_step=epoch)


def generate_mnist():
    logits, _, _, _, _ = neural_network() # [batch_size*28*28,256]

    predict = tf.reshape(tf.multinomial(tf.nn.softmax(logits), num_samples=1, seed=100), tf.shape(input_data))  # [batch,28,28,1]
    # predict=tf.reshape(logits,[batch_size,28,28,1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        # saver.restore(sess, 'girl.ckpt-49')
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        pics = np.zeros((batch_size, 28, 28, 1), dtype=np.float32)

        for i in range(28):
            for j in range(28):
                for k in range(1):
                    next_pic = sess.run(predict, feed_dict={input_data: pics})
                    pics[:, i, j, k] = next_pic[:, i, j, k]

        cv2.imwrite('mnist.jpg', pics[0])
        print('生成mnist图: mnist.jpg')


if __name__=="__main__":
    batch_size = 128
    epochs = 1
    input_data = tf.placeholder(tf.float32, [None, 28, 28, 1])  # [None,28,28,1]
    output_targets = tf.placeholder(tf.int32, [None, 28*28])  # [None,28*28]

    logdir='./model/'

    train =1

    if train==1:
        train_neural_network()
    elif train==-1:
        generate_mnist()
    else:
        print('1 train ;-1 inference')
        exit(-1)