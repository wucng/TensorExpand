#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np

# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# 一张图片是28*28,FNN是一次性把数据输入到网络，RNN把它分成块
chunk_size = 28
chunk_n = 28

rnn_size = 256

n_output_layer = 10  # 输出层

X = tf.placeholder('float', [None, chunk_n, chunk_size])
Y = tf.placeholder('float',[None,n_output_layer])


# 定义待训练的神经网络
def recurrent_neural_network(data):
    layer = {'w_': tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
             'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    # initial_state=lstm_cell.zero_state(batch_size,tf.float32) # 限制了每次输入的数据只能是batch_size个数据 （去掉 输入的batch_size就可以随意）
    '''
    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(0, chunk_n, data)
    '''
    ''' 
    # static_rnn
    data=tf.unstack(data,chunk_n,axis=1)
    outputs, status =tf.nn.static_rnn(lstm_cell,data,initial_state=initial_state,dtype=tf.float32)
    # outputs, status = tf.nn.rnn(lstm_cell, data, dtype=tf.float32)
    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
    '''
    # dynamic_rnn
    outputs, status = tf.nn.dynamic_rnn(lstm_cell, data, initial_state=None, dtype=tf.float32)
    ouput = tf.add(tf.matmul(outputs[:, -1, :], layer['w_']), layer['b_'])

    return ouput,status


# 每次使用100条数据进行训练
batch_size = 100


# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict,status= recurrent_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 1
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, 'model.ckpt')
        epoch_loss = 0
        # state=None
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples // batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([-1, chunk_n, chunk_size])

                # if state==None:
                feed_dict = {X: x, Y: y}
                # else:
                #     feed_dict = {X: x, Y: y,initial_state:state}

                _, c,_ = session.run([optimizer, cost_func,status], feed_dict=feed_dict)
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)

        saver.save(session,'model.ckpt')
        '''
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        state= session.run(lstm_cell.zero_state(64, tf.float32))
        print('准确率: ', accuracy.eval({X: mnist.test.images[:64].reshape(-1, chunk_n, chunk_size), Y: mnist.test.labels[:64],initial_state:state}))
        '''


def test_neural_network(X, Y):
    predict,status = recurrent_neural_network(X)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver=tf.train.Saver(tf.global_variables())
        saver.restore(session,'model.ckpt')

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # state_= session.run(lstm_cell.zero_state(64, tf.float32))

        [acc,_]=session.run([accuracy, status], feed_dict={X: mnist.test.images.reshape(-1, chunk_n, chunk_size),Y: mnist.test.labels})

        print('准确率: ', acc)

if __name__=="__main__":
    train=1 # 1 train -1 test
    if train==1:
        train_neural_network(X, Y)
    elif train==-1:
        test_neural_network(X,Y)
    else:
        print('1 train -1 test')
        exit(-1)

