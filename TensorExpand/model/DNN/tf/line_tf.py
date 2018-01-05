#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
iris 数据 特征列为4，类别数为3
采用线性分类 by tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import argparse
import sys
import pandas as pd
import numpy as np

class Model(object):
    def __init__(self,X,Y,weights,biases,learning_rate):
        self.X=X
        self.Y=Y
        self.weights=weights
        self.biases=biases
        self.learning_rate=learning_rate

    def inference(self,activation='softmax'):
        if activation=='softmax':
            pred=tf.nn.softmax(tf.matmul(self.X, self.weights['out']) + self.biases['out'])
        else:
            pred=tf.nn.bias_add(tf.matmul(self.X, self.weights['out']),self.biases['out'])
        return pred

    def loss(self,pred_value,MSE_error=False,one_hot=True):
        if MSE_error:return tf.reduce_mean(tf.reduce_sum(
            tf.square(pred_value-self.Y),reduction_indices=[1]))
        else:
            if one_hot:
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.Y, logits=pred_value))
            else:
                return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(self.Y, tf.int32), logits=pred_value))

    def evaluate(self,pred_value,one_hot=True):
        if one_hot:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.argmax(self.Y, 1))
            # correct_prediction = tf.nn.in_top_k(pred_value, Y, 1)
        else:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.cast(self.Y, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self,cross_entropy):
        global_step = tf.Variable(0, trainable=False)
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy, global_step=global_step)

start_index = 0
class Inputs(object):
    def __init__(self,file_path,batch_size,one_hot=True):
        self.file_path=file_path
        self.batch_size=batch_size
        self.data=pd.read_csv(self.file_path)

        to_replaced = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        replaced_value = [0, 1, 2]
        self.data = self.data.replace(to_replace=to_replaced[0], value=replaced_value[0]). \
            replace(to_replace=to_replaced[1], value=replaced_value[1]). \
            replace(to_replace=to_replaced[2], value=replaced_value[2]).values

        np.random.seed(100)  # 设置随机因子
        np.random.shuffle(self.data)

    def next_batch(self,data, batch_size):
        global start_index  # 必须定义成全局变量
        global second_index  # 必须定义成全局变量

        second_index = start_index + batch_size
        if second_index > len(data):
            second_index = len(data)
        data1 = data[start_index:second_index]
        # lab=labels[start_index:second_index]
        start_index = second_index
        if start_index >= len(data):
            start_index = 0

        # 将每次得到batch_size个数据按行打乱
        index = [i for i in range(len(data1))]  # len(data1)得到的行数
        np.random.shuffle(index)  # 将索引打乱
        data1 = data1[index]

        # 提起出数据和标签
        # img = data1[:, 0:-1].astype(np.float16)
        img = data1[:, 1:-1].astype(np.float16)
        # 归一化
        img=(img-np.mean(img,0))/(np.var(img,0)+0.001)

        label = data1[:, -1]
        label = label.astype(np.uint8)  # 类型转换

        return img, label

    def inputs(self):
        batch_xs, batch_ys = self.next_batch(self.data,self.batch_size)
        return batch_xs, batch_ys
    def test_inputs(self):
        return (self.data[:,1:-1][:50] - np.mean(self.data[:,1:-1][:50], 0)) / \
               (np.var(self.data[:,1:-1][:50], 0) + 0.001),self.data[:,-1][:50]



FLAGS=None
def train():
    # Input layer
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 4], 'x')
        y_ = tf.placeholder(tf.float32, [None,], 'y_') # 这里使用非one_hot标签

    # Store layers weight & bias
    weights = {
        'out': tf.Variable(tf.random_normal([4, FLAGS.num_class]))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([FLAGS.num_class]))
    }

    input_model = Inputs(FLAGS.data_dir, FLAGS.batch_size, one_hot=FLAGS.one_hot)
    model=Model(x,y_,weights,biases,FLAGS.learning_rate)

    y=model.inference(activation='softmax')
    cross_entropy = model.loss(y, MSE_error=False, one_hot=FLAGS.one_hot)
    train_op = model.train(cross_entropy)
    accuracy = model.evaluate(y, one_hot=FLAGS.one_hot)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        for step in range(FLAGS.num_steps):
            batch_xs, batch_ys = input_model.inputs()
            train_op.run({x: batch_xs, y_: batch_ys})

            if step % FLAGS.disp_step == 0:
                acc=accuracy.eval({x: batch_xs, y_: batch_ys})
                print("step", step, 'acc', acc,
                      'loss', cross_entropy.eval({x: batch_xs, y_: batch_ys}))


        # test acc
        test_x,test_y=input_model.test_inputs()
        acc = accuracy.eval({x: test_x, y_: test_y})
        print('test acc', acc)

def main(_):
    train()

if __name__=="__main__":
    # 设置必要参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=3,
                        help='Number of class.')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help = 'Number of steps to run trainer.')
    parser.add_argument('--disp_step', type=int, default=100,
                        help='Number of steps to display.')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of mini training samples.')
    parser.add_argument('--one_hot', type=bool, default=False,
                        help='One-Hot Encoding.')
    parser.add_argument('--data_dir', type=str, default='../../data/iris.csv',
            help = 'Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)