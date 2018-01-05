#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
iris 数据 特征列为4，类别数为3
采用线性分类 by sklearn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf
import os
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn import datasets,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


start_index = 0
class Inputs(object):
    def __init__(self,file_path,batch_size=32,one_hot=True):
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

    def next_batch(self):
        global start_index  # 必须定义成全局变量
        global second_index  # 必须定义成全局变量

        second_index = start_index + self.batch_size
        if second_index > len(self.data):
            second_index = len(self.data)
        data1 = self.data[start_index:second_index]
        # lab=labels[start_index:second_index]
        start_index = second_index
        if start_index >= len(self.data):
            start_index = 0

        # 将每次得到batch_size个数据按行打乱
        index = [i for i in range(len(data1))]  # len(data1)得到的行数
        np.random.shuffle(index)  # 将索引打乱
        data1 = data1[index]

        # 提起出数据和标签
        # img = data1[:, 0:-1].astype(np.float16)
        img = data1[:, 1:-1].astype(np.float16)

        # 归一化
        # img=(img-np.mean(img,0))/(np.var(img,0)+0.001)
        img=preprocessing.scale(img,0)

        label = data1[:, -1]
        label = label.astype(np.uint8)  # 类型转换

        return img, label

    def inputs(self):
        batch_xs, batch_ys = self.next_batch()
        return batch_xs, batch_ys
    def test_inputs(self):
        # return (self.data[:,0:-1][:50] - np.mean(self.data[:,0:-1][:50], 0)) / \
        #        (np.var(self.data[:,0:-1][:50], 0) + 0.001),self.data[:,-1][:50]
        return preprocessing.scale(self.data[:,1:-1][:50],0),self.data[:,-1][:50]


FLAGS=None
def train():
    data=Inputs(file_path=FLAGS.data_dir,batch_size=FLAGS.batch_size)
    data_X,data_Y=data.inputs()
    print(data_X.shape,data_Y.shape) # (150, 4) (150,)
    test_X,test_Y=data.test_inputs()
    iris_model = LinearRegression(normalize=True)
    iris_model.fit(data_X,data_Y)
    print(iris_model.score(test_X,test_Y))
    # print(iris_model.get_params())
    print('w:',iris_model.coef_,'b:',iris_model.intercept_)

def main():
    train()

if __name__=="__main__":
    # 设置必要参数
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=150,
                        help='Number of mini training samples.')

    parser.add_argument('--data_dir', type=str, default='../../data/iris.csv',
            help = 'Directory for storing input data')

    FLAGS, unparsed = parser.parse_known_args()
    train()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

'''
0.950169676916
w: [-0.0909303  -0.01902717  0.3994506   0.46298156] b: 0.999984238301
'''

