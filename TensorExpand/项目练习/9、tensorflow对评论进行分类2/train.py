#! /usr/bin/python
# -*- coding: utf8 -*-

'''
由于训练样本太大，不宜直接全部转成向量，而是按batc_size转成向量，减少内存消耗
'''

import os
import random
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()


def get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline() # 读取一行


# 从文件中随机选择n条记录
def get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size # 统计文件size
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines


def get_test_dataset(test_file):
    with open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y


test_x, test_y = get_test_dataset('tesing.csv')

#######################################################################

n_input_layer = len(lex)  # 输入层

n_layer_1 = 2000  # hide layer
n_layer_2 = 2000  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 3  # 输出层

# dnn
def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output

input_size = len(lex)
num_classes = 3
# cnn  也可以使用1D卷积
def cnn_net():
    # embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedding_size = 128
        W = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) # [batch_size,len(lex),128,1]
    # convolution + maxpool layer
    num_filters = 128
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters] # [3,128,1,128]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID") # [batch_size,len(lex)-2,1,128]
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], # [batch_size,1,1,128]
                                    padding='VALID')
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes) # 3*128
    h_pool = tf.concat(pooled_outputs,3) # [batch_size,1,1,128*3]
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # [batch_size,128*3]
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b) # [batch_size,3]

    return output


def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    '''
    optimizer = tf.train.AdamOptimizer(1e-3)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
	grads_and_vars = optimizer.compute_gradients(loss)
	train_op = optimizer.apply_gradients(grads_and_vars)
	
    '''

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        lemmatizer = WordNetLemmatizer()
        saver = tf.train.Saver(tf.global_variables())
        i = 0
        pre_accuracy = 0
        while True:  # 一直训练
            batch_x = []
            batch_y = []

            # if model.ckpt文件已存在:
            #	saver.restore(session, 'model.ckpt')  恢复保存的session

            try:
                lines = get_n_random_line('training.csv', batch_size)
                for line in lines:
                    label = line.split(':%:%:%:')[0]
                    tweet = line.split(':%:%:%:')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大

                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                session.run([optimizer, cost_func], feed_dict={X: batch_x, Y: batch_y})
            except Exception as e:
                print(e)

            # 准确率
            if i > 100:
                correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                accuracy = accuracy.eval({X: test_x, Y: test_y})
                if accuracy > pre_accuracy:  # 保存准确率最高的训练模型
                    print('准确率: ', accuracy)
                    pre_accuracy = accuracy
                    saver.save(session, 'model.ckpt')  # 保存session
                i = 0
            i += 1

# 使用训练好的模型
def prediction(tweet_text):
    predict = neural_network(X)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, 'model.ckpt')

        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(tweet_text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1

        # print(predict.eval(feed_dict={X:[features]})) [[val1,val2,val3]]
        res = session.run(tf.argmax(predict.eval(feed_dict={X: [features]}), 1))
        return res

if __name__=='__main__':
    X = tf.placeholder('float')
    Y = tf.placeholder('float')
    batch_size = 90

    train=-1 # 1 train ;-1 inference

    if train == 1:
        train_neural_network(X, Y)
    elif train == -1:
        print(prediction("I am very happe"))
    else:
        print('1 train ;-1 inference')
        exit(-1)


