#! /usr/bin/python
# -*- coding: utf8 -*-

'''
# 输入x不定长 ,此时只能一条条处理 即batc_size=1 否则会报错
# 或者 找到最长的序列，其他的用0 填充 保证所有序列长度一致
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim, rnn, legacy_seq2seq
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ] 
"""
from nltk.stem import WordNetLemmatizer

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""

pos_file = 'pos.txt'
neg_file = 'neg.txt'


# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []

    # 读取文件
    def process_file(pos_file):
        with open(pos_file, 'r') as f:
            lex = []
            lines = f.readlines()  # 读取全部行
            # print(lines)
            for line in lines:  # 每一行对应一条评论
                words = word_tokenize(line.lower())  # 每一行拆成一个个单词（不是字母）
                lex += words  # 将所有单词 放在列表中 （包括重复的）
            return lex

    lex += process_file(pos_file)
    lex += process_file(neg_file)
    # print(len(lex))
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]  # 词形还原 (cats->cat) 过去时 进行时 等还原成一般形式

    word_count = Counter(lex)  # 统计每个词出现的频率 返回类似一个字典（起到去重） 转成字典 dict(word_count.items())
    # print(word_count)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比  选取词出现的频率 20~2000之间的词
            lex.append(word)  # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
    return lex


lex = create_lexicon(pos_file, neg_file)  # 词汇表


# lex里保存了文本中出现过的单词。

# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = word_tokenize(review.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))  # 取词汇表的长度（保证所有特征长度相等） 而不是取每条评论的长度（每条评论长度不一致）
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return [features, clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
            dataset.append(one_sample)

    # print(len(dataset))
    return dataset


def normalize_dataset_1(lex):
    dataset = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = word_tokenize(review.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # features = np.zeros(len(lex)) # 取词汇表的长度（保证所有特征长度相等） 而不是取每条评论的长度（每条评论长度不一致）
        features = np.zeros(len(words))  # 特征长度不固定

        for word in words:
            if word in lex:
                features[words.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return [features, clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
            dataset.append(one_sample)

    # print(len(dataset))
    return dataset


# dataset = normalize_dataset(lex)
dataset = normalize_dataset_1(lex)  # 特征输入长度不固定
random.shuffle(dataset)
"""
#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
"""

# 取样本中的10%做为测试数据
test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)  # [10662,2]

train_dataset = dataset[:-test_size]  # [9596,2]  train_dataset[0][0] 对应评论数据 ；train_dataset[0][1] 对应评论数据的标签
test_dataset = dataset[-test_size:]  #

# Feed-Forward Neural Network
# 定义每个层有多少'神经元''
n_input_layer = len(lex)  # 输入层 词汇表尺寸，也是每条评论对应的向量长度

n_layer_1 = 100  # 1000  # hide layer
n_layer_2 = 1000  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 2  # 输出层


# 定义待训练的神经网络（DNN）
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


# 输入x不定长 ,此时只能一条条处理 即batc_size=1 否则会报错
# 或者 找到最长的序列，其他的用0 填充 保证所有序列长度一致
def rnn_net_1(x):
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_layer_1, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    inputs = slim.embed_sequence(tf.cast(x, tf.int32), vocab_size=len(lex),
                                 embed_dim=n_layer_1)  # # [batch_size,-1,n_layer_1]
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    # outputs shape [batch_size,-1,n_layer_1]
    # outputs = tf.reshape(outputs, [-1, n_layer_1])
    logits = slim.fully_connected(outputs[:, -1, :], n_output_layer, activation_fn=None)  # [-1,2]

    return logits


# rnn 网络 输入x 定长
def rnn_net(x):
    # 将x 从二维转成3维 目的使用rnn（输入需是3维数据）
    inputs = slim.embed_sequence(tf.cast(x, tf.int32), vocab_size=len(lex),
                                 embed_dim=n_layer_1)  # [none,len(train_dataset[0][0]),n_layer_1] 3维
    cell = tf.contrib.rnn.BasicLSTMCell(n_layer_1, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 4, state_is_tuple=True)  # 多层核
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    # outputs shape [batch_size,len(train_dataset[0][0]),n_layer_1]

    results = slim.fully_connected(outputs[:, -1, :], n_output_layer, activation_fn=None)  # [batch,2]

    return results


# 每次使用50条数据进行训练
batch_size = 1  # 50

# X = tf.placeholder('float', [None, len(train_dataset[0][0])])
# [None, len(train_x)]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
# Y = tf.placeholder('float') # Y = tf.placeholder('float',[None,2])

# 输入不定长
X = tf.placeholder(tf.int32, [batch_size, None])  # 不定长
Y = tf.placeholder(tf.int32, [batch_size, None])  # 定长


# 使用数据训练神经网络
def train_neural_network(X, Y):
    # predict = neural_network(X)
    # predict = rnn_net(X)
    predict = rnn_net_1(X)  # [batch_size,2]  Y shape [batch_size,]

    # loss = legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],
    #                                                len(lex))
    # cost = tf.reduce_mean(loss)
    # cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))

    cost_func = legacy_seq2seq.sequence_loss([predict], [Y], [tf.ones_like(Y, dtype=tf.float32)], len(lex))
    cost = tf.reduce_mean(cost_func)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(0.001).apply_gradients(zip(grads, tvars))
    # optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001

    epochs = 13
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0

        # i = 0
        random.shuffle(train_dataset)
        train_x = train_dataset[:, 0]
        train_y = train_dataset[:, 1]

        steps = len(train_x) // batch_size

        for epoch in range(epochs):
            start = 0;
            end = 0
            for step in range(steps):
                end = start + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                batch_x = list(batch_x)
                batch_y = list(batch_y)

                _, c = session.run([optimizer, cost_func], feed_dict={X: batch_x, Y: batch_y})
                epoch_loss += c

                start = end

            '''       
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
                epoch_loss += c
                i += batch_size
            '''
            print(epoch, ' : ', epoch_loss / steps)

        text_x = test_dataset[:, 0]
        text_y = test_dataset[:, 1]
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: list(text_x), Y: list(text_y)}))


train_neural_network(X, Y)
