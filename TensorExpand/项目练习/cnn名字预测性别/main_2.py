#! /usr/bin/python
# -*- coding: utf8 -*-

'''
输入：[-1,8]
标签：[-1,2]

model: 卷积

如何将文本数据转成对应的向量，使用到tf.nn.embedding_lookup 或 tflearn.embedding

'''
import tensorflow as tf
import numpy as np
import pandas as pd

def csv2xy(name_dataset):
    '''
    解析CSV数据，得到样本（文本）与标签
    :param name_dataset: CSV路径
    :return: 样本（文本），标签
    '''
    train_x = [] # 存储样本
    train_y = [] # 存储标签
    data=pd.read_csv(name_dataset)
    for i,j in data.values:
        train_x.append(i)
        if j=='男':
            train_y.append([0,1])
        else:
            train_y.append([1,0])

    return train_x,train_y

# 数据已shuffle
#shuffle_indices = np.random.permutation(np.arange(len(train_y)))
#train_x = train_x[shuffle_indices]
#train_y = train_y[shuffle_indices]

# 词汇表（参看聊天机器人练习）
# 统计所有名字对应的字的词汇表，然后根据该词汇表对每个词进行编码 转成对应的数字
def Txt2Vec(train_x):
    '''
    文本转成对应的向量
    :param train_x: 输入的文本 如:[['李珊珊'],['令狐冲'],['洪七公']]
    :return: 返回对应的向量 如：[[20 23 23],[30 35 45],[37 97 89]
    '''
    counter = 0
    vocabulary = {}
    for name in train_x:
        counter += 1
        tokens = [word for word in str(name)]
        for word in tokens:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1

    vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)
    print(len(vocabulary_list))  # 去重后的词汇个数
    # '''

    # 字符串转为向量形式
    vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
    train_x_vec = []
    for name in train_x:
        name_vec=[]
        for word in str(name):
            name_vec.append(vocab.get(word))
        while len(name_vec)<max_name_length:
            name_vec.append(0) # 保证每个序列长度都为8
        train_x_vec.append(name_vec)
    return train_x_vec,vocabulary_list

# 或
def Txt2Vec2(train_x):
    '''
    文本转成对应的向量
    :param train_x: 输入的文本 如:[['李珊珊'],['令狐冲'],['洪七公']]
    :return: 返回对应的向量 如：[[20 23 23],[30 35 45],[37 97 89]
    '''
    vocab_list=[' ']
    for name in train_x:
        [vocab_list.append(word) for word in str(name)]
    # 转成set 实现去重
    vocabulary_list=set(sorted(vocab_list))
    vocabulary_list=list(vocabulary_list) # set-->list 防止元素随意打乱（set元素无序的 会随意打乱）
    print(len(vocabulary_list)) # 去重后的词汇个数
    # '''

    # 字符串转为向量形式
    vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
    train_x_vec = []
    for name in train_x:
        name_vec=[]
        for word in str(name):
            name_vec.append(vocab.get(word))
        while len(name_vec)<max_name_length:
            name_vec.append(0) # 保证每个序列长度都为8
        train_x_vec.append(name_vec)
    return train_x_vec,vocabulary_list


# 序列填充函数
# 参考：tflearn.data_utils.pad_sequences
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences.

    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).

    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)

    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

#######################################################

def neural_network(vocabulary_size, embedding_size=128, num_filters=128):
    # embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X) # [-1,8,128]
        # embedded_chars=tflearn.embedding(net, input_dim=vocabulary_size, output_dim=embedding_size)
        # embedded_chars = slim.embed_sequence(X, vocabulary_size, embedding_size)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) # [-1,8,128,1] 变成4维方便使用卷积

    # convolution + maxpool layer
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    # 使用3个不同的卷积核卷积，最后将结果合并
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters] # [3,128,1,128]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID") # [-1,6,1,128]  (8+2*0-3)/1 +1=6

            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID') # [-1,1,1,128]

            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes) # 128*3
    h_pool = tf.concat(pooled_outputs, 3) # 合并  [-1,1,1,128*3]
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # [-1,128*3]
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer()) # [128*3,2]
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b) # [-1,2]

    return output

# 训练
def train_neural_network():
    output = neural_network(len(vocabulary_list)) # [-1,2]

    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    # train_op=optimizer.minimize(loss)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(201):
            for i in range(num_batch):
                batch_x = train_x_vec[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]

                try:
                    _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
                    print(e,i,loss_)
                except:
                    print('error')
            # 保存模型
            if e % 50 == 0:
                saver.save(sess, "name2sex.model", global_step=e)

# 使用训练的模型
def detect_sex(name_list):
    x = []
    for name in name_list:
        name_vec = []
        for word in name:
            name_vec.append(vocab.get(word))
        while len(name_vec) < max_name_length:
            name_vec.append(0)
        x.append(name_vec)

    output = neural_network(len(vocabulary_list))

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")

        predictions = tf.argmax(output, 1)
        res = sess.run(predictions, {X: x, dropout_keep_prob: 1.0})

        i = 0
        for name in name_list:
            print(name, '女' if res[i] == 0 else '男')
            i += 1

if __name__=="__main__":
    name_dataset = 'name.csv'
    train_x, train_y = csv2xy(name_dataset)

    max_name_length = max([len(str(name)) for name in train_x])
    print("最长名字的字符数: ", max_name_length)
    max_name_length = 8

    train_x_vec,vocabulary_list = Txt2Vec(train_x)

    input_size = max_name_length  # 每个序列长度 8
    num_classes = 2  # 男 或 女

    batch_size = 64
    num_batch = len(train_x_vec) // batch_size

    X = tf.placeholder(tf.int32, [None, input_size])  # [None, 1,input_size] # 1:1个序列，input_size：每个序列长度
    Y = tf.placeholder(tf.float32, [None, num_classes])

    dropout_keep_prob = tf.placeholder(tf.float32)

    # 训练
    train_neural_network()

    # 测试
    neural_network(len(vocabulary_list))

    detect_sex(["白富美", "高帅富", "王婷婷", "田野"])
