# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist


# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
'''
我们将使用由IMDB电影评论组成的50000个数据集。 
Keras有一个内置的功能来下载一个类似的数据集（但显然是一半的大小）。 
然而，Keras的版本已经将数据集中的文本转换为整数标记，这是使用自然语言工作的关键部分，
本教程中还将演示这些语言，因此我们下载了实际的文本数据。
注意：数据集为84 MB，将自动下载。
'''
import imdb
# 如果要将文件保存在其他目录中，请更改此项。
# imdb.data_dir = "data/IMDB/"

# 自动下载并提取文件。
# imdb.maybe_download_and_extract()
x_train_text, y_train = imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)

# 将下面的一些用途组合成一个数据集。
data_text = x_train_text + x_test_text
# 从训练集打印一个例子，看看数据看起来是否正确。
# print(x_train_text[1])

# 构建词汇表
num_words = 10000 # 只选取其中10000个词构建成词汇表
tokenizer = Tokenizer(num_words=num_words)
# print(tokenizer)

tokenizer.fit_on_texts(data_text) # 得到构建的词汇表

'''
if num_words is None:
    num_words = len(tokenizer.word_index)

print(num_words) # 10000
'''
# print(tokenizer.word_index)

# 根据建立的词汇表将文本数据转成词向量
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
# print(type(x_train_tokens)) ;exit(-1) # <class 'list'>
# print(type(x_train_text)) # list
# np.asarray(x_train_text).shape # (25000,)

x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

# 由于没条文本长度不一致，没法按批训练数据（每批次数据长度需一致），
# 但可以一条一条的训练（batch_size=1）
# A）要么我们确保整个数据集中的所有序列具有相同的长度
# B）或我们编写一个自定义数据生成器，以确保序列在每个批次中具有相同的长度

# 解决方案（A）更简单，但如果我们使用数据集中最长序列的长度，那么我们浪费了大量内存。 这对于大型数据集尤为重要。
# 所以为了做出妥协，我们将使用涵盖数据集中大部分序列的序列长度，然后我们将截断较长的序列并填充较短的序列。
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

# 我们允许的向量的最大数量设置为平均值加2个标准差
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

# 这涵盖了约95％的数据集
# np.sum(num_tokens < max_tokens) / len(num_tokens) # 0.945

'''
当填充或截断具有不同长度的序列时，我们需要确定是否要填充或截断“pre”或“post”。 
如果序列被截断，则意味着序列的一部分被简单地丢弃。 如果序列被填充，则表示将零添加到序列中。

所以'pre'或'post'的选择可能很重要，因为它决定了我们是否在截断时丢弃了序列的第一部分或最后一部分，
并决定了在填充时是否将序列的零或开头添加到序列的末尾。 这可能会混淆循环神经网络。
'''
pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)

x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

print(x_train_pad.shape) # (25000, 544)
print(x_test_pad.shape) # (25000, 544)

# 反向映射 词向量转成对应的文本
idx = tokenizer.word_index # dict
inverse_map = dict(zip(idx.values(), idx.keys()))
# print(idx)
'''
{"'rabid": 60582,
 'elliptic': 67416,
 'angelic': 16008,
 'galactic': 23852,
 "'breakin'": 85947,
 'suchen': 117451,
 ...
 
'''

# 辅助函数用于将词向量列表转换回单词串。
def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]

    # Concatenate all words.
    text = " ".join(words)

    return text

# 我们可以通过将标记列表转换回单词来重新创建除标点符号和其他符号之外的文本
# tokens_to_string(x_train_tokens[1])

batch_size=64
# ---------改成layer API--------------#
from tensorflow.contrib import slim,rnn
x=tf.placeholder(tf.float32,[None,544])
Y=tf.placeholder(tf.float32,[None,])

inputs=slim.embed_sequence(ids=tf.cast(x, tf.int32),vocab_size=num_words,embed_dim=8,scope='layer_embedding')
cell1=rnn.GRUCell(16,activation=tf.nn.relu)
cell2=rnn.GRUCell(8,activation=tf.nn.relu)
cell3=rnn.GRUCell(4,activation=tf.nn.relu)

cell=rnn.MultiRNNCell([cell1,cell2,cell3],state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
predict=slim.fully_connected(outputs[:,-1,:], 2, activation_fn=None)  # [-1,2]

cost_func = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=tf.to_int64(Y)))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost_func)

accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict,1),tf.to_int64(Y)),tf.float32))


epochs = 3
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    epoch_loss = 0
    steps=len(x_train_pad)//batch_size

    for epoch in range(epochs):
        start = 0
        end = 0
        for step in range(steps):
            end = start + batch_size
            batch_x = x_train_pad[start:end]
            batch_y = y_train[start:end]
            # batch_x = list(batch_x)
            # batch_y = list(batch_y)

            _, c,acc = session.run([optimizer, cost_func,accuracy], feed_dict={x: batch_x, Y: batch_y})
            epoch_loss += c

            start = end

            if step%100==0:
                print('step',step,'|','loss',c,'|','acc',acc)

        print(epoch, ' : ', epoch_loss / steps)


    # 测试
    acc=[]
    steps_2 = len(x_test_pad[:500]) // batch_size
    start = 0
    end = 0
    for step in range(steps_2):
        end = start + batch_size
        batch_x = x_test_pad[start:end]
        batch_y = y_test[start:end]

        acc_ = session.run(accuracy, feed_dict={x: batch_x, Y: batch_y})
        acc.append(acc_)

        start = end
        if step % 2 == 0:
            print('step', step, '|', 'acc', acc_)

    print('mean acc',np.mean(acc))


    # 新数据
    text1 = "This movie is fantastic! I really like it because it is so good!"
    text2 = "Good movie!"
    text3 = "Maybe I like this movie."
    text4 = "Meh ..."
    text5 = "If I were a drunk teenager then this movie might be good."
    text6 = "Bad movie!"
    text7 = "Not a good movie!"
    text8 = "This movie really sucks! Can I get my money back please?"
    texts = [text1, text2, text3, text4, text5, text6, text7, text8]
    tokens = tokenizer.texts_to_sequences(texts)
    tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                               padding=pad, truncating=pad)

    pred=session.run(predict,{x:tokens_pad})
    print('pred',np.argmax(pred,1))