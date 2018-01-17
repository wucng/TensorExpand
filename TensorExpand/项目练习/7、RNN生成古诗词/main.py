#! /usr/bin/python
# -*- coding: utf8 -*-

'''
batch_size固定，input_size 不固定
loss 函数 legacy_seq2seq.sequence_loss_by_example


# 做标签时 前一个词预测后一个词
# 如：[寒随穷律变，] --> 寒随穷律变，]]  即 [-->寒, 寒-->随,随-->穷,穷-->律,律-->变,变-->，；，-->]
# 生成文本时 先输入 [ -->词1 词1-->词2 ... 词n-->]  得到诗句 词1，词2 ... 词n
'''

import collections
import numpy as np
import tensorflow as tf
import codecs
from tensorflow.contrib import legacy_seq2seq

#-------------------------------数据预处理---------------------------#

poetry_file ='poetry.txt'
logdir = './model/'
train=-1 # 1 train ;-1 inference

# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8') as f:
    for line in f:
        try:
            title, content = line.strip().split(':')  # strip() 删除字符串中的空白（删除换行符）
            content = content.replace(' ', '') # 删除空格
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
            # break
        except Exception as e:
            pass


# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]

counter = collections.Counter(all_words) # 统计每个字符出现的次数
count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # 排序
words, _ = zip(*count_pairs) # 得到去重后的词汇表 words是元组

# 取前多少个常用字
words = words[:len(words)] + (' ',) # words元组   (' ',)只有一个元素时，不加逗号会认为不是元组
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words)))) # 每个词对应一个索引值 的 字典

# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

# 每次取64首诗进行训练
if train==1:
    batch_size = 64
if train==-1:
    batch_size = 1
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []

# 做标签时 前一个词预测后一个词
# 如：[寒随穷律变，] --> 寒随穷律变，]]  即 [-->寒, 寒-->随,随-->穷,穷-->律,律-->变,变-->，；，-->]
# 生成文本时 先输入 [ -->词1 词1-->词2 ... 词n-->]  得到诗句 词1，词2 ... 词n

for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batches = poetrys_vector[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]

    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    """
    xdata             ydata
    [6,2,4,6,9]       [2,4,6,9,9]
    [1,4,2,8,5]       [4,2,8,5,5]
    """
    x_batches.append(xdata)
    y_batches.append(ydata)

#---------------------------------------RNN--------------------------------------#

input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])

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

    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words) + 1])
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words) + 1, rnn_size]) # len(words) + 1表示词汇表的大小
            inputs = tf.nn.embedding_lookup(embedding, input_data) # [batch_size,-1,rnn_size]

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    # outputs shape [batch_size,-1,rnn_size]
    output = tf.reshape(outputs, [-1, rnn_size])

    logits = tf.matmul(output, softmax_w) + softmax_b # [-1,len(words) + 1]
    probs = tf.nn.softmax(logits)

    return logits, last_state, probs, cell, initial_state

#训练
def train_neural_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1]) # [-1,]
    # loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],
    #                                               len(words))
    loss=legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(50):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            n = 0
            for batche in range(n_chunk):
                train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
                n += 1
                print(epoch, batche, train_loss)
            if epoch % 7 == 0:
                saver.save(sess, logdir+'poetry.module', global_step=epoch)

# -------------------------------生成古诗---------------------------------#
# 使用训练完成的模型

def gen_poetry():
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        sample=min(sample,len(words)-1)
        return words[sample]  # 索引返回对应的词

    _, last_state, probs, cell, initial_state = neural_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        # module_file = tf.train.latest_checkpoint('.')
        # saver.restore(sess, module_file)

        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        state_ = sess.run(cell.zero_state(1, tf.float32))

        # 做标签时 前一个词预测后一个词
        # 如：[寒随穷律变，] --> 寒随穷律变，]]  即 [-->寒, 寒-->随,随-->穷,穷-->律,律-->变,变-->，；，-->]
        # 生成文本时 先输入 [ -->词1 词1-->词2 ... 词n-->]  得到诗句 词1，词2 ... 词n

        x = np.array([list(map(word_num_map.get, '['))])
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})

        word = to_word(probs_) # 随机生成词汇
        #word='梦' # 包含固定词汇
        poem = ''
        # i=0
        while word != ']':
            # i+=1
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_num_map[word]
            '''
            #if word=='。':
            if i==12:
                word='云'
                continue
            if i==24:
                word='烟'
                continue
            '''
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
            # word = words[np.argmax(probs_)]

        return poem

# 生成藏头诗
def gen_poetry_with_head(head):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        sample = min(sample, len(words) - 1)
        return words[sample]  # 索引返回对应的词
        # return words[sample]

    _, last_state, probs, cell, initial_state = neural_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        # saver.restore(sess, 'poetry.module-49')

        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


        state_ = sess.run(cell.zero_state(1, tf.float32))
        poem = ''
        i = 0
        for word in head:
            while word != '，' and word != '。':
                poem += word
                x = np.array([list(map(word_num_map.get, word))])
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
                word = to_word(probs_)
                # time.sleep(1)
            if i % 2 == 0:
                poem += '，'
            else:
                poem += '。'
            i += 1
        return poem



if __name__=="__main__":
    if train==1:
        train_neural_network()
    elif train==-1:
        # print(gen_poetry())
        print(gen_poetry_with_head('一二三四'))
    else:
        print('1 train ;-1 inference')
        exit(-1)
