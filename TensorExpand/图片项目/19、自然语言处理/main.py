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
imdb.maybe_download_and_extract()
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


# 创建循环神经网络
model = Sequential()
'''
RNN中的第一层是所谓的嵌入层，其将每个整数标记转换为值向量。
这是必要的，因为对于10000字的词汇表，整数标记可以取0到10000之间的值。 
RNN无法处理如此广泛的数值。 嵌入层被训练为RNN的一部分，并将学习将具有
相似语义含义的单词映射到相似的嵌入向量，如下面将进一步示出的那样。

首先我们为每个整数标记定义嵌入向量的大小。 在这种情况下，我们将其设置为8，
以便每个整数标记都将转换为长度为8的矢量。嵌入矢量的值通常大致落在-1.0和1.0之间，尽管它们可能会略微超过这些值。

嵌入向量的大小通常在100-300之间选择，但对于情感分析来说，它似乎可以很好地工作。
'''
embedding_size = 8
'''
嵌入层还需要知道词汇表（num_words）中的单词数量和填充的标记序列（max_tokens）的长度。 
我们也给这个图层一个名字，因为我们需要在下面进一步检索它的权重。
'''
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding')) # [n,544,8]

model.add(GRU(units=16, return_sequences=True)) # [n,n,16]

model.add(GRU(units=8, return_sequences=True)) # [n,n,8]

model.add(GRU(units=4)) # [n,4]

model.add(Dense(1, activation='sigmoid')) # [n,1]

optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
layer_embedding (Embedding)  (None, 544, 8)            80000     
_________________________________________________________________
gru_1 (GRU)                  (None, None, 16)          1200      
_________________________________________________________________
gru_2 (GRU)                  (None, None, 8)           600       
_________________________________________________________________
gru_3 (GRU)                  (None, 4)                 156       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 5         
=================================================================
Total params: 81,961
Trainable params: 81,961
Non-trainable params: 0
_________________________________________________________________
'''

# 训练循环神经网络
# 我们使用5％的训练集作为一个小的验证集
model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=3, batch_size=64)

# 性能测试集
result = model.evaluate(x_test_pad, y_test)
print("Accuracy: {0:.2%}".format(result[1]))

# 错误分类的文本示例
# 为了显示错误分类文本的例子，我们首先计算测试集中前1000个文本的预测情绪
y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

# 这些预测数字在0.0到1.0之间。 我们使用截止点/阈值，并且说0.5以上的所有值取1.0，
# 所有低于0.5的值取0.0。 这给了我们一个0.0或1.0的预测“类”。
cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])

# 测试集中前1000个文本的真实“类”是需要比较的。

cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]
# 在使用的1000篇文章中，有多少是错误分类的？
# len(incorrect) #

# 让我们看看第一个错误分类的文本。 我们会多次使用它的索引。
idx = incorrect[0]
print(idx)

# 这些是文本预测的和真实的类别
# y_pred[idx]
# cls_true[idx]

# 新数据
# 让我们尝试分类我们组成的新文本。 其中一些是显而易见的，而另一些则使用否定和讽刺来试图混淆模型，将文本分类错误。
text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]

# 我们首先将这些文本转换为整数标记数组，因为这是模型所需要的。
tokens = tokenizer.texts_to_sequences(texts)

# 为了将不同长度的文本输入到模型中，我们还需要填充和截断它们。
tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
print(tokens_pad.shape) # (8, 544)

# 我们现在可以使用训练好的模型来预测这些文本的情绪
result=model.predict(tokens_pad)
print(result)