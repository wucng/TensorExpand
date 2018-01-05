应用一维卷积网络对IMDB情感数据集中的单词序列进行分类。
[toc]

# 1、TFlearn
```python
# -*- coding: utf-8 -*-
"""
Simple example using convolutional neural network to classify IMDB
sentiment dataset.
References:
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
    - Kim Y. Convolutional Neural Networks for Sentence Classification[C]. 
    Empirical Methods in Natural Language Processing, 2014.
Links:
    - http://ai.stanford.edu/~amaas/data/sentiment/
    - http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


n_class=2
hiddle_layes=128
hiddle_layes_2=128
time_seqs=1
input_length_each_seq=100

n_datas=10000
# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=n_datas,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=input_length_each_seq, value=0.)
testX = pad_sequences(testX, maxlen=input_length_each_seq, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY,n_class)
testY = to_categorical(testY,n_class)

# Building convolutional network
network = input_data(shape=[None, input_length_each_seq], name='input') # [none,100]
network = tflearn.embedding(network, input_dim=n_datas, output_dim=128) # [none,100,128]
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2") # [none,98,128]
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2") # [none,97,128]
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2") # [none,96,128]
network = merge([branch1, branch2, branch3], mode='concat', axis=1) # [none,291,128]
network = tf.expand_dims(network, 2) # [none,291,1,128]
network = global_max_pool(network) # [none,128]
network = dropout(network, 0.5)
network = fully_connected(network, n_class, activation='softmax') # [none,2]
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch = 5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)
```
# 2、TFlearn+tf

```python
# -*- coding: utf-8 -*-
"""
Simple example using convolutional neural network to classify IMDB
sentiment dataset.
References:
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
    - Kim Y. Convolutional Neural Networks for Sentence Classification[C]. 
    Empirical Methods in Natural Language Processing, 2014.
Links:
    - http://ai.stanford.edu/~amaas/data/sentiment/
    - http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


n_class=2
hiddle_layes=128
hiddle_layes_2=128
time_seqs=1
input_length_each_seq=100

n_datas=10000
# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=n_datas,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=input_length_each_seq, value=0.)
testX = pad_sequences(testX, maxlen=input_length_each_seq, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY,n_class)
testY = to_categorical(testY,n_class)

# Building convolutional network
network = input_data(shape=[None, input_length_each_seq], name='input') # [none,100]

W=tf.get_variable('W',[n_datas,128],initializer=tf.random_uniform_initializer)
network = tf.cast(network, tf.int32)
network = tf.nn.embedding_lookup(W, network)
# network = tflearn.embedding(network, input_dim=n_datas, output_dim=128) # [none,100,128]

branch1=tf.layers.conv1d(network,128,3,padding='valid',activation=tf.nn.relu,kernel_regularizer=tf.nn.l2_loss)
branch2=tf.layers.conv1d(network,128,4,padding='valid',activation=tf.nn.relu,kernel_regularizer=tf.nn.l2_loss)
branch3=tf.layers.conv1d(network,128,5,padding='valid',activation=tf.nn.relu,kernel_regularizer=tf.nn.l2_loss)
network=tf.concat([branch1,branch2,branch3], 1)
# branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2") # [none,98,128]
# branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2") # [none,97,128]
# branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2") # [none,96,128]
# network = merge([branch1, branch2, branch3], mode='concat', axis=1) # [none,291,128]

network = tf.expand_dims(network, 2) # [none,291,1,128]

network=tf.reduce_max(network, [1, 2])
# network = global_max_pool(network) # [none,128]

network = dropout(network, 0.5)
network = fully_connected(network, n_class, activation='softmax') # [none,2]
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch = 1, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)
```
