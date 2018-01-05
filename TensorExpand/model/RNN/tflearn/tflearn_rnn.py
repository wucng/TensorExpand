# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import

import tflearn
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
trainX = pad_sequences(trainX, maxlen=input_length_each_seq, value=0.) # 每个序列长度都填充为100
testX = pad_sequences(testX, maxlen=input_length_each_seq, value=0.) # [22500,100]
# Converting labels to binary vectors
trainY = to_categorical(trainY,n_class)
testY = to_categorical(testY,n_class)

# Network building
net = tflearn.input_data([None, input_length_each_seq]) # [none,100]
net = tflearn.embedding(net, input_dim=n_datas, output_dim=hiddle_layes)
# print(net.shape) # (?, 100, 128)
# exit(-1)
# net = tflearn.input_data([None, time_seqs,input_length_each_seq])
net = tflearn.lstm(net, hiddle_layes_2, dropout=0.8)
net = tflearn.fully_connected(net, n_class, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY,n_epoch=1, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
