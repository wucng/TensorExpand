#! /usr/bin/python
# -*- coding: utf8 -*-

'''
使用pytorch自带的批训练

nn+rnn

输入 x 数据类型必须是：float32、float64 shape[none,c,h,w] ,tensorflow [none,h,w,c]
输入 y 数据类型必须是：uint8、int16、int32、int64 , 且是 非one_hot 标签

# pytorch only supported types are: double, float, int64, int32, and uint8.
x=np.array([1,2,3],np.float16) # float16
x=torch.from_numpy(x) # 报错
x=Variable(x) # 报错

x=np.array([1,2,3],np.float32) # float32 (float)
x=torch.from_numpy(x) # FloatTensor
x=Variable(x) # FloatTensor

# np.float 等价于np.float64
x=np.array([1,2,3],np.float64) # float64 (double)
x=torch.from_numpy(x) # DoubleTensor
x=Variable(x) # DoubleTensor

x=np.array([1,2,3],np.uint8) # uint8
x=torch.from_numpy(x) # ByteTensor
x=Variable(x) # ByteTensor

x=np.array([1,2,3],np.int8) # int8
x=torch.from_numpy(x) # # 报错
x=Variable(x) # # 报错

x=np.array([1,2,3],np.int16) # int16
x=torch.from_numpy(x) # ShortTensor
x=Variable(x) # ShortTensor

# np.int 等价于np.int32
x=np.array([1,2,3],np.int32) # int32
x=torch.from_numpy(x) # IntTensor
x=Variable(x) # IntTensor

x=np.array([1,2,3],np.int64) # int64
x=torch.from_numpy(x) # LongTensor
x=Variable(x) # LongTensor
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import pandas as pd
# import tensorflow as tf
import numpy as np
import glob

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
display_step = 10
EPOCH=3

# Network Parameters
img_h=19
img_w=19
img_c=4
n_input = img_h*img_w*img_c
n_classes = 2 #
dropout = 0.75 #
hide_layers=128
hide_layers_2=256

# 加载数据
filepaths=glob.glob('./data_*.pkl')
for i,filepath in enumerate(filepaths):
    if i==0:
        data=pd.read_pickle(filepath)
    else:
        data=np.vstack((data,pd.read_pickle(filepath)))
# np.random.shuffle(data)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# train_loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
# xx=np.reshape(data[:,:-1],[-1,img_h,img_w,img_c]).transpose([0,3,1,2]).astype(np.float32) # [-1,4,19,19]
xx=np.reshape(data[:,:-1],[-1,img_h,img_w*img_c]).astype(np.float32) # [-1,19,19*4]
yy=data[:,-1].astype(np.uint8) # uint8、int16、int32、int64 , 且是 非one_hot 标签
torch_dataset = Data.TensorDataset(data_tensor=torch.from_numpy(xx), target_tensor=torch.from_numpy(yy))

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (4, 19, 19)
            nn.Conv2d(
                in_channels=img_c,              # input height
                out_channels=32,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 19, 19)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 9, 9)
        )
        self.conv2 = nn.Sequential(         # input shape (32, 9, 9)
            nn.Conv2d(32, 64, 5, 1, 2),     # output shape (64, 9, 9)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 4, 4)
        )
        self.out = nn.Linear(64 * 4 * 4, n_classes)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 64 * 5 * 5)
        output = self.out(x)
        return output#, x    # return x for visualization
'''
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.input=nn.Linear(img_w*img_c,hide_layers) # [-1,19,128]

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=hide_layers,
            hidden_size=hide_layers_2,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hide_layers_2, n_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size) [-1,19,19*4]
        # r_out shape (batch, time_step, output_size) [-1,19,256]
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x=self.input(x) # [-1,19,128]
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = rnn(b_x)             # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            # test_output = cnn(test_x)
            pred_y = torch.max(output, 1)[1].data.squeeze() # LongTensor 非Variable
            accuracy = sum(pred_y == y) / float(y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
