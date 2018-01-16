#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 超参数
a=0.5
lr=0.5
batch_size=64
epochs=1
displayer=5

# 建立神经网络编码器
Ax=tf.placeholder(tf.float32,[None,28*28*1],name='Ax')
Px=tf.placeholder(tf.float32,[None,28*28*1],name='Px')
Nx=tf.placeholder(tf.float32,[None,28*28*1],name='Nx')

def conv_net(x,reuse):
    x=tf.reshape(x,[-1,28,28,1])

    conv1=tf.layers.conv2d(x,32,3,padding='SAME',activation=tf.nn.relu,reuse=reuse,name='conv1') # [-1,28,28,32]
    pool1=tf.layers.max_pooling2d(conv1,2,2,'SAME',name='pool1') # [-1,14,14,32]

    conv2 = tf.layers.conv2d(pool1, 64, 3, padding='SAME', activation=tf.nn.relu,reuse=reuse, name='conv2')  # [-1,14,14,64]
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME', name='pool2')  # [-1,7,7,64]

    conv3 = tf.layers.conv2d(pool2, 64, 3, padding='SAME', activation=tf.nn.relu,reuse=reuse, name='conv3')  # [-1,7,7,64]
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'SAME', name='pool3')  # [-1,4,4,64]

    fc1=tf.reshape(pool3,[-1,4*4*64],name='flatten') # [-1,4*4*64]
    fc1=tf.layers.dense(fc1,1024,activation=tf.nn.relu,reuse=reuse,name='fc1') # [-1,1024]

    out = tf.layers.dense(fc1, 128, activation=None,reuse=reuse,name='fc2') # [-1,128]

    return out

A_out=conv_net(Ax,None)
P_out=conv_net(Px,True)
N_out=conv_net(Nx,True)

# loss function
'''
loss_AP=tf.losses.mean_squared_error(A_out,P_out)
loss_AN=tf.losses.mean_squared_error(A_out,N_out)
loss_AP_mean=loss_AP
loss_AN_mean=loss_AN
loss=tf.reduce_mean(tf.maximum(loss_AP-loss_AN+a,0))
'''
loss_AP=tf.square(A_out-P_out)
loss_AN=tf.square(A_out-N_out)
loss_AP_mean=tf.reduce_mean(loss_AP)
# loss_AN_mean=tf.reduce_mean(loss_AN)
loss=tf.reduce_mean(tf.maximum(loss_AP-loss_AN+a,0))

train_op=tf.train.AdamOptimizer(lr).minimize(loss)

# Initializing the variables
# init = tf.global_variables_initializer()

# 加载数据
def LoadImages(paths):
    '''
    加载mnist图片数据
    :param paths: 图片路径
    :return: 影像数据（不需要标签）
    '''
    imgs=glob.glob(paths)
    flag=True
    for img in imgs:
        data=Image.open(img)
        data=np.array(data,np.uint8)/255.
        data=data.flatten()[np.newaxis,:]
        if flag:
            data2=data
            flag=False
        else:
            data2=np.vstack((data2,data))

    return data2

data_0=LoadImages('D:/mnist_images/0/*.bmp')
data_1=LoadImages('D:/mnist_images/1/*.bmp')
data_2=LoadImages('D:/mnist_images/2/*.bmp')

np.random.shuffle(data_0)
np.random.shuffle(data_1)
np.random.shuffle(data_2)

data_train=[]
data_test=[]

data_train.append(data_0[10:])
data_train.append(data_1[10:])
data_train.append(data_2[10:])

data_test.append(data_0[:10])
data_test.append(data_1[:10])
data_test.append(data_2[:10])

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for index in range(3): # 3类数据
        steps = len(data_train[index]) // batch_size
        np.random.shuffle(data_train[index])
        A=data_train[index]

        index_ = np.arange(0, len(A))
        np.random.shuffle(index_)
        P = A[index_]
        '''
        np.random.shuffle(data_train[index])
        P = data_train[index]
        '''

        if index==0:
            N=np.vstack((data_train[index+1],data_train[index+2]))
            np.random.shuffle(N)
        if index==1:
            N=np.vstack((data_train[index-1],data_train[index+1]))
            np.random.shuffle(N)
        if index==2:
            N=np.vstack((data_train[index-2],data_train[index-1]))
            np.random.shuffle(N)

        for epoch in range(epochs):
            start = 0;end=0
            for step in range(steps):
                end=min(len(data_train[index]),start+batch_size)

                feed_dict={Ax:A[start:end],Px:P[start:end],Nx:N[start:len(A[start:end])+start]}

                sess.run(train_op, feed_dict=feed_dict)

                if end==len(data_train[index]):
                    start=0
                else:
                    start=end

                if step%displayer==0:
                    loss_=sess.run(loss,feed_dict)
                    print('index', index, '|', 'epoch', epoch, '|', 'step', step, '|', 'loss', loss_)

                    '''
                    loss_,loss_AP_mean_,loss_AN_mean_=sess.run([loss,loss_AP_mean,loss_AN_mean],feed_dict)
                    print('index',index,'|','epoch',epoch,'|','step',
                          step,'|','loss',loss_,'loss_AP_mean',loss_AP_mean_,'loss_AN_mean_',loss_AN_mean_)
                    '''

    # test
    '''
    for index in range(3):  # 3类数据

        A = data_test[index]
        np.random.shuffle(A)
        index_=np.arange(0,len(A))
        np.random.shuffle(index_)
        P=A[index_]

        if index==0:
            N=np.vstack((data_test[index+1],data_test[index+2]))
            np.random.shuffle(N)
        if index==1:
            N=np.vstack((data_test[index-1],data_test[index+1]))
            np.random.shuffle(N)
        if index==2:
            N=np.vstack((data_test[index-2],data_test[index-1]))
            np.random.shuffle(N)

        for i in range(len(A)):
            feed_dict = {Ax: A[i][np.newaxis,:], Px: P[i][np.newaxis,:], Nx: N[i][np.newaxis,:]}
            loss_ = sess.run(loss, feed_dict)

            print('index', index, '|', 'loss', loss_)
    '''

    A=np.vstack((data_test[0],data_test[1]))
    A=np.vstack((A,data_test[2]))

    # np.random.shuffle(A)

    P0=data_train[0]
    P1=data_train[1]
    P2=data_train[2]

    label=[]
    real=[0]*10+[1]*10+[2]*10
    for i in range(len(A)):
        feed_dict_1={Ax:A[i][np.newaxis,:],Px:P0[np.random.choice(len(P0),1)]}
        feed_dict_2 = {Ax: A[i][np.newaxis, :], Px: P1[np.random.choice(len(P1), 1)]}
        feed_dict_3 = {Ax: A[i][np.newaxis, :], Px: P2[np.random.choice(len(P2), 1)]}

        loss0=loss_AP_mean.eval(feed_dict_1)
        loss1 = loss_AP_mean.eval(feed_dict_2)
        loss2 = loss_AP_mean.eval(feed_dict_3)

        loss_=[loss0,loss1,loss2]
        label.append(list.index(loss_,min(loss_)))

    # print('label',label,'real',real)

    acc=0
    for i,j in zip(label,real):
        if int(i)-int(j)==0:
            acc+=1

    print('acc',acc/len(label))














