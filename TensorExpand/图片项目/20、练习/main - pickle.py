# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from glob import glob
from PIL import Image
import os
import pickle

def load_pickle(pickel_path):
    with open(pickel_path, 'rb') as file:
        return pickle.load(file)

# 设置超参数
num_class=3755
image_size=32
# learning_rate=1e-5
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate=tf.train.polynomial_decay(0.1,global_step,1000,1e-5)
epochs=1
train=1 # 1 train,0 test
batch_size=128
keep=0.8
logdir='./checkpoint/'

x=tf.placeholder(tf.float32,[None,image_size,image_size])
y_=tf.placeholder(tf.int64,[None,])
keep_rate=tf.placeholder(tf.float32)
is_training=tf.placeholder(tf.bool)


# 搭建网络
net=tf.expand_dims(x,-1) # [n,64,64,1]
'''
net=tf.layers.conv2d(net,32,7,padding='same',name='conv1')
net=tf.layers.conv2d(net,64,7,padding='same',name='conv2')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,32,32,64]
'''
net=tf.layers.conv2d(net,64,7,padding='same',name='conv3')
net=tf.layers.conv2d(net,128,7,padding='same',name='conv4')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,16,16,128]

net=tf.layers.conv2d(net,128,7,padding='same',name='conv5')
net=tf.layers.conv2d(net,256,5,padding='same',name='conv6')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,8,8,256]

net=tf.layers.conv2d(net,256,5,padding='same',name='conv7')
net=tf.layers.conv2d(net,512,5,padding='same',name='conv8')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,4,4,512]

net=tf.layers.conv2d(net,512,5,padding='same',name='conv9')
net=tf.layers.conv2d(net,1024,3,padding='same',name='conv10')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,2,2,1024]

net=tf.layers.conv2d(net,1024,3,padding='same',name='conv11')
net=tf.layers.conv2d(net,num_class,3,padding='same',name='conv12')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,1,1,3755]

net=tf.reshape(net,[-1,num_class])
net=slim.dropout(net,keep_rate,is_training=is_training)

prediction=slim.softmax(net) # [n,3755]

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__=="__main__":

    sess = tf.InteractiveSession()
    data = load_pickle('./data/train_0.pickle')
    images=[]
    labels=[]

    for i in range(len(data)):
        if i%2==0:
            images.append(data[i])
        else:
            labels.append(data[i])
    acc_last=0.0
    steps = len(images) // batch_size
    acc_list = []

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    saver=tf.train.Saver(tf.global_variables())

    if not os.path.exists(logdir): os.mkdir(logdir)
    # 验证之前是否已经保存了检查点文件
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    if train == 1:  # 训练
        for epoch in range(epochs):
            start=0;end=0
            index=np.arange(0,len(images))
            np.random.shuffle(index)
            images=np.asarray(images,np.float32)[index]
            labels=np.asarray(labels,np.int64)[index]

            for step in range(steps):
                # 验证之前是否已经保存了检查点文件
                # if step%500==0:
                #     ckpt = tf.train.get_checkpoint_state(logdir)
                #     if ckpt and ckpt.model_checkpoint_path:
                #         saver.restore(sess, ckpt.model_checkpoint_path)

                # Run training steps or whatever
                end=min((len(images),start+batch_size))

                curr_x_train_batch=images[start:end]
                curr_y_train_batch =labels[start:end]
                sess.run([optimizer, global_step],
                         {x: curr_x_train_batch, y_: curr_y_train_batch, keep_rate: keep, is_training: True})
                # optimizer.run({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:keep,is_training:True})
                start=end

                acc = accuracy.eval({x: curr_x_train_batch, y_: curr_y_train_batch, keep_rate: 1., is_training: True})
                if step % 20 == 0:
                    # acc=accuracy.eval({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:1.,is_training:True})
                    loss = cost.eval({x: curr_x_train_batch, y_: curr_y_train_batch, keep_rate: 1., is_training: True})
                    print('step', step, '|', 'acc', acc, '|', 'loss', loss)

                if acc > acc_last:
                    acc_last = acc
                    print('acc_last', acc_last)
                    saver.save(sess, logdir + 'model.ckpt', global_step=step)

    if train == 0:  # 测试
        start = 0;end = 0
        for step in range(steps):
            end = min((len(images), start + batch_size))
            curr_x_test_batch, curr_y_test_batch = images[start:end],labels[start:end]
            if step % 20 == 0:
                acc = accuracy.eval({x: curr_x_test_batch, y_: curr_y_test_batch, keep_rate: 1., is_training: True})
                acc_list.append(acc)
                print('step', step, '|', 'acc', acc)
            start=end
        print('test acc', np.mean(acc_list))