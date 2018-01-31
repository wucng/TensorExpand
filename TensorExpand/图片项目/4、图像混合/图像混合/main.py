#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import argparse
# import PIL.Image
# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
#获得buckets路径
parser.add_argument('--buckets', type=str, default='./MNIST_data',
                    help='input data path')
#获得checkpoint路径
parser.add_argument('--checkpointDir', type=str, default='model',
                    help='output model path')
FLAGS, _ = parser.parse_known_args()


mnist = input_data.read_data_sets(FLAGS.buckets, one_hot=False)
ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")

# 定义待训练的神经网络
def convolutional_neural_network(X,reuse):
        X=tf.reshape(X,[-1,28,28,1])

        # encode
        conv1=tf.layers.conv2d(X,32,3,padding='SAME',activation=tf.nn.relu,reuse=reuse,name='conv1') # [-1,28,28,32]
        pool1=tf.layers.max_pooling2d(conv1,2,2,'SAME',name='pool1') # [-1,14,14,32]

        conv2=tf.layers.conv2d(pool1,64,3,padding='SAME',activation=tf.nn.relu,reuse=reuse,name='conv2') # [-1,14,14,64]
        encode = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME',name='encode') # [-1,7,7,64]

        # decode
        conv3 = tf.layers.conv2d_transpose(encode, 32, 7, 2, padding='SAME', activation=tf.nn.relu, reuse=reuse,
                                           name='conv_trans')  # [-1,14,14,32]
        decode = tf.layers.conv2d_transpose(conv3, 1, 5, 2, padding='SAME', activation=tf.nn.relu, reuse=reuse,
                                            name='decode')  # [-1,28,28,1]
        return encode, decode

def decode(X,reuse):
    # [-1, 7, 7, 64*2]
    # decode
    # decode
    conv1 = tf.layers.conv2d_transpose(X, 64, 7, 1, padding='SAME', activation=tf.nn.relu, reuse=reuse,
                                       name='conv_trans1')  # [-1,7,7,64]

    conv2 = tf.layers.conv2d_transpose(conv1, 32, 7, 2, padding='SAME', activation=tf.nn.relu, reuse=reuse,
                                       name='conv_trans2')  # [-1,14,14,32]
    decode_ = tf.layers.conv2d_transpose(conv2, 1, 5, 2, padding='SAME', activation=tf.nn.relu, reuse=reuse,
                                        name='output')  # [-1,28,28,1]

    return decode_

def train_neural_network():
    batch_size=128
    X1=tf.placeholder(tf.float32,[None,28,28,1])
    X2=tf.placeholder(tf.float32,[None,7,7,64*2])
    X3 = tf.placeholder(tf.float32, [None, 28, 28, 1])
    global reuse
    reuse=False
    encode_, decode_=convolutional_neural_network(X1,reuse=reuse)

    output=decode(X2, reuse)

    # 自编码loss
    loss1=tf.losses.mean_squared_error(labels=X1,predictions=decode_)
    train_op1=tf.train.AdamOptimizer().minimize(loss1)
    # decode loss
    loss2=tf.losses.mean_squared_error(labels=X3,predictions=output)
    train_op2 = tf.train.AdamOptimizer().minimize(loss2)

    total_batches = mnist.train.images.shape[0] // batch_size
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpointDir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # 同时训练2个模型
        for epoch in range(1):
            epoch_costs1 = np.empty(0)  # 空数组
            epoch_costs2 = np.empty(0)
            for step in range(20):#(total_batches):
                if (epoch+step)!=0:reuse=True

                # -----训练模型1------------#
                x, _ = mnist.train.next_batch(batch_size)
                x=np.reshape(x,[-1,28,28,1])

                _, loss1_ = sess.run([train_op1, loss1], feed_dict={X1: x})
                epoch_costs1=np.append(epoch_costs1, loss1_)


                # ------训练模型2-----------#
                encode__=sess.run(encode_,{X1:x}) # [128,7,7,64]

                encode_input=np.concatenate((encode__[0:-1],encode__[1:]),axis=-1) # [127,7,7,64*2]

                dst=cv2.addWeighted(x[0:-1],0.7,x[1:],0.3,0) # [127,28,28,1]

                _, loss2_ = sess.run([train_op2, loss2], feed_dict={X2: encode_input,X3:dst})
                epoch_costs2=np.append(epoch_costs2, loss2_)

                if step%10==0:
                    print("Epoch: ", epoch, '|', "step", step, " Loss1: ", loss1_, '|', 'Loss2: ',loss2_)

            print("Epoch: ", epoch, '|', "step",step," Loss1: ", np.mean(epoch_costs1), '|', 'Loss2: ',
                  np.mean(epoch_costs2))

            save_path = saver.save(sess, ckpt_path, global_step=epoch)
            print("Model saved in file: %s" % save_path)
            cv2.imshow('1',x[0])
            cv2.imshow('2',x[1])
            cv2.imshow('12',output.eval({X2: encode_input,X3:dst})[0])
            cv2.waitKey(0)

if __name__=="__main__":
    train_neural_network()