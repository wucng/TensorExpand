# -*- coding:utf-8 -*-

'''
根据配置文件，编写相应的神经网络
'''

import tensorflow as tf
# from tensorflow.contrib import slim
from Config_file_analyze import Config_file_analyze
from Data_interface import Data_interface
import os
import argparse
import numpy as np

class Net(object):
    def __init__(self,conf):
        self.network=conf.network
        self.net=conf.net
        # self.org =conf.org

        self.num_class=10
        self.tf_net()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost,self.global_step)
        # tf.train.MomentumOptimizer(self.learning_rate,self.momentum)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._net,-1), self.y_), tf.float32))

    def tf_net(self):
        '''根据配置文件，编写相应的网络，使用tensorflow'''
        for net_structure in self.network:
            if 'net' in net_structure:
                cont=self.net[net_structure]
                self.batch=int(cont['batch'])
                self.height=int(cont['height'])
                self.width=int(cont['width'])
                self.channels = int(cont['channels'])

                self.policy=str(cont['policy'])
                self.power=int(cont['power'])
                self.momentum = float(cont['momentum'])
                self.decay=float(cont['decay'])
                # self.max_crop = float(cont['max_crop'])
                self.learning_rate = float(cont['learning_rate'])
                self.max_batches=int(cont['max_batches'])

                self.global_step=tf.Variable(0,trainable=False)

                if 'poly' in self.policy:
                    self.learning_rate=tf.train.polynomial_decay(self.learning_rate,
                                                                 self.global_step,self.max_batches,1e-7,self.power)

                self.steps=int(cont['steps'])

                self.train=int(cont['train']) # 1 训练 ，0 测试

                self.epochs=int(cont['epochs'])

                cont=None

                self.x=tf.placeholder(tf.float32,[None,self.height,self.width,self.channels])
                self.y_=tf.placeholder(tf.int64,[None,])
                self.istrain=tf.placeholder(tf.bool)
                self.dropout=tf.placeholder(tf.float32)

                self._net = self.x
                print(self._net.shape)


            if 'convolutional' in net_structure:
                cont = self.net[net_structure]
                self.filters = int(cont['filters'])
                self.size = int(cont['size'])
                self.stride = int(cont['stride'])
                self.pad=cont['pad']
                self.activation = cont['activation']
                # self.batch_normalize=int(cont['batch_normalize'])

                self._net=tf.layers.conv2d(self._net,self.filters,self.size,self.stride,self.pad,name=net_structure)
                if 'leaky' in self.activation:
                    self._net=tf.nn.leaky_relu(self._net)
                else:
                    self._net = tf.nn.relu(self._net)

                # if self.batch_normalize:
                #     self._net=slim.batch_norm(self._net,is_training=self.istrain)
                cont = None
                print(self._net.shape)

            if 'maxpool' in net_structure:
                cont = self.net[net_structure]
                self.size = int(cont['size'])
                self.stride = int(cont['stride'])

                self._net=tf.layers.max_pooling2d(self._net,self.size,self.stride,'same',name=net_structure)
                cont = None
                print(self._net.shape)

            if 'connected' in net_structure:
                cont = self.net[net_structure]
                self.output = int(cont['output'])
                self.activation = cont['activation']

                if len(self._net.shape)>2:
                    self._net=slim.flatten(self._net)

                self._net=tf.layers.dense(self._net,self.output,name=net_structure)
                if 'relu' in self.activation:
                    self._net=tf.nn.relu(self._net)
                if 'linear' in self.activation:
                    pass # 不使用激活函数

                cont = None
                print(self._net.shape)

            if 'avgpool' in net_structure:
                size=self._net.shape[1:3]
                self._net=tf.layers.average_pooling2d(self._net,size,1,'valid',name=net_structure)
                # self._net=tf.squeeze(self._net)
                self._net=tf.reshape(self._net,[-1,self.num_class])
                print(self._net.shape)

            if 'dropout' in net_structure:
                cont = self.net[net_structure]
                self.probability=float(cont['probability'])
                self._net = tf.nn.dropout(self._net, self.dropout,name=net_structure)
                cont = None

            if 'softmax' in net_structure:
                self._net=tf.nn.softmax(self._net,name=net_structure)

            if 'cost' in net_structure:
                self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_,logits=self._net,name=net_structure))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Configuration file path", type=str, default='cfg/cifar.cfg')  # 配置文件路径
    parser.add_argument("--data_path", help="data file path", type=str, default='cfg/cifar.data')  # 数据文件路径
    args = parser.parse_args()
    print("args:", args)

    conf = Config_file_analyze(args.config_path)
    net=Net(conf)
    data = Data_interface(args.data_path,net.height,net.width,net.batch)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # 验证之前是否已经保存了检查点文件
        if not os.path.exists(data.backup):os.makedirs(data.backup)
        ckpt = tf.train.get_checkpoint_state(data.backup)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if net.train:
            for epoch in range(net.epochs):
                data.data_shuffle(data.train_data)

                for step in range(net.steps):
                    data.data_next_batch(data.train_data)
                    imgs=data.batch_data[0]
                    feed_dict={net.x:imgs,net.y_:data.batch_data[1],net.istrain:True,net.dropout:net.probability}
                    sess.run([net.train_op,net.global_step],feed_dict)

                    if step%50==0:
                        feed_dict[net.dropout]=1.
                        acc=net.accuracy.eval(feed_dict)
                        loss=net.cost.eval(feed_dict)
                        print('epoch',epoch,'|','step',step,'|','acc',acc,'|','loss',loss)

                save_path=saver.save(sess,os.path.join(data.backup,'model.ckpt'))
                print('save model to {}'.format(save_path))

        else: # 验证
            accs=[]
            for step in range(10):
                data.data_next_batch(data.valid_data)
                imgs = data.batch_data[0]
                feed_dict = {net.x: imgs, net.y_: data.batch_data[1], net.istrain: True,
                             net.dropout:1.0}

                acc = net.accuracy.eval(feed_dict)
                accs.append(acc)
                print('step',step,'|','acc',acc)

            print('acc_mean',np.mean(accs))