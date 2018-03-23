# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import pandas as pd
import cv2
import os



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="The number of iterations", type=int, default=1000) # 迭代次数
parser.add_argument("-s", "--steps", help="The number of steps", type=int, default=5000) # 每个迭代执行的步数
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=2) # 每步训练的样本数
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-6) # 学习效率
parser.add_argument("-k", "--keep", help="keep rate", type=float, default=0.7) # dropout out保持率
parser.add_argument("-m", "--mode", help="1 train,0 test", type=int, default=1) # 模式，训练or测试
parser.add_argument("-ld", "--logdir", help="model save path", type=str, default='./checkpoint/') # 模型参数保存位置

args = parser.parse_args()
print("args:",args)


# 数据接口
class Data_info(object):
    def __init__(self,annotations='annotations.csv',classes='classes.csv'):
        self.annotations=annotations
        self.classes=classes
        self.start_index=0
        self.second_index=0
        self.batch_size=args.batch_size

        self.read_csv()

        # self.next_batch()

    def shutffle(self):
        np.random.shuffle(self.data)

    def read_csv(self):
        data_fram = pd.read_csv(self.classes, header=None)
        self.Id = dict(data_fram.values)  # 类名与id对应
        data_fram = None

        data_fram = pd.read_csv(self.annotations, header=None)
        self.data = data_fram.values
        # np.random.shuffle(self.data)
        data_fram = None

    def class2ID(self,classname):
        return self.Id[classname]

    def next_batch(self):
        # self.start_index = 0
        self.second_index = self.start_index + self.batch_size

        if self.second_index > len(self.data):
            self.second_index = len(self.data)

        data1 = self.data[self.start_index:self.second_index]

        self.start_index = self.second_index
        if self.start_index >= len(self.data):
            self.start_index = 0

        # 将每次得到batch_size个数据按行打乱
        index = [i for i in range(len(data1))]  # len(data1)得到的行数
        np.random.shuffle(index)  # 将索引打乱
        data1 = data1[index]

        img_paths = data1[:, 0]  # 得到的是图片的路径，后面的解析成数组

        label = data1[:, 1:]  # 最后一列为类名，得换成对应的id
        last_label = np.asarray(list(map(self.class2ID, label[:, -1])), np.uint8)[:, np.newaxis]
        self.labels = np.hstack((label[:, :-1], last_label)).astype(np.int16)

        # self.load_image(['./JPEGImages/000032.jpg','./JPEGImages/000033.jpg','./JPEGImages/000061.jpg','./JPEGImages/000063.jpg'])
        self.load_image(img_paths)

    # 将边框外的像素都屏蔽掉（对当前的分类与回归都没用）
    def load_image(self,img_paths):
        img=[]
        for i,path in enumerate(img_paths):
            path=os.path.join(os.path.dirname(self.annotations),path)
            img_=cv2.imread(path)
            bbox=self.labels[i,:4]
            # img_mask=np.zeros(img_.shape[:2],np.uint8)
            img_mask=np.zeros_like(img_,np.uint8)
            # cv2.fillPoly(img_mask,[bbox],1)
            cv2.rectangle(img_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 1, -1)
            img_*=img_mask # 屏蔽掉对象框以外的像素
            # img_=cv2.bitwise_and(img_,img_mask)
            img.append(cv2.resize(img_,(500,400))) # 转成400x500

        img=np.asarray(img,np.float16) # [batch_size,h,w,c]

        # 数据归一化
        # self.imgs=(img-np.min(img,0,keepdims=True))/(np.max(img,0,keepdims=True)-np.min(img,0,keepdims=True)) # 0.~1.

        self.imgs=img/255.-0.5 # -0.5~0.5


# 模型接口
class Model(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 400, 500,3])
        self.y_ = tf.placeholder(tf.int64, [None, ]) # 对应class id
        self.y__=tf.placeholder(tf.float32, [None,4]) # 对应边界框 4个坐标值
        self.keep_rate = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.num_class=4+20 # 4个边框点，20个类别（对应one_hot格式）
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate=args.learning_rate

    def Network(self):
        net = tf.layers.conv2d(self.x, 64, 7, 1, 'same', activation=tf.nn.leaky_relu) # [None,400,500,64]
        net = slim.batch_norm(net, is_training=self.is_training)
        net = tf.layers.max_pooling2d(net, 2, 2, 'same')  # [n,200,250,64]

        branch1 = tf.layers.conv2d(net, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,200,250,64]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,200,250,64]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)

        net = tf.nn.leaky_relu(net + branch2) # [n,200,250,64]
        branch1 = tf.layers.conv2d(net, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,200,250,64]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,200,250,64]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)

        net = tf.nn.leaky_relu(net + branch2)
        branch1 = tf.layers.conv2d(net, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,200,250,64]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,200,250,64]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)

        net = tf.nn.leaky_relu(net + branch2)
        net = tf.layers.max_pooling2d(net, 2, 2, 'same')  # [n,100,125,64]

        branch1 = tf.layers.conv2d(net, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        # net=tf.nn.leaky_relu(net+branch2)
        net = tf.layers.conv2d(net, 128, 1, 1, 'same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        net = tf.nn.leaky_relu(net + branch2)  # [n,100,125,128]

        branch1 = tf.layers.conv2d(net, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)  # [n,100,125,128]

        branch1 = tf.layers.conv2d(net, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)  # [n,100,125,128]

        branch1 = tf.layers.conv2d(net, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 128, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,100,125,128]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)  # [n,100,125,128]
        net = tf.layers.max_pooling2d(net, 2, 2, 'same')  # [n,50,63,128]

        branch1 = tf.layers.conv2d(net, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        # net=tf.nn.leaky_relu(net+branch2)
        net = tf.layers.conv2d(net, 256, 1, 1, 'same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        net = tf.nn.leaky_relu(net + branch2)  # [n,4,4,256]

        branch1 = tf.layers.conv2d(net, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)

        branch1 = tf.layers.conv2d(net, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)

        branch1 = tf.layers.conv2d(net, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)

        branch1 = tf.layers.conv2d(net, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)

        branch1 = tf.layers.conv2d(net, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 256, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,50,63,256]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)
        net = tf.layers.max_pooling2d(net, 2, 2, 'same')  # [n,25,32,256]

        branch1 = tf.layers.conv2d(net, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,25,32,512]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,25,32,512]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.layers.conv2d(net, 512, 1, 1, 'same', activation=tf.nn.leaky_relu)  # [n,25,32,512]
        net = tf.nn.leaky_relu(net + branch2)  # [n,25,32,512]
        # net=tf.nn.leaky_relu(net+branch2)

        branch1 = tf.layers.conv2d(net, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,25,32,512]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,25,32,512]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)

        branch1 = tf.layers.conv2d(net, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,25,32,512]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,25,32,512]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)
        net = tf.layers.max_pooling2d(net, 2, 2, 'same')  # [n,13,16,512]

        branch1 = tf.layers.conv2d(net, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,13,16,512]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,13,16,512]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)
        net = tf.layers.max_pooling2d(net, 2, 2, 'valid')  # [n,6,8,512]

        branch1 = tf.layers.conv2d(net, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,6,8,512]
        branch1 = slim.batch_norm(branch1, is_training=self.is_training)
        branch2 = tf.layers.conv2d(branch1, 512, 3, 1, padding='same', activation=tf.nn.leaky_relu)  # [n,6,8,512]
        branch2 = slim.batch_norm(branch2, is_training=self.is_training)
        net = tf.nn.leaky_relu(net + branch2)
        net = tf.layers.max_pooling2d(net, 2, 2, 'valid')  # [n,3,4,512]

        net=tf.reshape(net,[-1,3*4*512])
        fc1 = tf.layers.dense(net, 1024, activation=tf.nn.leaky_relu)
        fc1 = slim.dropout(fc1, self.keep_rate, is_training=self.is_training)

        self.prediction = tf.layers.dense(fc1, self.num_class)

    def loss(self):
        loss1=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction[:,4:], labels=self.y_))
        loss2=tf.losses.mean_squared_error(labels=self.y__,predictions=self.prediction[:,:4])
        self.cost=tf.add(loss1,loss2)/2.

    def train(self):
        self.Network()
        self.loss()
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,self.global_step)

    def evaluation(self):
        correct_pred = tf.equal(tf.argmax(self.prediction[:,4:], 1), self.y_)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



if __name__=="__main__":
    model = Model()
    model.train()
    optimizer = model.optimizer
    # 加载数据
    data = Data_info('./CSV/annotations.csv', './CSV/classes.csv')

    sess = tf.InteractiveSession()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    saver = tf.train.Saver(tf.global_variables())

    if not os.path.exists(args.logdir): os.mkdir(args.logdir)

    for epoch in range(args.epochs):
        data.shutffle() # 每个批次打乱数据
        # 每个epoch重新加载上一个epoch的模型参数训练
        ckpt = tf.train.get_checkpoint_state(args.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for step in range(args.steps):
            data.next_batch()
            batch_x,batch_y=data.imgs, data.labels

            sess.run([optimizer, model.global_step],
                     {model.x: batch_x, model.y_: batch_y[:,-1],model.y__: batch_y[:,:-1],
                      model.keep_rate: args.keep, model.is_training: True})

            if step % 100 == 0:
                model.evaluation()
                acc = model.accuracy.eval(
                    {model.x: batch_x, model.y_: batch_y[:, -1], model.y__: batch_y[:, :-1],
                     model.keep_rate: 1., model.is_training: True})
                loss = model.cost.eval({model.x: batch_x, model.y_: batch_y[:,-1],model.y__: batch_y[:,:-1],
                      model.keep_rate: 1., model.is_training: True})
                print('epoch', epoch, '|', 'step', step, '|', 'acc', acc, '|', 'loss', loss)

        saver.save(sess, args.logdir + 'model.ckpt')

    sess.close()