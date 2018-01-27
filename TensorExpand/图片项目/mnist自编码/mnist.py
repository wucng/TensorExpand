#! /usr/bin/python3
# -*- coding: utf8 -*-

'''
自编码 实现解压缩（encode+decode）
encode实现分类
decode实现生产

注：mnist数据 只是0,1两个数字（并不是0~1）组成的数据，输入得随机生成数要满足这个规律

参考模型流程图 使用卷积网络 对mnist进行分类
并同时训练非监督+监督模型
包含非监督方法（自编码）
监督分类
'''

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
# import cv2
import os
import sys
import argparse
# from PIL import Image
# from scipy import ndimage
import scipy.misc
import glob

# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(1000)

parser = argparse.ArgumentParser()
#获得buckets路径
parser.add_argument('--buckets', type=str, default='./MNIST_data',
                    help='input data path')

parser.add_argument('--epochs', type=int, default=50,
                    help='epochs')

parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')

parser.add_argument('--n_class', type=int, default=10,
                    help='num class')

parser.add_argument('--train', type=int, default=0,
                    help='1 train -1 inference 0 classes')

#获得checkpoint路径
parser.add_argument('--checkpointDir', type=str, default='model',
                    help='output model path')
FLAGS, _ = parser.parse_known_args()



mnist = input_data.read_data_sets(FLAGS.buckets, one_hot=True)

n_output_layer = FLAGS.n_class

# 定义待训练的神经网络
def convolutional_neural_network(X,X_,reuse,flag=True):
    if flag:
        X=tf.reshape(X,[-1,28,28,1])

        # encode
        conv1=tf.layers.conv2d(X,32,3,padding='SAME',activation=tf.nn.relu,reuse=reuse,name='conv1') # [-1,28,28,32]
        pool1=tf.layers.max_pooling2d(conv1,2,2,'SAME',name='pool1') # [-1,14,14,32]

        conv2=tf.layers.conv2d(pool1,64,3,padding='SAME',activation=tf.nn.relu,reuse=reuse,name='conv2') # [-1,14,14,64]
        encode = tf.layers.max_pooling2d(conv2, 2, 2, 'SAME',name='encode') # [-1,7,7,64]

        # decode
        conv3 = tf.layers.conv2d_transpose(encode, 32, 8, 1, padding='valid', activation=tf.nn.relu, reuse=reuse,
                                           name='conv_trans')  # [-1,14,14,32]
        decode = tf.layers.conv2d_transpose(conv3, 1, 15, 1, padding='valid', activation=tf.nn.relu, reuse=reuse,
                                            name='decode')  # [-1,28,28,1]
        return encode, decode
    else:
        # X_ # [-1,7,7,64]
        # decode
        conv3 = tf.layers.conv2d_transpose(X_, 32, 8, 1, padding='valid', activation=tf.nn.relu, reuse=reuse,
                                           name='conv_trans')  # [-1,14,14,32]
        decode = tf.layers.conv2d_transpose(conv3, 1, 15, 1, padding='valid', activation=tf.nn.relu, reuse=reuse,
                                            name='decode')  # [-1,28,28,1]
        return decode


# 每次使用100条数据进行训练
batch_size = FLAGS.batch_size
epochs = FLAGS.epochs
ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")

# 使用数据训练神经网络
def train_neural_network(X, X_):
    encode, decode = convolutional_neural_network(X, None, False)  # 自编码器 （encode+decode）
    decode_ = convolutional_neural_network(None, X_, True, False)  # 生成器 （decode）
    # 非监督模型1
    un_cost = tf.losses.mean_squared_error(labels=tf.reshape(X, [-1, 28, 28, 1]), predictions=decode)
    un_optimizer = tf.train.AdamOptimizer().minimize(un_cost)
    # 非监督模型2
    un_cost_ = tf.losses.mean_squared_error(labels=tf.reshape(X, [-1, 28, 28, 1]), predictions=decode_)
    un_optimizer_ = tf.train.AdamOptimizer().minimize(un_cost_)

    # correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    total_batches = mnist.train.images.shape[0] // batch_size

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpointDir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # 同时训练2个非监督模型
        for epoch in range(epochs):
            un_epoch_costs = np.empty(0) # 空数组
            un_epoch_costs_ = np.empty(0)
            for step in range(total_batches):
                x, _ = mnist.train.next_batch(batch_size)
                # x_=np.random.random([batch_size,7,7,64])
                x_=np.random.randint(0,2,[batch_size,7,7,64]).astype(np.float32) # 得到的随机数只有0和1

                _, un_c = sess.run([un_optimizer, un_cost], feed_dict={X: x,X_:x_})
                un_epoch_costs = np.append(un_epoch_costs, un_c)

                _, un_c_ = sess.run([un_optimizer_, un_cost_], feed_dict={X: x,X_:x_})
                un_epoch_costs_ = np.append(un_epoch_costs_, un_c_)

            print("Epoch: ", epoch,'|', " un_Loss: ", np.mean(un_epoch_costs),'|','un_Loss_: ', np.mean(un_epoch_costs_))

            save_path = saver.save(sess, ckpt_path, global_step=epoch)
            print("Model saved in file: %s" % save_path)

        print("------------------------------------------------------------------")

        # 生成mnist
        # decode_images=sess.run(decode_,feed_dict={X_:np.random.random([1,7,7,64])})
        # cv2.imwrite('mnist.jpg',decode_images[0])
        # Image.fromarray(decode_images[0]).save(os.path.join(FLAGS.checkpointDir,'mnist.jpg'))
        # 输出pb文件
        '''
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["input_x","input_x_","conv2","decode"])
        with tf.gfile.FastGFile(os.path.join(FLAGS.checkpointDir, 'expert-graph.pb'), mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        '''

# 压缩 解码 生成
def inference(X,X_):
    encode, decode = convolutional_neural_network(X, None, False)  # 自编码器 （encode+decode）
    decode_ = convolutional_neural_network(None, X_, True, False)  # 生成器 （decode）

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # saver.restore(sess, os.path.join(FLAGS.checkpointDir,'model.ckpt-49'))
        ckpt_path_1=glob.glob(os.path.join(FLAGS.checkpointDir,'model.ckpt*'))[-1][:-5]
        saver.restore(sess, ckpt_path_1)
        # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpointDir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        #     sess.run(tf.global_variables_initializer())

        # 生成mnist
        decode_images=sess.run(decode_,feed_dict={X_:np.random.randint(0,2,[10,7,7,64]).astype(np.float32)}) # 0和1的随机数
        # Image.fromarray(decode_images[0].reshape([28,28])).save(os.path.join(FLAGS.checkpointDir,'mnist.jpg'))
        # cv2.imwrite(os.path.join(FLAGS.checkpointDir,'mnist.jpg'), decode_images[0])
        [scipy.misc.imsave(os.path.join(FLAGS.checkpointDir,'mnist_'+str(i)+'.jpg'),
                           decode_images[i].reshape([28,28])) for i in range(10)]

        # 压缩
        encode_image=sess.run(encode,feed_dict={X:mnist.test.images[0:10].reshape([-1,28*28*1])})

        # 解压
        decode_image=sess.run(decode_,feed_dict={X_:encode_image})
        # Image.fromarray(decode_image[0].reshape([28,28])).save(os.path.join(FLAGS.checkpointDir, 'mnist2.jpg'))
        # cv2.imwrite(os.path.join(FLAGS.checkpointDir, 'mnist2.jpg'), decode_image[0])
        [scipy.misc.imsave(os.path.join(FLAGS.checkpointDir, 'mnist2_'+str(i)+'.jpg'),
                           decode_image[i].reshape([28,28])) for i in range(10)]

# 分类
def classification(X):
    encode, _ = convolutional_neural_network(X, None, False)  # 自编码器 （encode+decode）
    # 1、先计算所有图片得到的encode 如：0 encode_0，... 9 encode_9 （各类取100张计算其平均值）
    # 2、计算新传入的图片（未知类别）的encode_new
    # 3、比较encode_new与encode_0~encode_9的距离，距离越小越接近该类别，把其归为该类别，从而实现分类
    image_0=[]
    image_1 = []
    image_2 = []
    image_3 = []
    image_4 = []
    image_5 = []
    image_6 = []
    image_7 = []
    image_8 = []
    image_9 = []
    img_x=mnist.train.images
    lable_y=mnist.train.labels
    for label,img in zip(lable_y,img_x):
        if np.argmax(label,0)==0 and len(image_0)<=100:
            image_0.append(img)
        if np.argmax(label,0)==1 and len(image_1)<=100:
            image_1.append(img)
        if np.argmax(label,0)==2 and len(image_2)<=100:
            image_2.append(img)
        if np.argmax(label,0)==3 and len(image_3)<=100:
            image_3.append(img)
        if np.argmax(label,0)==4 and len(image_4)<=100:
            image_4.append(img)
        if np.argmax(label,0)==5 and len(image_5)<=100:
            image_5.append(img)
        if np.argmax(label,0)==6 and len(image_6)<=100:
            image_6.append(img)
        if np.argmax(label,0)==7 and len(image_7)<=100:
            image_7.append(img)
        if np.argmax(label,0)==8 and len(image_8)<=100:
            image_8.append(img)
        if np.argmax(label,0)==9 and len(image_9)<=100:
            image_9.append(img)

        if len(image_0)+len(image_1)+len(image_2)+len(image_3)+len(image_4)+len(image_5)+ \
            len(image_6)+len(image_7)+len(image_8)+len(image_9) >=1000:\
            break

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # saver.restore(sess, os.path.join(FLAGS.checkpointDir,'model.ckpt-49'))
        ckpt_path_1 = glob.glob(os.path.join(FLAGS.checkpointDir, 'model.ckpt*'))[-1][:-5]
        saver.restore(sess, ckpt_path_1)

        encode_0=[encode.eval(feed_dict={X:i[np.newaxis,:]}) for i in image_0]
        encode_0=np.array(encode_0).mean(0) # [1,7,7,64]

        encode_1 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_1]
        encode_1 = np.array(encode_1).mean(0)  # [1,7,7,64]

        encode_2 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_2]
        encode_2 = np.array(encode_2).mean(0)  # [1,7,7,64]

        encode_3 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_3]
        encode_3 = np.array(encode_3).mean(0)  # [1,7,7,64]

        encode_4 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_4]
        encode_4 = np.array(encode_4).mean(0)  # [1,7,7,64]

        encode_5 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_5]
        encode_5 = np.array(encode_5).mean(0)  # [1,7,7,64]

        encode_6 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_6]
        encode_6 = np.array(encode_6).mean(0)  # [1,7,7,64]

        encode_7 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_7]
        encode_7 = np.array(encode_7).mean(0)  # [1,7,7,64]

        encode_8 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_8]
        encode_8 = np.array(encode_8).mean(0)  # [1,7,7,64]

        encode_9 = [encode.eval(feed_dict={X: i[np.newaxis, :]}) for i in image_9]
        encode_9 = np.array(encode_9).mean(0)  # [1,7,7,64]

        encode_list=[encode_0,encode_1,encode_2,encode_3,encode_4,encode_5,encode_6,encode_7,encode_8,encode_9]

        # ---------------------------#
        encode_new=encode.eval({X:mnist.test.images[:1000]}) # [10,7,7,64]
        pred=[]
        pred_all=[]
        for i in encode_new:
            for j in encode_list:
                pred_all.append(sess.run(tf.losses.mean_squared_error(labels=i[np.newaxis,:],predictions=j)))
            pred.append(np.array(pred_all).argmin(0))
            pred_all=[]

        correct_prediction = tf.equal(np.array(pred), tf.argmax(mnist.test.labels[:1000], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print('acc',sess.run(accuracy))

        # print('pred',pred,'\n','real',np.argmax(mnist.test.labels[:100],-1))

def main(_):
    X = tf.placeholder('float', [None, 28 * 28 * 1],name='input_x')
    X_ = tf.placeholder('float', [None, 7, 7, 64],name='input_x_')
    if FLAGS.train==1:
        train_neural_network(X, X_)
    elif FLAGS.train==-1:
        inference(X, X_)
    elif FLAGS.train==0:
        classification(X)
    else:
        print('1 train -1 inference 0 classes')
        exit(-1)

if __name__ == '__main__':
    tf.app.run(main=main)