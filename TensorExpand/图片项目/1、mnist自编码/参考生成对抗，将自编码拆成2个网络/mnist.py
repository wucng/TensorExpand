#! /usr/bin/python3
# -*- coding: utf8 -*-

'''
参考GAN，将自编码网络拆分成编码器与解码器 分别训练（参考GAN的训练方式）
自编码 实现解压缩（encode+decode）
encode实现分类
decode实现生产

注：mnist数据 只是0,1两个数字（并不是0~1）组成的数据，输入得随机生成数要满足这个规律

参考模型流程图 使用卷积网络 对mnist进行分类
并同时训练非监督+监督模型
包含非监督方法（自编码）
监督分类
'''

import os
import sys
import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import slim
import numpy as np
import os
import scipy.misc
import scipy

FLAGS = None

def main(_):
    mnist = input_data.read_data_sets(FLAGS.buckets, one_hot=True)

    # 模型存储名称
    ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")

    # x = tf.placeholder(tf.float32, [None, 784])
    # y_ = tf.placeholder(tf.float32, [None, 10])

    def save_images(images, size, image_path):
        return imsave(inverse_transform(images), size, image_path)

    def imsave(images, size, path):
        return scipy.misc.imsave(path, merge(images, size))
        # return scipy.misc.imsave(path, images)

    def inverse_transform(images):
        return (images + 1.) / 2.

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image

        return img


    initializer = tf.truncated_normal_initializer(stddev=0.02)

    def generator(z):
        '''
        The generator takes a vector of random numbers and transforms it into a 32x32 image.
        :param z: a vector of random numbers
        :return: a 32x32 image
        '''
        with tf.variable_scope('G'):
            zP=slim.layers.fully_connected(z,4*4*256,activation_fn=tf.nn.leaky_relu,normalizer_fn=slim.batch_norm,
                                           scope='g_project', weights_initializer=initializer) # [n,4*4*256]

            zCon = tf.reshape(zP, [-1, 4, 4, 256])  # [n,4,4,256]

            gen1 = slim.convolution2d_transpose(zCon, num_outputs=64,
                                                kernel_size=[5, 5], stride=[2, 2],padding="SAME",
                                                normalizer_fn=slim.batch_norm,activation_fn=tf.nn.leaky_relu, scope='g_conv1',
                                                weights_initializer=initializer)  # [n,8,8,64] 采样方式SAME h*stride

            gen2 = slim.convolution2d_transpose( \
                gen1, num_outputs=32, kernel_size=[5, 5], stride=[2, 2], \
                padding="SAME", normalizer_fn=slim.batch_norm, \
                activation_fn=tf.nn.leaky_relu, scope='g_conv2',
                weights_initializer=initializer)  # [n,16,16,32] 采样方式SAME h*stride

            gen3 = slim.convolution2d_transpose( \
                gen2, num_outputs=16, kernel_size=[5, 5], stride=[2, 2], \
                padding="SAME", normalizer_fn=slim.batch_norm, \
                activation_fn=tf.nn.leaky_relu, scope='g_conv3',
                weights_initializer=initializer)  # [n,32,32,16] 采样方式SAME h*stride

            g_out = slim.convolution2d_transpose( \
                gen3, num_outputs=1, kernel_size=[32, 32], padding="SAME", \
                biases_initializer=None, activation_fn=tf.nn.tanh, \
                scope='g_out', weights_initializer=initializer)  # [n,32,32,1] 这里stride默认为1

        return g_out

    def discriminator(bottom, reuse=False):
        '''
        The discriminator network takes as input a 32x32 image and 
        transforms it into a single valued probability of being generated from real-world data.
        :param bottom: a 32x32 image
        :param reuse: 
        :return: a single valued (0 or 1)
        '''
        with tf.variable_scope('D', reuse=reuse):
            dis1 = slim.convolution2d(bottom, 16, [4, 4], stride=[2, 2], padding="SAME", \
                                      biases_initializer=None, activation_fn=tf.nn.leaky_relu, \
                                      reuse=reuse, scope='d_conv1', weights_initializer=initializer)  # [n,16,16,16]

            dis2 = slim.convolution2d(dis1, 32, [4, 4], stride=[2, 2], padding="SAME", \
                                      normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu, \
                                      reuse=reuse, scope='d_conv2', weights_initializer=initializer)  # [n,8,8,32]

            dis3 = slim.convolution2d(dis2, 64, [4, 4], stride=[2, 2], padding="SAME", \
                                      normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu, \
                                      reuse=reuse, scope='d_conv3', weights_initializer=initializer)  # [n,4,4,64]

            d_out = slim.fully_connected(slim.flatten(dis3), 1, activation_fn=tf.nn.sigmoid, \
                                         reuse=reuse, scope='d_out', weights_initializer=initializer)  # [n,1]

        return d_out

    def encode(X,reuse):

        with tf.variable_scope('D', reuse=reuse):
            dis1 = slim.convolution2d(X, 16, [4, 4], stride=[2, 2], padding="SAME", \
                                      biases_initializer=None, activation_fn=tf.nn.leaky_relu, \
                                      reuse=reuse, scope='d_conv1', weights_initializer=initializer)  # [n,16,16,16]

            dis2 = slim.convolution2d(dis1, 32, [4, 4], stride=[2, 2], padding="SAME", \
                                      normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu, \
                                      reuse=reuse, scope='d_conv2', weights_initializer=initializer)  # [n,8,8,32]

            dis3 = slim.convolution2d(dis2, 64, [4, 4], stride=[2, 2], padding="SAME", \
                                      normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu, \
                                      reuse=reuse, scope='d_conv3', weights_initializer=initializer)  # [n,4,4,64]
        return dis3 # [n,4,4,64]
    def decode(X,reuse):
        with tf.variable_scope('G'):
            gen1 = slim.convolution2d_transpose(X, num_outputs=64,
                                                kernel_size=[5, 5], stride=[2, 2], padding="SAME",
                                                normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu,reuse=reuse,
                                                scope='g_conv1',
                                                weights_initializer=initializer)  # [n,8,8,64] 采样方式SAME h*stride

            gen2 = slim.convolution2d_transpose( \
                gen1, num_outputs=32, kernel_size=[5, 5], stride=[2, 2], \
                padding="SAME", normalizer_fn=slim.batch_norm, \
                activation_fn=tf.nn.leaky_relu, scope='g_conv2',reuse=reuse,
                weights_initializer=initializer)  # [n,16,16,32] 采样方式SAME h*stride

            gen3 = slim.convolution2d_transpose( \
                gen2, num_outputs=16, kernel_size=[5, 5], stride=[2, 2], \
                padding="SAME", normalizer_fn=slim.batch_norm, \
                activation_fn=tf.nn.leaky_relu, scope='g_conv3',reuse=reuse,
                weights_initializer=initializer)  # [n,32,32,16] 采样方式SAME h*stride

            g_out = slim.convolution2d_transpose( \
                gen3, num_outputs=1, kernel_size=[32, 32], padding="SAME", \
                biases_initializer=None, activation_fn=tf.nn.tanh, \
                reuse=reuse,scope='g_out', weights_initializer=initializer)  # [n,32,32,1] 这里stride默认为1

        return g_out

    # Connecting them together
    tf.reset_default_graph()

    # z_size = 100  # Size of z vector used for generator.

    # These two placeholders are used for input into the generator and discriminator, respectively.
    # z_in = tf.placeholder(shape=[None, z_size], dtype=tf.float32)  # Random vector
    # real_in = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)  # Real images
    real_in = tf.placeholder('float', [None, 32,32,1], name='input_x')
    z_in = tf.placeholder('float', [None, 4, 4, 64], name='input_x_')


    # Gz = generator(z_in)  # Generates images from random z vectors [n,32,32,1]
    # Dx = discriminator(real_in)  # Produces probabilities for real images [n,1]
    # Dg = discriminator(Gz, reuse=True)  # Produces probabilities for generator images [n,1]

    # Ex=encode(real_in,False) # [-1,4,4,64]
    # Dx=decode(Ex,False) # [-1,32,32,1]
    # Dx_=decode(z_in,True) # [-1,32,32,1]

    Dx_ = decode(z_in, False)  # [-1,32,32,1]
    Ex = encode(real_in, False)  # [-1,4,4,64]
    Dx = decode(Ex, True)  # [-1,32,32,1]




    # These functions together define the optimization objective of the GAN.
    # d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))  # This optimizes the discriminator.
    # g_loss = -tf.reduce_mean(tf.log(Dg))  # This optimizes the generator.
    d_loss=tf.losses.mean_squared_error(labels=real_in,predictions=Dx)
    g_loss=tf.losses.mean_squared_error(labels=real_in,predictions=Dx_)
    '''
    tvars = tf.trainable_variables()
    d_params = [v for v in tvars if v.name.startswith('D/')]
    g_params = [v for v in tvars if v.name.startswith('G/')]
    # The below code is responsible for applying gradient descent to update the GAN.
    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    # d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #Only update the weights for the discriminator network.
    # g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #Only update the weights for the generator network.
    d_grads = trainerD.compute_gradients(d_loss, d_params)  # Only update the weights for the discriminator network.
    g_grads = trainerG.compute_gradients(g_loss, g_params)  # Only update the weights for the generator network.

    update_D = trainerD.apply_gradients(d_grads)
    update_G = trainerG.apply_gradients(g_grads)

    # '''
    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

    update_D = trainerD.minimize(d_loss)
    update_G = trainerG.minimize(g_loss)
    # '''
    # Training the network
    batch_size = 128  # Size of image batch to apply at each iteration.
    # iterations = 500000  # Total number of iterations to use.
    sample_directory = './figs'  # Directory to save sample images from generator in.
    model_directory = FLAGS.checkpointDir  # Directory to save trained model to.
    if not os.path.exists(sample_directory): os.makedirs(sample_directory)
    if not os.path.exists(model_directory): os.makedirs(model_directory)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(20):
            for step in range(mnist.train.images.shape[0]//batch_size):
                zs = np.random.uniform(-1.0, 1.0, size=[batch_size, 4,4,64]).astype(
                    np.float32)  # Generate a random z batch
                xs, _ = mnist.train.next_batch(batch_size)  # Draw a sample batch from MNIST dataset.
                xs = (np.reshape(xs, [batch_size, 28, 28, 1]) - 0.5) * 2.0  # Transform it to be between -1 and 1
                xs = np.lib.pad(xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
                                constant_values=(-1, -1))  # Pad the images so the are 32x32

                _,d_Loss=sess.run([update_D,d_loss],feed_dict={real_in:xs})
                _,g_Loss=sess.run([update_G,g_loss],feed_dict={z_in:zs,real_in:xs})

                if step % 50 == 0:
                    print('epoch',epoch,'|','step',step,'|','d_loss',d_Loss,'|','g_loss',g_Loss)

            saver.save(sess,os.path.join(model_directory,'model.ckpt'),global_step=epoch)

            z2 = np.random.uniform(-1.0, 1.0, size=[batch_size, 4,4,64]).astype(
                np.float32)  # Generate another z batch
            newZ = sess.run(Dx_, feed_dict={z_in: z2})  # Use new z to get sample images from generator.
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            save_images(np.reshape(newZ[0:36], [36, 32, 32]), [6, 6],
                        sample_directory + '/fig' + str(epoch) + '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 获得buckets路径
    parser.add_argument('--buckets', type=str, default='../MNIST_data',
                        help='input data path')
    # 获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='model',
                        help='output model path')
    # FLAGS = parser.parse_args()
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
