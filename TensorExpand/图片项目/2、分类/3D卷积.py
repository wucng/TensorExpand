# -*- coding: utf-8 -*-

import os
import sys
import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):

    mnist=input_data.read_data_sets(FLAGS.buckets,one_hot=True)

    #模型存储名称
    ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")

    x=tf.placeholder(tf.float32,[None,784])
    y_=tf.placeholder(tf.float32,[None,10])

    input_x=tf.reshape(x,[-1,28,28,1])
    input_x=tf.layers.dense(input_x,256,activation=tf.nn.leaky_relu,kernel_initializer=tf.uniform_unit_scaling_initializer) # [-1,28,28,256]
    input_x=tf.reshape(input_x,[-1,28,28,32,8])

    conv1=tf.layers.conv3d(input_x,32,7,padding='SAME',activation=tf.nn.leaky_relu) # [-1,28,28,32,32]
    pool1=tf.layers.max_pooling3d(conv1,2,2,padding='SAME') # [-1,14,14,16,32]

    conv2 = tf.layers.conv3d(pool1, 64, 3, padding='SAME', activation=tf.nn.leaky_relu)  # [-1,14,14,16,64]
    pool2 = tf.layers.max_pooling3d(conv2, 2, 2, padding='SAME')  # [-1,7,7,8,64]

    conv3 = tf.layers.conv3d(pool2, 64, 3, padding='SAME', activation=tf.nn.leaky_relu)  # [-1,7,7,8,64]
    pool3 = tf.layers.max_pooling3d(conv3, 2, 2, padding='SAME')  # [-1,4,4,4,64]

    conv4 = tf.layers.conv3d(pool3, 32, 3, padding='SAME', activation=tf.nn.leaky_relu)  # [-1,4,4,4,32]
    pool4 = tf.layers.max_pooling3d(conv4, 2, 2, padding='SAME')  # [-1,2,2,2,32]

    conv5 = tf.layers.conv3d(pool4, 10, 3, padding='SAME', activation=tf.nn.leaky_relu)  # [-1,2,2,2,10]
    pool5 = tf.layers.max_pooling3d(conv5, 2, 2, padding='SAME')  # # [-1,1,1,1,10]

    y=tf.reshape(pool5,[-1,10])

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    batch_size=32
    batchs=mnist.train.num_examples // batch_size


    for epoch in range(1):
        for step in range(batchs):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})

            if step%50==0:
                acc=accuracy.eval({x:batch_x,y_:batch_y})
                print('epoch',epoch,'|','step',step,'|','acc',acc)

        test_acc = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
        print('epoch', epoch, '|', 'test_acc', test_acc)

        save_path = saver.save(sess, ckpt_path,global_step=epoch)
        print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 获得buckets路径
    parser.add_argument('--buckets', type=str, default=r'C:\Users\Administrator\Desktop\test\fashion_mnist',
                        help='input data path')
    # 获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='model',
                        help='output model path')
    # FLAGS = parser.parse_args()
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)