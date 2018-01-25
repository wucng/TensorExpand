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

    # the Variables we need to train
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    batch_size=128
    batchs=mnist.train.num_examples // batch_size


    for epoch in range(2):
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
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
