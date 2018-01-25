# -*- coding: utf-8 -*-

import os
import sys
import argparse

import tensorflow as tf

FLAGS = None

def read_image(file_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label

def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    capacity = 3 * batch_size
    image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
    one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
    return image_batch, one_hot_labels

def main(_):
    #训练数据集
    train_file_path = os.path.join(FLAGS.buckets, "train.tfrecords")
    #测试数据集
    test_file_path = os.path.join(FLAGS.buckets, "test.tfrecords")
    #模型存储名称
    ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")

    train_image_filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(train_file_path))
    train_images, train_labels = read_image_batch(train_image_filename_queue, 100)

    test_image_filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(test_file_path))
    test_images, test_labels = read_image_batch(test_image_filename_queue, 100)

    # the Variables we need to train
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    x = tf.reshape(train_images, [-1, 784])
    y = tf.matmul(x, W) + b
    y_ = tf.to_float(train_labels)

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    x_test = tf.reshape(test_images, [-1, 784])
    y_pred = tf.matmul(x_test, W) + b
    y_test = tf.to_float(test_labels)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # start queue runner
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Train and Test 
    for i in range(100):
        sess.run(train_step)
        if ((i + 1) % 10 == 0):
            print("step:", i + 1, "accuracy:", sess.run(accuracy))

    print("accuracy: " , sess.run(accuracy))
    save_path = saver.save(sess, ckpt_path)
    print("Model saved in file: %s" % save_path)

    # stop queue runner
    coord.request_stop()
    coord.join(threads)

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

