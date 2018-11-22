#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from itertools import groupby
# from collections import defaultdict
# from PIL import Image
# import numpy as np

"""
tfrecord to numpy for train
http://blog.csdn.net/wc781708249/article/details/78013275
"""

# Load Images
def load_images_from_tfrecord(tfrecord_file,h=32,w=32,c=3,batch_size=32):
    '''
    tfrecord to numpy for train
    :param tfrecord_file: tfrecord文件路径，如："output/training-images/*.tfrecords"
    :param h: 样本高
    :param w: 样本宽
    :param c: 样本通道数
    :param batch_size: 每批次训练样本数
    :return: 样本，标签
    '''
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(tfrecord_file)) # 加载多个Tfrecode文件
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            # 'label': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        })

    record_image = tf.decode_raw(features['image'], tf.float32)

    # Changing the image into this shape helps train and visualize the output by converting it to
    # be organized like an image.
    # 修改图像的形状有助于训练和输出的可视化
    image = tf.reshape(record_image, [h, w, c])

    # label = tf.cast(features['label'], tf.string)
    label = tf.cast(features['label'], tf.int64)


    # label string-->int 0,1 标签
    # label = tf.case({tf.equal(label, tf.constant('ants')): lambda: tf.constant(0),
    #                         tf.equal(label, tf.constant('bees')): lambda: tf.constant(1),
    #                         }, lambda: tf.constant(-1), exclusive=True)


    min_after_dequeue = 1000
    batch_size = batch_size
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)


    '''
    # Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
    labels = list(map(lambda c: c.split("\\")[-2], glob.glob(imagepath))) # 找到目录名（标签） linux使用 "/"
    # Match every label from label_batch and return the index where they exist in the list of classes
    # 匹配每个来自label_batch的标签并返回它们在类别列表中的索引
    train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0], label_batch, dtype=tf.int64)
    '''

    # Converting the images to a float of [0,1) to match the expected input to convolution2d
    # 将图像转换为灰度值位于[0,1)的浮点类型，
    float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
    return float_image_batch,label_batch




if __name__=="__main__":
    img_batch,label_batch=load_images_from_tfrecord("output/training-images/*.tfrecords",28,28,1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                for i in range(100):
                    val, l = sess.run([img_batch, label_batch])
                    if i%5==0:
                        print(val.shape, l.shape,l)
                else:
                    break
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)