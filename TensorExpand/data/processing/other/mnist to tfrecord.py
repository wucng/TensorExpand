#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tflearn
# from tflearn.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

"""
mnist to tfrecord
"""

def mnist_to_tfrecord(images,labels,record_location):
    m = 0
    writer=None
    for image,label in zip(images,labels):
        image=np.reshape(image,[28,28,1]).astype(np.float32)
        image_bytes = image.tobytes()
        # image_label = label.encode("utf-8")
        label=np.uint8(label)

        if m%5000==0:
            if writer:
                writer.close()
            record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=m)
            writer = tf.python_io.TFRecordWriter(record_filename)
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        }))
        writer.write(example.SerializeToString())
        m+=1
    writer.close()

if __name__=="__main__":
    # (X, Y), (testX, testY) = cifar10.load_data()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # print(X.shape) # (50000, 32, 32, 3)
    # print(Y.shape) # (50000,)
    mnist_to_tfrecord(mnist.train.images, mnist.train.labels, './output/training-images/training-images')
    mnist_to_tfrecord(mnist.test.images, mnist.test.labels, './output/testing-images/testing-images')
