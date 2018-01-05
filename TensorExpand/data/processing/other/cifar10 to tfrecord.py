#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflearn
from tflearn.datasets import cifar10
import tensorflow as tf
import numpy as np

"""
cifar10 to tfrecord
"""


def cifar10_to_tfrecord(images,labels,record_location):
    m = 0
    writer=None
    for image,label in zip(images,labels):
        image=np.reshape(image,[32,32,3]).astype(np.float32)
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
    (X, Y), (testX, testY) = cifar10.load_data()
    # print(X.shape) # (50000, 32, 32, 3)
    # print(Y.shape) # (50000,)
    cifar10_to_tfrecord(X, Y, './output/training-images/training-images')
    cifar10_to_tfrecord(testX, testY, './output/testing-images/testing-images')
