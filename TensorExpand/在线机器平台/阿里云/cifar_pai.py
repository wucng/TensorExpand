# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

from six.moves import urllib
import tarfile

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tensorflow.python.lib.io import file_io
import os
import sys
import numpy as np
import pickle
import argparse

FLAGS = None

def load_data(dirname, one_hot=False):
    X_train = []
    Y_train = []

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    fpath = os.path.join(dirname, 'test_batch')
    X_test, Y_test = load_batch(fpath)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    if one_hot:
        Y_train = to_categorical(Y_train, 10)
        Y_test = to_categorical(Y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)

#reporthook from stackoverflow #13881092
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def load_batch(fpath):
    object = file_io.read_file_to_string(fpath)
    #origin_bytes = bytes(object, encoding='latin1')
    # with open(fpath, 'rb') as f:
    if sys.version_info > (3, 0):
        # Python3
        d = pickle.loads(object, encoding='latin1')
    else:
        # Python2
        d = pickle.loads(object)
    data = d["data"]
    labels = d["labels"]
    return data, labels

def main(_):
    dirname = os.path.join(FLAGS.buckets, "")
    (X, Y), (X_test, Y_test) = load_data(dirname)
    print("load data done")

    X, Y = shuffle(X, Y)
    Y = to_categorical(Y, 10)
    Y_test = to_categorical(Y_test, 10)

    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network building
    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=96, run_id='cifar10_cnn')
    model_path = os.path.join(FLAGS.checkpointDir, "model.tfl")
    print(model_path)
    model.save(model_path)

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

