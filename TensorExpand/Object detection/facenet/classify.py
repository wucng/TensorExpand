"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC

'''
CUDA_VISIBLE_DEVICES=1 python3 src/classify.py ./train/ ../20170512-110547/20170512-110547.pb --batch_size 128
'''

def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            np.random.shuffle(dataset)

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") # [n,128]
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]

            x = tf.placeholder(tf.float32, (None, 128), name='x')
            y_true = tf.placeholder(tf.int64, shape=[None, ], name='y_true')

            # add some layer
            with tf.variable_scope('Div'):
                fc1 = tf.layers.dense(x, 128, activation=tf.nn.elu, name='fc1')
                fc=tf.layers.dense(fc1,48,activation=tf.nn.sigmoid,name='coding_layer')
                y_pred=tf.layers.dense(fc,751,activation=tf.nn.softmax,name='output')

            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

            correct_prediction = tf.equal(tf.argmax(y_pred,1), y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # 只更新coding_layer与output这两层参数，其他的参数不更新
            tvars = tf.trainable_variables()  # 获取所有可以更新的变量
            d_params = [v for v in tvars if v.name.startswith('Div/')]
            lr=1e-3
            train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost, var_list=d_params)
            init = tf.variables_initializer(var_list=d_params)

            sess.run(init)
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            # emb_array = np.zeros((nrof_images, embedding_size))

            for epoch in range(10):
                for i in range(nrof_batches_per_epoch):
                    start_index = i*args.batch_size
                    end_index = min((i+1)*args.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    # emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array= sess.run(embeddings, feed_dict=feed_dict)

                    feed_dict2={x:emb_array,y_true:labels[start_index:end_index]}
                    sess.run(train_op,feed_dict2)

                    if i%20==0:
                        loss,acc=sess.run([cost,accuracy],feed_dict2)
                        print('loss',loss,'|','acc',acc)

                saver=tf.train.Saver()
                saver.save(sess,'./models/model.ckpt')

            sess.close()
            exit(0)

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')

    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
