"""
参考:
https://github.com/tensorflow/model/research/slim/datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"2,3"
# from datasets import dataset_utils

_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

LABELS_FILENAME = 'labels.txt'

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'size',
    'seven',
    'eight',
    'nine',
]


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))

def _extract_images_and_labels(filename,is_training):
    """extract_images_and_labels
    :param filename:
    :param is_training:
    :return:
    images:A numpy array of shape [number_of_images, height, width, channels]
    labels:a vector of int64 label IDs
    """
    mnist = input_data.read_data_sets(filename, one_hot=False)  # 'MNIST_data'
    if is_training:
        images = mnist.train.images.reshape(-1, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
        labels = mnist.train.labels.astype(np.int64)
    else:
        images = mnist.test.images.reshape(-1, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
        labels = mnist.test.labels.astype(np.int64)

    return images,labels


def _add_to_tfrecord(data_filename, is_training,
                     tfrecord_writer):
  """Loads data from the binary MNIST files and writes files to a TFRecord.

  Args:
    data_filename: The filename of the MNIST images.
    labels_filename: The filename of the MNIST labels.
    num_images: The number of images in the dataset.
    tfrecord_writer: The TFRecord writer to use for writing.
  """
  # images = _extract_images(data_filename, num_images)
  # labels = _extract_labels(labels_filename, num_images)

  images,labels=_extract_images_and_labels(data_filename,is_training)

  num_images=len(images)
  shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_png = tf.image.encode_png(image)

    with tf.Session('') as sess:
      for j in range(num_images):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
        sys.stdout.flush()

        png_string = sess.run(encoded_png, feed_dict={image: images[j]})

        example = image_to_tfexample(
            png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/mnist_%s.tfrecord' % (dataset_dir, split_name)


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    _add_to_tfrecord(dataset_dir, True, tfrecord_writer)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    _add_to_tfrecord(dataset_dir, False, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the MNIST dataset!')

if __name__=="__main__":
    run("./datas/MNIST_data")