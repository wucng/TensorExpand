"""
参考:
https://github.com/tensorflow/model/research/slim/datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Shortcuts for later.
queues = tf.contrib.slim.queues
# layers = tf.contrib.layers
# ds = tf.contrib.distributions
# framework = tf.contrib.framework

MNIST_DATA_DIR = './datas/MNIST_data'

slim = tf.contrib.slim

_FILE_PATTERN = 'mnist_%s.tfrecord'

_SPLITS_TO_SIZES = {'train': 60000, 'test': 10000}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [28 x 28 x 1] grayscale image.',
    'label': 'A single integer between 0 and 9',
}

LABELS_FILENAME = 'labels.txt'

def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature(
          [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
      'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if has_labels(dataset_dir):
    labels_to_names = read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)

def provide_data(split_name, batch_size, dataset_dir, num_readers=1,
                 num_threads=1,num_epochs=None):
  """Provides batches of MNIST digits.

  Args:
    split_name: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the MNIST data can be found.
    num_readers: Number of dataset readers.
    num_threads: Number of prefetching threads.

  Returns:
    images: A `Tensor` of size [batch_size, 28, 28, 1]
    one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
      each row has a single element set to one and the rest set to zeros.
    num_samples: The number of total samples in the dataset.

  Raises:
    ValueError: If `split_name` is not either 'train' or 'test'.
  """
  dataset = get_split( split_name, dataset_dir=dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == 'train'),
      num_epochs=num_epochs)
  [image, label] = provider.get(['image', 'label'])

  # Preprocess the images.
  image = (tf.to_float(image) - 128.0) / 128.0

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=5 * batch_size)

  one_hot_labels = tf.one_hot(labels, dataset.num_classes)
  return images, one_hot_labels, dataset.num_samples


def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.

    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')

batch_size = 32
with tf.name_scope('inputs'):
    with tf.device('/cpu:0'):
        images, one_hot_labels, _ = provide_data(
            'train', batch_size, MNIST_DATA_DIR, num_threads=4,num_epochs=None) # num_epochs=None 数据将无限循环。

global_step=tf.train.create_global_step()#tf.train.global_step()
update_global_step=tf.assign_add(global_step,1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            with queues.QueueRunners(sess):
                images_np,labels_np,_,gstep = sess.run([images,one_hot_labels,update_global_step,global_step])
                # print(images_np.shape)
                # print(labels_np.shape)
                print(np.argmax(labels_np,-1))
                if gstep==10:
                    break
    except Exception as e:
    # except tf.errors.OutOfRangeError:
    #     print('Done training -- epoch limit reached')
        print(e)
    finally:
        coord.request_stop()
    coord.join(threads)