# -*- coding:utf-8 -*-
# file_name:data_download.py
# time: 2018.8.8 16:40
"""
使用tensorflow 数据下载并转成.npz
参考：https://github.com/tensorflow/models/tree/master/official/boosted_trees/data_download.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

# pylint: disable=g-bad-import-order
import numpy as np
import pandas as pd
from six.moves import urllib
from absl import app as absl_app
from absl import flags
import tensorflow as tf

URL_ROOT = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine'
#"https://archive.ics.uci.edu/ml/machine-learning-databases/00280"
INPUT_FILE = 'wine.data'#"HIGGS.csv.gz"
NPZ_FILE = 'wine.data.npz'#"HIGGS.csv.gz.npz"  # numpy compressed file to contain "data" array.


def _download_higgs_data_and_save_npz(data_dir):
  """Download higgs data and store as a numpy compressed file."""
  input_url = os.path.join(URL_ROOT, INPUT_FILE)
  np_filename = os.path.join(data_dir, NPZ_FILE)
  if tf.gfile.Exists(np_filename):
    raise ValueError("data_dir already has the processed data file: {}".format(
        np_filename))
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MkDir(data_dir)
  # 2.8 GB to download.
  try:
    tf.logging.info("Data downloading...")
    temp_filename, _ = urllib.request.urlretrieve(input_url)
    # Reading and parsing 11 million csv lines takes 2~3 minutes.
    tf.logging.info("Data processing... taking multiple minutes...")

    # with gzip.open(temp_filename, "rb") as csv_file: # .gzp文件打开方式
    with open(temp_filename, "rb") as csv_file:
      data = pd.read_csv(
          csv_file,
          dtype=np.float32,
          names=["c%02d" % i for i in range(14)]  # label +13 features,  # 29 label + 28 features.
      ).as_matrix()
  finally:
    tf.gfile.Remove(temp_filename)

  # Writing to temporary location then copy to the data_dir (0.8 GB).
  f = tempfile.NamedTemporaryFile()
  np.savez_compressed(f, data=data)
  tf.gfile.Copy(f.name, np_filename)
  tf.logging.info("Data saved to: {}".format(np_filename))


def main(unused_argv):
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MkDir(FLAGS.data_dir)
  _download_higgs_data_and_save_npz(FLAGS.data_dir)


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="/tmp/higgs_data",
      help="Directory to download higgs dataset and store training/eval data.")

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_data_download_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
