# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import tensorflow as tf

def variableSummary(var):
  if not isinstance(var, list):
    var=[var]

  for v in var:
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(v)
      tf.summary.scalar('mean/' + v.op.name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(v - mean)))
      tf.summary.scalar('stddev/' + v.op.name, stddev)
      tf.summary.scalar('max/' + v.op.name, tf.reduce_max(v))
      tf.summary.scalar('min/' + v.op.name, tf.reduce_min(v))
      tf.histogram_summary(v.op.name, v)

def createSummaryForAllVars():
  variableSummary(tf.trainable_variables())

def pyhtonFloatSummary(name):
  p=tf.placeholder(tf.float32)
  s=tf.summary.scalar(name, p)
  return s, p

def imageSummary(var):
  res=[]
  for name in var:
    res.append(tf.image_summary(name, var[name]))

  return res