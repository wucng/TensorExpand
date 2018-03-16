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
import threading

def nonlinear(imageList, lower, upper):
	with tf.name_scope('nonlinear') as scope:
		factor = tf.random_uniform([], lower, upper)

		res=[]
		for i in imageList:
			res.append(tf.pow(i, factor))

		return res

def randomNormal(imageList, stddev):
	with tf.name_scope('randomNormal') as scope:
		factor = tf.random_uniform([], 0, stddev)

		res=[]
		for i in imageList:
			res.append(i+tf.random_normal(tf.shape(i), mean=0.0, stddev=factor))

		return res

def mirror(image, boxes):
	def doMirror(image, boxes):
		image = tf.reverse(image, axis=[2])
		x0,y0,x1,y1 = tf.unstack(boxes, axis=1)

		w=tf.cast(tf.shape(image)[2], tf.float32)
		x0_m=w-x1
		x1_m=w-x0

		return image, tf.stack([x0_m,y0,x1_m,y1], axis=1)
			
	uniform_random = tf.random_uniform([], 0, 1.0)
	return tf.cond(uniform_random < 0.5, lambda: (image, boxes), lambda: doMirror(image, boxes))

def augment(image, boxes, classes):
	with tf.name_scope('augmentation') as scope:
		image=image/255.0
		image = nonlinear([image], 0.8, 1.2)[0]

		image, boxes = mirror(image, boxes)

		image = tf.image.random_contrast(image, lower=0.3, upper=1.3)
		image = tf.image.random_brightness(image, max_delta=0.3)

		image = randomNormal([image], 0.025)[0]

		image = tf.clip_by_value(image, 0, 1.0)*255

		return image, boxes, classes
