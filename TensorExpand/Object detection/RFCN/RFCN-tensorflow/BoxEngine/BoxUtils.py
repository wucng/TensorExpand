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

def iou(boxes, refBoxes, oneToAll=True):
	with tf.name_scope("IOU"):
		x0, y0, x1, y1 = tf.unstack(boxes, axis=1)
		ref_x0, ref_y0, ref_x1, ref_y1 = tf.unstack(refBoxes, axis=1)

		#Calculate box IOU
		x0=tf.reshape(x0,[-1,1])
		y0=tf.reshape(y0,[-1,1])
		x1=tf.reshape(x1,[-1,1])
		y1=tf.reshape(y1,[-1,1])

		if oneToAll:
			boxShape = [1,-1]
		else:
			boxShape = [-1,1]

		ref_x0=tf.reshape(ref_x0,boxShape)
		ref_y0=tf.reshape(ref_y0,boxShape)
		ref_x1=tf.reshape(ref_x1,boxShape)
		ref_y1=tf.reshape(ref_y1,boxShape)

		max_x0 = tf.maximum(x0, ref_x0)
		max_y0 = tf.maximum(y0, ref_y0)
		min_x1 = tf.minimum(x1, ref_x1)
		min_y1 = tf.minimum(y1, ref_y1)

		intersect = tf.maximum(min_x1 - max_x0 + 1, 0.0) * tf.maximum(min_y1 - max_y0 + 1, 0.0)
		union = (x1 - x0 + 1) * (y1 - y0 + 1) + (ref_x1 - ref_x0 + 1) * (ref_y1 - ref_y0 + 1) - intersect

		iou = intersect / union

		return iou

def filterSmallBoxes(boxes, others=None, minSize=16.0):
	with tf.name_scope("filterSmallBoxes"):
		x0,y0,x1,y1 = tf.unstack(boxes, axis=1)
		
		okIndices = tf.where(tf.logical_and((x1-x0) >= (minSize-1), (y1-y0) >= (minSize-1)))
		okIndices = tf.cast(okIndices, tf.int32)

		res = [tf.gather_nd(boxes, okIndices)]

		if others is not None:
			for o in others:
				res.append(tf.gather_nd(o, okIndices))

		return res

def x0y0x1y1_to_xywh(x0,y0,x1,y1):
	with tf.name_scope("x0y0x1y1_to_xywh"):
		x = (x0+x1)/2.0
		y = (y0+y1)/2.0
		w = x1-x0+1
		h = y1-y0+1
		return x,y,w,h

def xywh_to_x0y0x1y1(x,y,w,h):
	with tf.name_scope("xywh_to_x0y0x1y1"):
		w_per_2=w/2.0
		h_per_2=h/2.0

		return x-w_per_2+0.5, y-h_per_2+0.5, x+w_per_2-0.5, y+h_per_2-0.5

def nnToCenteredBox(x_raw, y_raw, w_raw, h_raw, wA, hA, inputDownscale, offset):
	with tf.name_scope('nnToCenteredBox'):
		s = tf.shape(x_raw)
		wIn = s[2]
		hIn = s[1]
		x = x_raw*wA + tf.reshape((tf.range(0.0,tf.cast(wIn, tf.float32), dtype=tf.float32)+0.5)*inputDownscale, [-1,1]) + offset[1]
		y = y_raw*hA + tf.reshape((tf.range(0.0,tf.cast(hIn, tf.float32), dtype=tf.float32)+0.5)*inputDownscale, [-1,1,1]) + offset[0]
		w = tf.exp(w_raw) * wA
		h = tf.exp(h_raw) * hA

		return x,y,w,h

def mergeBoxData(list):
	with tf.name_scope('mergeBoxData'):
		l2 = []
		for l in list:
			l2.append(tf.expand_dims(l, -1))
		
		res = tf.concat(l2, tf.rank(list[0]))
		return tf.reshape(res, [-1,len(list)])
	


def mergeCoordinates(x,y,w,h):
	with tf.name_scope('mergeCoordinates'):
		return mergeBoxData(list(xywh_to_x0y0x1y1(x,y,w,h)))


def nnToImageBoxes(x_raw, y_raw, w_raw, h_raw, wA, hA, inputDownscale, offset):
	with tf.name_scope("nnToImageBoxes"):
		return mergeCoordinates(*nnToCenteredBox(x_raw, y_raw, w_raw, h_raw, wA, hA, inputDownscale, offset))
