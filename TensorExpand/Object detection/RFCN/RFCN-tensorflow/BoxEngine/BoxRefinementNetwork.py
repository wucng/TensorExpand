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
import tensorflow.contrib.slim as slim
from BoxEngine.ROIPooling import positionSensitiveRoiPooling
import math
import Utils.RandomSelect
import BoxEngine.BoxUtils as BoxUtils
import Utils.MultiGather as MultiGather
import BoxEngine.Loss as Loss

class BoxRefinementNetwork:
	POOL_SIZE=3

	def __init__(self, input, nCategories, downsample=16, offset=[32,32], hardMining=True):
		self.downsample = downsample
		self.offset = offset
		self.nCategories = nCategories
		self.classMaps = slim.conv2d(input, (self.POOL_SIZE**2)*(1+nCategories), 3, activation_fn=None, scope='classMaps')
		self.regressionMap = slim.conv2d(input, (self.POOL_SIZE**2)*4, 3, activation_fn=None, scope='regressionMaps')

		self.hardMining=hardMining

		#Magic parameters.
		self.posIouTheshold = 0.5
		self.negIouThesholdHi = 0.5
		self.negIouThesholdLo = 0.1
		self.nTrainBoxes = 128
		self.nTrainPositives = 32
		self.falseValue = 0.0002

	def roiPooling(self, layer, boxes):
		return positionSensitiveRoiPooling(layer, boxes, offset=self.offset, downsample=self.downsample, roiSize=self.POOL_SIZE)

	def roiMean(self, layer, boxes):
		with tf.name_scope("roiMean"):
			return tf.reduce_mean(self.roiPooling(layer, boxes), axis=[1,2])

	def getBoxScores(self, boxes):
		with tf.name_scope("getBoxScores"):
			return self.roiMean(self.classMaps, boxes)

	def classRefinementLoss(self, boxes, refs):
		with tf.name_scope("classRefinementLoss"):
			netScores = self.getBoxScores(boxes)
			refOnehot = tf.one_hot(refs, self.nCategories+1, on_value=1.0 - self.nCategories*self.falseValue, off_value=self.falseValue)
		
			return tf.nn.softmax_cross_entropy_with_logits(logits=netScores, labels=refOnehot)

	def refineBoxes(self, boxes, needSizes):
		with tf.name_scope("refineBoxes"):
			boxFineData = self.roiMean(self.regressionMap, boxes)

			x,y,w,h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(boxes, axis=1))
			x_rel, y_rel, w_rel, h_rel = tf.unstack(boxFineData, axis=1)

			if needSizes:
				refSizes = tf.stack([h,w], axis=1)

			x = x + x_rel * w
			y = y + y_rel * h

			w = w * tf.exp(w_rel)
			h = h * tf.exp(h_rel)

			refinedBoxes = tf.stack(BoxUtils.xywh_to_x0y0x1y1(x,y,w,h), axis=1)

			if needSizes:
				return refinedBoxes, refSizes, boxFineData[:,2:4]
			else:
				return refinedBoxes

	def boxRefinementLoss(self, boxes, refBoxes):
		with tf.name_scope("boxesRefinementLoss"):
			refinedBoxes, refSizes, rawSizes = self.refineBoxes(boxes, True)
			return Loss.boxRegressionLoss(refinedBoxes, rawSizes, refBoxes, refSizes)

	def loss(self, proposals, refBoxes, refClasses):
		with tf.name_scope("BoxRefinementNetworkLoss"):
			proposals = tf.stop_gradient(proposals)

			def getPosLoss(positiveBoxes, positiveRefIndices, nPositive):
				with tf.name_scope("getPosLoss"):
					positiveRefIndices =  tf.reshape(positiveRefIndices,[-1,1])

					positiveClasses, positiveRefBoxes = MultiGather.gather([refClasses, refBoxes], positiveRefIndices)
					positiveClasses = tf.cast(tf.cast(positiveClasses,tf.int8) + 1, tf.uint8)

					if not self.hardMining:
						selected = Utils.RandomSelect.randomSelectIndex(tf.shape(positiveBoxes)[0], nPositive)
						positiveBoxes, positiveClasses, positiveRefBoxes = MultiGather.gather([positiveBoxes, positiveClasses, positiveRefBoxes], selected)

					return tf.tuple([self.classRefinementLoss(positiveBoxes, positiveClasses) + self.boxRefinementLoss(positiveBoxes, positiveRefBoxes), tf.shape(positiveBoxes)[0]])

			def getNegLoss(negativeBoxes, nNegative):
				with tf.name_scope("getNetLoss"):
					if not self.hardMining:
						negativeIndices = Utils.RandomSelect.randomSelectIndex(tf.shape(negativeBoxes)[0], nNegative)
						negativeBoxes = tf.gather_nd(negativeBoxes, negativeIndices)

					return self.classRefinementLoss(negativeBoxes, tf.zeros(tf.stack([tf.shape(negativeBoxes)[0],1]), dtype=tf.uint8))
			
			def getRefinementLoss():
				with tf.name_scope("getRefinementLoss"):
					iou = BoxUtils.iou(proposals, refBoxes)
					
					maxIou = tf.reduce_max(iou, axis=1)
					bestIou = tf.expand_dims(tf.cast(tf.argmax(iou, axis=1), tf.int32), axis=-1)

					#Find positive and negative indices based on their IOU
					posBoxIndices = tf.cast(tf.where(maxIou > self.posIouTheshold), tf.int32)
					negBoxIndices = tf.cast(tf.where(tf.logical_and(maxIou < self.negIouThesholdHi, maxIou > self.negIouThesholdLo)), tf.int32)

					#Split the boxes and references
					posBoxes, posRefIndices = MultiGather.gather([proposals, bestIou], posBoxIndices)
					negBoxes = tf.gather_nd(proposals, negBoxIndices)

					#Add GT boxes
					posBoxes = tf.concat([posBoxes,refBoxes], 0)
					posRefIndices = tf.concat([posRefIndices, tf.reshape(tf.range(tf.shape(refClasses)[0]), [-1,1])], 0)

					#Call the loss if the box collection is not empty
					nPositive = tf.shape(posBoxes)[0]
					nNegative = tf.shape(negBoxes)[0]

					if self.hardMining:
						posLoss = tf.cond(nPositive > 0, lambda: getPosLoss(posBoxes, posRefIndices, 0)[0], lambda: tf.zeros((0,), tf.float32))
						negLoss = tf.cond(nNegative > 0, lambda: getNegLoss(negBoxes, 0), lambda: tf.zeros((0,), tf.float32))

						allLoss = tf.concat([posLoss, negLoss], 0)
						return tf.cond(tf.shape(allLoss)[0]>0, lambda: tf.reduce_mean(Utils.MultiGather.gatherTopK(allLoss, self.nTrainBoxes)), lambda: tf.constant(0.0))
					else:
						posLoss, posCount = tf.cond(nPositive > 0, lambda: getPosLoss(posBoxes, posRefIndices, self.nTrainPositives), lambda: tf.tuple([tf.constant(0.0), tf.constant(0,tf.int32)]))
						negLoss = tf.cond(nNegative > 0, lambda: getNegLoss(negBoxes, self.nTrainBoxes-posCount), lambda: tf.constant(0.0))

						nPositive = tf.cast(tf.shape(posLoss)[0], tf.float32)
						nNegative = tf.cond(nNegative > 0, lambda: tf.cast(tf.shape(negLoss)[0], tf.float32), lambda: tf.constant(0.0))
						
						return (tf.reduce_mean(posLoss)*nPositive + tf.reduce_mean(negLoss)*nNegative)/(nNegative+nPositive)
	

		return tf.cond(tf.logical_and(tf.shape(proposals)[0] > 0, tf.shape(refBoxes)[0] > 0), lambda: getRefinementLoss(), lambda:tf.constant(0.0))

	def getBoxes(self, proposals, proposal_scores, maxOutputs=30, nmsThreshold=0.3, scoreThreshold=0.8):
		if scoreThreshold is None:
			scoreThreshold = 0

		with tf.name_scope("getBoxes"):
			scores = tf.nn.softmax(self.getBoxScores(proposals))
			
			classes = tf.argmax(scores, 1)
			scores = tf.reduce_max(scores, axis=1)
			posIndices = tf.cast(tf.where(tf.logical_and(classes > 0, scores>scoreThreshold)), tf.int32)

			positives, scores, classes = MultiGather.gather([proposals, scores, classes], posIndices)
			positives = self.refineBoxes(positives, False)

			#Final NMS
			posIndices = tf.image.non_max_suppression(positives, scores, iou_threshold=nmsThreshold, max_output_size=maxOutputs)
			posIndices = tf.expand_dims(posIndices, axis=-1)
			positives, scores, classes = MultiGather.gather([positives, scores, classes], posIndices)	
			
			classes = tf.cast(tf.cast(classes,tf.int32) - 1, tf.uint8)

			return positives, scores, classes