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
import math
import Utils.RandomSelect
import BoxEngine.BoxUtils as BoxUtils
import BoxEngine.Loss as Loss
import Utils.MultiGather as MultiGather

class RPN:
	def __init__(self, input, anchors=None, immediateSize=512, weightDecay=1e-5, inputDownscale=16, offset=[32,32]):
		self.input = input
		self.anchors = anchors
		self.inputDownscale = inputDownscale
		self.offset = offset
		self.anchors = anchors if anchors is not None else self.makeAnchors([64,128,256,512])
		print("Anchors: ", self.anchors)
		self.tfAnchors = tf.constant(self.anchors, dtype=tf.float32)

		self.hA=tf.reshape(self.tfAnchors[:,0],[-1])
		self.wA=tf.reshape(self.tfAnchors[:,1],[-1])

		self.nAnchors = len(self.anchors)

		self.positiveIouThreshold=0.7
		self.negativeIouThreshold=0.3
		self.regressionWeight=1.0
		
		self.nBoxLosses=256
		self.nPositiveLosses=128

		#dimensions
		with tf.name_scope('dimension_info'):
			s = tf.shape(self.input)
			self.hIn = s[1]
			self.wIn = s[2]

		
		self.imageH = tf.cast(self.hIn*self.inputDownscale+self.offset[0]*2, tf.float32)
		self.imageW = tf.cast(self.wIn*self.inputDownscale+self.offset[1]*2, tf.float32)

		self.define(immediateSize, weightDecay)


	def define(self, immediateSize, weightDecay):
		with tf.name_scope('RPN'):
			with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weightDecay), padding='SAME'):
				#box prediction layers
				with tf.name_scope('NN'):
					net = slim.conv2d(self.input, immediateSize, 3, activation_fn=tf.nn.relu)
					scores = slim.conv2d(net, 2*self.nAnchors, 1, activation_fn=None)
					boxRelativeCoordinates = slim.conv2d(net, 4*self.nAnchors, 1, activation_fn=None)

				#split coordinates
				x_raw, y_raw, w_raw, h_raw = tf.split(boxRelativeCoordinates, 4, axis=3)

				#Save raw box sizes for loss
				self.rawSizes = BoxUtils.mergeBoxData([w_raw, h_raw])
							
				#Convert NN outputs to BBox coordinates
				self.boxes = BoxUtils.nnToImageBoxes(x_raw, y_raw, w_raw, h_raw, self.wA, self.hA, self.inputDownscale, self.offset)

				#store the size of every box
				with tf.name_scope('box_sizes'):
					boxSizes = tf.reshape(self.tfAnchors, [1,1,1,-1,2])
					boxSizes = tf.tile(boxSizes, tf.stack([1,self.hIn,self.wIn,1,1]))
					self.boxSizes = tf.reshape(boxSizes, [-1,2])

				#scores
				self.scores = tf.reshape(scores, [-1,2])

	def genAllAnchors(self):
		with tf.name_scope('genAllAnchors'):
			z = tf.zeros([1, self.hIn, self.wIn, self.nAnchors], tf.float32)
			return BoxUtils.nnToImageBoxes(z, z, z, z, self.wA, self.hA, self.inputDownscale, self.offset)

	def loss(self, refBoxes):
		def getPositiveBoxes(boxes):
			with tf.name_scope('getPositiveBoxes'):
				iou = BoxUtils.iou(boxes, refBoxes)
				
				maxIou = tf.reduce_max(iou, axis=1)
				bestIou = tf.expand_dims(tf.cast(tf.argmax(iou, axis=1), tf.int32), axis=-1)

				bestAnchors = tf.argmax(iou, axis=0)
				#Box matching matrix
				boxMatches = tf.cast(iou > self.positiveIouThreshold, tf.float32)

				boxMatches = tf.minimum(boxMatches + tf.transpose(tf.one_hot(bestAnchors, tf.shape(boxMatches)[0])), 1.0)

				boxMatchMatrix = tf.stop_gradient(boxMatches)

				#Find positive boxes
				oneIfPositive = tf.reduce_max(boxMatchMatrix, axis=1)
				oneIfPositive = tf.stop_gradient(oneIfPositive)

				return oneIfPositive, maxIou, bestIou


		def getPositiveLoss(boxes, rawSizes, boxSizes, positiveIndices, bestIou, classificationLoss):
			with tf.name_scope('getPositiveLoss'):
				positiveBoxes, positiveRawSizes, positiveBoxSizes, positiveRefIndices, positiveClassificationLoss = \
					MultiGather.gather([boxes, rawSizes, boxSizes, bestIou, classificationLoss], positiveIndices)
			
				#Regression loss
				positiveRefs = tf.gather_nd(refBoxes, positiveRefIndices)
				return Loss.boxRegressionLoss(positiveBoxes, positiveRawSizes, positiveRefs, positiveBoxSizes)*self.regressionWeight  + positiveClassificationLoss

		def emptyPositiveLoss():
			with tf.name_scope('emptyPositiveLoss'):
				return tf.zeros((0,),tf.float32)

		def getNegativeLosses(boxes, negativeIndices, classificationLoss):
			with tf.name_scope('getNegativeLosses'):
				return tf.gather_nd(classificationLoss, negativeIndices)
			
		def emptyNegativeLoss():
			with tf.name_scope('emptyNegativeLoss'):
				return tf.zeros((0,),tf.float32)

		def calcAllLosses(refAnchors, boxes, rawSizes, scores, boxSizes):
			with tf.name_scope('calcAllRPNLosses'):
				oneIfPositive, maxIou, bestIou = getPositiveBoxes(refAnchors)
			
				#Classification loss
				refScores = tf.one_hot(tf.cast(oneIfPositive>0.5, tf.uint8), 2, on_value=0.999, off_value=0.001)
				classificationLoss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=refScores)

				#Split to positive and negative
				positiveIndices = tf.stop_gradient(tf.cast(tf.where(oneIfPositive >= 0.5), tf.int32))
				negativeIndices = tf.stop_gradient(tf.cast(tf.where(tf.logical_and(oneIfPositive < 0.5, maxIou < self.negativeIouThreshold)), tf.int32))

				p = tf.cond(tf.shape(positiveIndices)[0]>0, lambda:getPositiveLoss(boxes, rawSizes, boxSizes, positiveIndices, bestIou, classificationLoss), lambda: emptyPositiveLoss())
				n = tf.cond(tf.shape(negativeIndices)[0]>0, lambda:getNegativeLosses(boxes, negativeIndices, classificationLoss), lambda: emptyNegativeLoss())

				#return positive losses, negative losses, positive boxes, positive reference indices, negative boxes
				return p, n

		def selectAndSum(losses, n):
			with tf.name_scope('selectAndSum'):
				l = Utils.RandomSelect.randomSelectBatch(losses, n)
				return tf.reduce_mean(l)

		def calcLoss():
			with tf.name_scope('calcRPNLoss'):
				#Filter cross boundary boxes and get positives
				inAnchros, inBoxes, inScores, inBoxSizes, inRawSizes = self.filterCrossBoundaryBoxes(self.genAllAnchors(), [self.boxes, self.scores, self.boxSizes, self.rawSizes])
				positiveLosses, negativeLosses = calcAllLosses(inAnchros, inBoxes, inRawSizes, inScores, inBoxSizes)

				pCount = tf.shape(positiveLosses)[0]
				nCount = tf.shape(negativeLosses)[0]

				nPositive = tf.minimum(pCount, self.nPositiveLosses)
				nNegative = tf.minimum(self.nBoxLosses - nPositive, nCount)
				
				n = tf.cond(nNegative > 0, lambda: selectAndSum(negativeLosses, nNegative), lambda: tf.constant(0.0))
				p = tf.cond(nPositive > 0, lambda: selectAndSum(positiveLosses, nPositive), lambda: tf.constant(0.0))
				return n+p
		
		with tf.name_scope('RPNloss'):
			return tf.cond(tf.shape(refBoxes)[0]>0, lambda: calcLoss(), lambda: tf.constant(0.0))
		
	def getInsideMask(self, boxes, boxInsideRate=1.0):
		with tf.name_scope('getInsideMask'):
			x0, y0, x1, y1 = tf.unstack(boxes, axis=1)
			
			if boxInsideRate!=1.0:
				w = x1-x0+1.0
				h = y1-y0+1.0

				xOutside = (1.0 - boxInsideRate) * w
				yOutside = (1.0 - boxInsideRate) * h
			else:
				xOutside = 0
				yOutside = 0
		
			return tf.logical_and(tf.logical_and(x0 >= -xOutside, y0 > -yOutside), tf.logical_and(x1 < (tf.cast(self.imageW, tf.float32)+xOutside), y1 < (tf.cast(self.imageH, tf.float32)+yOutside)))
	
	def filterCrossBoundaryBoxes(self, boxes, others=[], boxInsideRate=1.0):
		with tf.name_scope('filterCrossBoundaryBoxes'):
			okIndices = tf.where(self.getInsideMask(boxes, boxInsideRate))
			okIndices = tf.cast(okIndices, tf.int32)

			return MultiGather.gather([boxes]+others, okIndices)

	def clipBoxesToEdge(self, boxes):
		with tf.name_scope("clipBoxesToEdge"):
			x0,y0,x1,y1 = tf.unstack(boxes, axis=1)
			x0 = tf.maximum(tf.minimum(x0, self.imageW), 0.0)
			x1 = tf.maximum(tf.minimum(x1, self.imageW), 0.0)
			y0 = tf.maximum(tf.minimum(y0, self.imageH), 0.0)
			y1 = tf.maximum(tf.minimum(y1, self.imageH), 0.0)
			return tf.stack([x0,y0,x1,y1], axis=1)
		

	def filterOutputBoxes(self, boxes, scores, others=[], preNmsCount=6000, maxOutSize=300, nmsThreshold=0.7): 
		with tf.name_scope("filter_output_boxes"):
			scores = tf.nn.softmax(scores)[:,1]
			scores = tf.reshape(scores,[-1])

			#Clip boxes to edge
			boxes = self.clipBoxesToEdge(boxes)

			#Remove empty boxes
			boxes, scores = BoxUtils.filterSmallBoxes(boxes, [scores])
			scores, boxes = tf.cond(tf.shape(scores)[0] > preNmsCount , lambda: tf.tuple(MultiGather.gatherTopK(scores, preNmsCount, [boxes])), lambda: tf.tuple([scores, boxes]))

			#NMS filter
			nmsIndices = tf.image.non_max_suppression(boxes, scores, iou_threshold=nmsThreshold, max_output_size=maxOutSize)
			nmsIndices = tf.expand_dims(nmsIndices, axis=-1)

			return MultiGather.gather([boxes, scores]+others, nmsIndices)
		
	def getPositiveOutputs(self, preNmsCount=6000, maxOutSize=300, nmsThreshold=0.7):
		boxes, scores = self.filterOutputBoxes(self.boxes, self.scores, preNmsCount=preNmsCount, nmsThreshold=nmsThreshold, maxOutSize=maxOutSize)
		return boxes, scores

	@staticmethod
	def makeAnchors(sizeList, sizeLim=[1024, 1024]):
		res = []
		for s in sizeList:
			res.append([s,s])
			if s*2 <= sizeLim[0]:
				res.append([int(s*math.sqrt(2)), int(s/math.sqrt(2))])
			if s*2 <= sizeLim[1]:
				res.append([int(s/math.sqrt(2)), int(s*math.sqrt(2))])

		return res
	