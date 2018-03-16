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

import Utils.RandomSelect

from InceptionResnetV2 import *
from BoxEngine.BoxNetwork import BoxNetwork


class BoxInceptionResnet(BoxNetwork):
	LAYER_NAMES = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','MaxPool_3a_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3',
			  'MaxPool_5a_3x3','Mixed_5b','Repeat','Mixed_6a','Repeat_1','Mixed_7a','Repeat_2','Block8','Conv2d_7b_1x1']

	def __init__(self, inputs, nCategories, name="BoxNetwork", weightDecay=0.00004, freezeBatchNorm=False, reuse=False, isTraining=True, trainFrom=None, hardMining=True):
		self.boxThreshold = 0.5

		try:
			trainFrom = int(trainFrom)
		except:
			pass

		if isinstance(trainFrom, int):
			trainFrom = self.LAYER_NAMES[trainFrom]


		print("Training network from "+(trainFrom if trainFrom is not None else "end"))

		with tf.variable_scope(name, reuse=reuse) as scope:
			self.googleNet = InceptionResnetV2("features", inputs, trainFrom=trainFrom, freezeBatchNorm=freezeBatchNorm)
			self.scope=scope
		
			with tf.variable_scope("Box"):
				#Pepeat_1 - last 1/16 layer, Mixed_6a - first 1/16 layer
				scale_16 = self.googleNet.getOutput("Repeat_1")[:,1:-1,1:-1,:]
				#scale_16 = self.googleNet.getOutput("Mixed_6a")[:,1:-1,1:-1,:]
				scale_32 = self.googleNet.getOutput("PrePool")

				with slim.arg_scope([slim.conv2d],
						weights_regularizer=slim.l2_regularizer(weightDecay),
						biases_regularizer=slim.l2_regularizer(weightDecay),
						padding='SAME',
						activation_fn = tf.nn.relu):

					net = tf.concat([ tf.image.resize_bilinear(scale_32, tf.shape(scale_16)[1:3]), scale_16], 3)
					rpnInput = slim.conv2d(net, 1024, 1)
					
					#BoxNetwork.__init__(self, nCategories, rpnInput, 16, [32,32], scale_32, 32, [32,32], weightDecay=weightDecay, hardMining=hardMining)
					featureInput = slim.conv2d(net, 1536, 1)
					BoxNetwork.__init__(self, nCategories, rpnInput, 16, [32,32], featureInput, 16, [32,32], weightDecay=weightDecay, hardMining=hardMining)
	
	def getVariables(self, includeFeatures=False):
		if includeFeatures:
			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
		else:
			vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name+"/Box/")
			vars += self.googleNet.getTrainableVars()

			print("Training variables: ", [v.op.name for v in vars])
			return vars

	def importWeights(self, sess, filename):
		self.googleNet.importWeights(sess, filename, includeTraining=True)
