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

from BoxEngine.BoxRefinementNetwork import BoxRefinementNetwork
from BoxEngine.RPN import RPN
import tensorflow as tf

class BoxNetwork:
	def __init__(self, nCategories, rpnLayer, rpnDownscale, rpnOffset, featureLayer=None, featureDownsample=None, featureOffset=None, weightDecay=1e-6, hardMining=True):
		if featureLayer is None:
			featureLayer=rpnLayer

		if featureDownsample is None:
			featureDownsample=rpnDownscale
			
		if featureOffset is None:
			rpnOffset=featureOffset

		with tf.name_scope("BoxNetwork"):
			self.rpn = RPN(rpnLayer, immediateSize=512, weightDecay=weightDecay, inputDownscale=rpnDownscale, offset=rpnOffset)
			self.boxRefiner = BoxRefinementNetwork(featureLayer, nCategories, downsample=featureDownsample, offset=featureOffset, hardMining=hardMining)

			self.proposals, self.proposalScores = self.rpn.getPositiveOutputs(maxOutSize=300)

		
	def getProposals(self, threshold=None):
		if threshold is not None and threshold>0:
			s = tf.cast(tf.where(self.proposalScores > threshold), tf.int32)
			return tf.gather_nd(self.proposals, s), tf.gather_nd(self.proposalScores, s)
		else:
			return self.proposals, self.proposalScores
		
	def getBoxes(self, nmsThreshold=0.3, scoreThreshold=0.8):
		return self.boxRefiner.getBoxes(self.proposals, self.proposalScores, maxOutputs=50, nmsThreshold=nmsThreshold, scoreThreshold=scoreThreshold)

	def getLoss(self, refBoxes, refClasses):
		return self.rpn.loss(refBoxes) + self.boxRefiner.loss(self.proposals, refBoxes, refClasses)
