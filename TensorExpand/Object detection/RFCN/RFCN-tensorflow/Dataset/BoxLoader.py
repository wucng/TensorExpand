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


import Dataset.coco.pycocotools.coco as coco
import random
import tensorflow as tf
import threading
import time
import numpy as np

class BoxLoader:
	QUEUE_CAPACITY=16

	def __init__(self, sources=[], initOnStart=True):
		self.totalCount = 0
		self.counts=[]
		self.sources=[]
		self.initDone=False
		self.initOnStart=initOnStart

		with tf.name_scope('dataset') as scope:
			self.queue = tf.FIFOQueue(dtypes=[tf.float32, tf.float32, tf.uint8],
					capacity=self.QUEUE_CAPACITY)

			self.image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name="image")
			self.boxes = tf.placeholder(dtype=tf.float32, shape=[None,4], name="boxes")
			self.classes = tf.placeholder(dtype=tf.uint8, shape=[None], name="classes")
			
			self.enqueueOp = self.queue.enqueue([self.image, self.boxes, self.classes])

		self.sources=sources[:]
	
	def categoryCount(self):
		return 80
	
	def threadFn(self, tid, sess):
		if tid==0:
			self.init()
		else:
			while not self.initDone:
				time.sleep(1)

		while True:
			img, boxes, classes=self.selectSource().load()
			try:
				sess.run(self.enqueueOp,feed_dict={self.image:img, self.boxes:boxes, self.classes:classes})
			except tf.errors.CancelledError:
				return

	def init(self):
		if not self.initOnStart:
			for s in self.sources:
				s.init()

		for s in self.sources:
			c = s.count()
			self.counts.append(c)
			self.totalCount+=c

		print("BoxLoader: Loaded %d files." % self.totalCount)
		self.initDone=True

	def startThreads(self, sess, nThreads=4):
		self.threads=[]
		for n in range(nThreads):
			t=threading.Thread(target=self.threadFn, args=(n,sess))
			t.daemon = True
			t.start()
			self.threads.append(t)

	def get(self):
		images, boxes, classes = self.queue.dequeue()
		images = tf.expand_dims(images, axis=0)

		images.set_shape([None, None, None, 3])
		boxes.set_shape([None,4])

		return images, boxes, classes

	def add(self, source):
		assert self.initDone==False
		if self.initOnStart:
			source.init()
		self.sources.append(source)

	def selectSource(self):
		i = random.randint(0, self.totalCount-1)
		acc = 0
		for j in range(len(self.counts)):
			acc += self.counts[j]
			if acc >= i:
				return self.sources[j]

	def count(self):
		return self.totalCount

	def getCaptions(self, categories):
		return self.sources[0].getCaptions(categories)

	def getCaptionMap(self):
		return self.getCaptions(np.arange(0,self.categoryCount()))