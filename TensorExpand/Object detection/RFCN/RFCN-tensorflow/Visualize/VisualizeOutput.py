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

from . import Visualize
import threading
import cv2

try:
	import queue
except:
	import Queue as queue

class OutputVisualizer:
	def __init__(self, opt, runManager, dataset, net, images, boxes, classes):
		self.opt = opt
		self.queue = queue.Queue()
		self.dataset = dataset
		self.palette = Visualize.Palette(dataset.categoryCount())

		predBoxes, predScores, predClasses = net.getBoxes()
		allPredBoxes, allPredScores, allPredClasses = net.getBoxes(scoreThreshold=0)
		proposals, proposalScores = net.getProposals()

		runManager.add("preview", [images, boxes, classes, predBoxes, predClasses, predScores, proposals, proposalScores, allPredBoxes, allPredScores, allPredClasses], modRun=self.opt.displayInterval)		
		self.startThread()


	def threadFn(self):
		while True:
			refImg, refBox, refClasses, pBoxes, pClasses, pScores, pProposals, pProposalScores, pAllBoxes, pAllScores, pAllClasses = self.queue.get()
			refImg=refImg[0]

			a = Visualize.drawBoxes(refImg, refBox, refClasses, self.dataset.getCaptions(refClasses), self.palette)
			b = Visualize.drawBoxes(refImg, pBoxes, pClasses, self.dataset.getCaptions(pClasses), self.palette, scores=pScores)
			c = Visualize.drawBoxes(refImg, pProposals, None, None, self.palette, scores=pProposalScores*0.3)
			d = Visualize.drawBoxes(refImg, pAllBoxes, pAllClasses, self.dataset.getCaptions(pAllClasses), self.palette, scores=pAllScores)

			preview = Visualize.tile(2,2, [a,b,c,d], ["input", "output", "proposals", "all detections"])

			cv2.imwrite(self.opt.name+"/preview/preview.jpg", preview)

			self.queue.task_done()
 
	def startThread(self):
		self.thread=threading.Thread(target=self.threadFn)
		self.thread.daemon = True
		self.thread.start()

	def draw(self, res):
		if "preview" not in res:
			return

		while not self.queue.empty():
			try:
				self.queue.get(False)
				self.queue.task_done()
			except queue.Empty:
				continue
 
		self.queue.put(res["preview"])
		