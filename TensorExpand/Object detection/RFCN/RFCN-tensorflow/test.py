#!/usr/bin/python
#
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

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from BoxInceptionResnet import BoxInceptionResnet
from Visualize import Visualize
from Utils import CheckpointLoader
from Utils import PreviewIO

parser = argparse.ArgumentParser(description="RFCN tester")
parser.add_argument('-gpu', type=str, default="0", help='Train on this GPU(s)')
parser.add_argument('-n', type=str, help='Network checkpoint file')
parser.add_argument('-i', type=str, help='Input file.')
parser.add_argument('-o', type=str, default="", help='Write output here.')
parser.add_argument('-p', type=int, default=1, help='Show preview')
parser.add_argument('-threshold', type=float, default=0.5, help='Detection threshold')
parser.add_argument('-delay', type=int, default=-1, help='Delay between frames in visualization. -1 for automatic, 0 for wait for keypress.')

opt=parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# categories=['diningtable', 'person', 'bottle', 'boat', 'train', 'bird', 'dog', 'cat', 'tvmonitor', 'cow', 'car', 'sofa',
# 			'horse', 'chair', 'pottedplant', 'bicycle', 'motorbike', 'aeroplane', 'sheep', 'bus']


palette = Visualize.Palette(len(categories))

image = tf.placeholder(tf.float32, [None, None, None, 3])
net = BoxInceptionResnet(image, len(categories), name="boxnet")

boxes, scores, classes = net.getBoxes(scoreThreshold=opt.threshold)


input = PreviewIO.PreviewInput(opt.i)
output = PreviewIO.PreviewOutput(opt.o, input.getFps())

def preprocessInput(img):
	def calcPad(size):
		m = size % 32
		p = int(m/2)
		s = size - m
		return s,p

	zoom = max(640.0 / img.shape[0], 640.0 / img.shape[1])
	img = cv2.resize(img, (int(zoom*img.shape[1]), int(zoom*img.shape[0])))

	if img.shape[0] % 32 != 0:
		s,p = calcPad(img.shape[0])
		img = img[p:p+s]

	if img.shape[1] % 32 != 0:
		s,p = calcPad(img.shape[1])
		img = img[:,p:p+s]

	return img

with tf.Session() as sess:
	if not CheckpointLoader.loadCheckpoint(sess, opt.n,None, ignoreVarsInFileNotInSess=True):
		print("Failed to load network.")
		sys.exit(-1)

	while True:
		img = input.get()
		if img is None:
			break

		img = preprocessInput(img)	

		rBoxes, rScores, rClasses = sess.run([boxes, scores, classes], feed_dict={image: np.expand_dims(img, 0)})

		res = Visualize.drawBoxes(img, rBoxes, rClasses, [categories[i] for i in rClasses.tolist()], palette, scores=rScores)

		output.put(input.getName(), res)

		if opt.p==1:
			cv2.imshow("result", res)
			if opt.o=="":
				cv2.waitKey(input.getDelay() if opt.delay <0 else opt.delay)
			else:
				cv2.waitKey(1)
