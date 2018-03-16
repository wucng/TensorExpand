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


from Utils.ArgSave import *
import sys
import os

parser = StorableArgparse(description='RFCN trainer.')
parser.add_argument('-learningRate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-adamEps', type=float, default=1e-8, help='Adam epsilon')
parser.add_argument('-dataset', type=str, default="/data/Datasets/COCO", help="Path to COCO dataset")
parser.add_argument('-name', type=str, default="save", help="Directory to save checkpoints")
parser.add_argument('-saveInterval', type=int, default=10000, help='Save model for this amount of iterations')
parser.add_argument('-reportInterval', type=int, default=30, help='Repeat after this amount of iterations')
parser.add_argument('-displayInterval', type=int, default=60, help='Display after this amount of iterations')
parser.add_argument('-optimizer', type=str, default='adam', help='sgd/adam/rmsprop')
parser.add_argument('-resume', type=str, help='Resume from this file', save=False)
parser.add_argument('-report', type=str, default="", help='Create report here', save=False)
parser.add_argument('-trainFrom', type=str, default="-1", help='Train from this layer. Use 0 for all, -1 for just the added layers')
parser.add_argument('-hardMining', type=int, default=1, help="Enable hard example mining.")
parser.add_argument('-gpu', type=str, default="0", help='Train on this GPU(s)')
parser.add_argument('-mergeValidationSet', type=int, default=1, help='Merge validation set to training set.')
parser.add_argument('-profile', type=int, default=0, help='Enable profiling', save=False)
parser.add_argument('-randZoom', type=int, default=1, help='Enable box aware random zooming and cropping')
parser.add_argument('-freezeBatchNorm', type=int, default=1, help='Freeze batch normalization during finetuning.')
parser.add_argument('-export', type=str, help='Export model here.', save=False)
parser.add_argument('-cocoVariant', type=str, default="2014", help='Coco variant to load. 2014 or 2017')

opt=parser.parse_args()

if not os.path.isdir(opt.name):
	os.makedirs(opt.name)

opt = parser.load(opt.name+"/args.json")
parser.save(opt.name+"/args.json")

if not os.path.isdir(opt.name+"/log"):
	os.makedirs(opt.name+"/log")

if not os.path.isdir(opt.name+"/save"):
	os.makedirs(opt.name+"/save")

if not os.path.isdir(opt.name+"/preview"):
	os.makedirs(opt.name+"/preview")


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from Dataset.CocoDataset import *
from Dataset.BoxLoader import *
from Utils.RunManager import *
from Utils.CheckpointLoader import *
from BoxInceptionResnet import *
from Dataset import Augment
from Visualize import VisualizeOutput
from Utils import Model
from Utils import Export
from tensorflow.python.client import timeline
import re

globalStep = tf.Variable(0, name='globalStep', trainable=False)
globalStepInc=tf.assign_add(globalStep,1)

Model.download()

dataset = BoxLoader()
dataset.add(CocoDataset(opt.dataset, randomZoom=opt.randZoom==1, set="train"+opt.cocoVariant))
if opt.mergeValidationSet==1:
	dataset.add(CocoDataset(opt.dataset, set="val"+opt.cocoVariant))


images, boxes, classes = Augment.augment(*dataset.get())


print("Number of categories: "+str(dataset.categoryCount()))
print(dataset.getCaptionMap())


net = BoxInceptionResnet(images, dataset.categoryCount(), name="boxnet", trainFrom=opt.trainFrom, hardMining=opt.hardMining==1, freezeBatchNorm=opt.freezeBatchNorm==1)
tf.losses.add_loss(net.getLoss(boxes, classes))

def createUpdateOp(gradClip=1):
	with tf.name_scope("optimizer"):
		optimizer=tf.train.AdamOptimizer(learning_rate=opt.learningRate, epsilon=opt.adamEps)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		totalLoss = tf.losses.get_total_loss()
		grads = optimizer.compute_gradients(totalLoss, var_list=net.getVariables())
		if gradClip is not None:
			cGrads = []
			for g, v in grads:
				if g is None:
					print("WARNING: no grad for variable "+v.op.name)
					continue
				cGrads.append((tf.clip_by_value(g, -float(gradClip), float(gradClip)), v))
			grads = cGrads

		update_ops.append(optimizer.apply_gradients(grads))
		return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')

trainOp=createUpdateOp()

saver=tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=100)


if opt.profile==1:
	runOptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	runMetadata = tf.RunMetadata()
	iterationsSinceStart=0
else:
	runOptions=None
	runMetadata=None

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
	if not loadCheckpoint(sess, opt.name+"/save/", opt.resume):
		print("Loading GoogleNet")
		net.importWeights(sess, "./inception_resnet_v2_2016_08_30.ckpt")
		#net.importWeights(sess, "initialWeights/", permutateRgb=False)
		print("Done.")

	if opt.export is not None:
		Export.exportModel(sess, opt.export, [lambda name: name.split("/")[0]=="boxnet" and not re.match("^[Aa]dam(_.*)?$",name.split("/")[-1])])
		sys.exit(-1)

	dataset.startThreads(sess)

	runManager = RunManager(sess, options=runOptions, run_metadata=runMetadata)
	runManager.add("train", [globalStepInc,trainOp], modRun=1)


	visualizer = VisualizeOutput.OutputVisualizer(opt, runManager, dataset, net, images, boxes, classes)

	i=1
	cycleCnt=0
	lossSum=0

	while True:
		#run various parts of the network
		res = runManager.modRun(i)

		if opt.profile==1:
			print("Profiling step %d" % iterationsSinceStart)
			iterationsSinceStart+=1
			if iterationsSinceStart==5:
				print("Writing profile data...")
				tl = timeline.Timeline(runMetadata.step_stats)
				ctf = tl.generate_chrome_trace_format()
				with open('timeline.json', 'w') as f:
					f.write(ctf)

				print("Done.")
				sys.exit(0)

			
		i, loss=res["train"]

		lossSum+=loss
		cycleCnt+=1

		visualizer.draw(res)

		if i % opt.reportInterval == 0:
			if cycleCnt>0:
				loss=lossSum/cycleCnt

			# lossS=sess.run(trainLossSum, feed_dict={
			# 	trainLossFeed: loss
			# })
			# log.add_summary(lossS, global_step=samplesSeen)

			epoch="%.2f" % (float(i) / dataset.count())
			print("Iteration "+str(i)+" (epoch: "+epoch+"): loss: "+str(loss))
			lossSum=0
			cycleCnt=0

		if i % opt.saveInterval == 0:
			print("Saving checkpoint "+str(i))
			saver.save(sess, opt.name+"/save/model_"+str(i), write_meta_graph=False)
