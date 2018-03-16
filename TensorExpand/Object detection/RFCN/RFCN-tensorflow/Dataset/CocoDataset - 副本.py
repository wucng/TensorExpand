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

from .coco.pycocotools import coco
import random
import numpy as np
import cv2
import tensorflow as tf
from . import BoxAwareRandZoom

class CocoDataset:
	def __init__(self, path, set="train2017", normalizeSize=True, randomZoom=True):
		print(path)
		self.path=path
		self.coco=None
		self.normalizeSize=normalizeSize
		self.set=set
		self.randomZoom=randomZoom

	def init(self):
		self.coco=coco.COCO(self.path+"/annotations/instances_"+self.set+".json")
		self.images=self.coco.getImgIds()

		self.toCocoCategory=[]
		self.fromCocoCategory={}

		cats = self.coco.dataset['categories']
		for i in range(len(cats)):
			self.fromCocoCategory[cats[i]["id"]] = i
			self.toCocoCategory.append(cats[i]["id"])

		print("Loaded "+str(len(self.images))+" COCO images")
	
  
	def getCaptions(self, categories):
		if categories is None:
			return None

		res = []
		if isinstance(categories, np.ndarray):
			categories = categories.tolist()

		for c in categories:
			res.append(self.coco.cats[self.toCocoCategory[c]]["name"])

		return res

	def load(self):
		while True:
			#imgId=self.images[1]
			#imgId=self.images[3456]
			imgId=self.images[random.randint(0, len(self.images)-1)]
	  
			instances = self.coco.loadAnns(self.coco.getAnnIds(imgId, iscrowd=False))
		
			#Ignore crowd images
			crowd = self.coco.loadAnns(self.coco.getAnnIds(imgId, iscrowd=True))
			if len(crowd)>0:
				continue;

			imgFile=self.path+"/"+self.set+"/"+self.coco.loadImgs(imgId)[0]["file_name"]
			img = cv2.imread(imgFile)

			if img is None:
				print("ERROR: Failed to load "+imgFile)
				continue

			sizeMul = 1.0
			padTop = 0
			padLeft = 0

			if len(instances)<=0:
				continue

			iBoxes=[{
						"x":int(i["bbox"][0]),
						"y":int(i["bbox"][1]),
						"w":int(i["bbox"][2]),
						"h":int(i["bbox"][3])
					} for i in instances]
	
			if self.randomZoom:
				img, iBoxes = BoxAwareRandZoom.randZoom(img, iBoxes, keepOriginalRatio=False, keepOriginalSize=False, keepBoxes=True)

			if self.normalizeSize:
				sizeMul = 640.0 / min(img.shape[0], img.shape[1])
				img = cv2.resize(img, (int(img.shape[1]*sizeMul), int(img.shape[0]*sizeMul)))

			m = img.shape[1] % 32
			if m != 0:
				padLeft = int(m/2)
				img = img[:,padLeft : padLeft + img.shape[1] - m]

			m = img.shape[0] % 32
			if m != 0:
				m = img.shape[0] % 32
				padTop = int(m/2)
				img = img[padTop : padTop + img.shape[0] - m]

			if img.shape[0]<256 or img.shape[1]<256:
				print("Warning: Image to small, skipping: "+str(img.shape))
				continue

			boxes=[]
			categories=[]
			for i in range(len(instances)):
				x1,y1,w,h = iBoxes[i]["x"],iBoxes[i]["y"],iBoxes[i]["w"],iBoxes[i]["h"]
				newBox=[int(x1*sizeMul) - padLeft, int(y1*sizeMul) - padTop, int((x1+w)*sizeMul) - padLeft, int((y1+h)*sizeMul) - padTop]
				newBox[0] = max(min(newBox[0], img.shape[1]),0)
				newBox[1] = max(min(newBox[1], img.shape[0]),0)
				newBox[2] = max(min(newBox[2], img.shape[1]),0)
				newBox[3] = max(min(newBox[3], img.shape[0]),0)

				if (newBox[2]-newBox[0]) >= 16 and (newBox[3]-newBox[1]) >= 16:
					boxes.append(newBox)
					categories.append(self.fromCocoCategory[instances[i]["category_id"]])

			if len(boxes)==0:
				print("Warning: No boxes on image. Skipping.")
				continue;

			boxes=np.array(boxes, dtype=np.float32)
			boxes=np.reshape(boxes, [-1,4])
			categories=np.array(categories, dtype=np.uint8)

			return img, boxes, categories

	def count(self):
		return len(self.images)
