
# -*- coding:utf-8 -*-

from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

annFile='./via_keypoints_coco.json'
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# imgIds = coco.getImgIds(imgIds = [324158])
imgIds = coco.getImgIds()
imgId=np.random.randint(0,len(imgIds))
img = coco.loadImgs(imgIds[imgId])[0]
dataDir = '.'
dataType = 'images'
I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
# I = io.imread('%s/%s'%(dataDir,img['file_name']))

plt.axis('off')
plt.imshow(I)
plt.show()


# load and display instance annotations
# 加载实例掩膜
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# catIds=coco.getCatIds()
catIds=[]
for ann in coco.dataset['annotations']:
    if ann['image_id']==imgIds[imgId]:
        catIds.append(ann['category_id'])

plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()


# initialize COCO api for person keypoints annotations
# annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)
# load and display keypoints annotations
# 加载肢体关键点
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
'''
# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)
# load and display caption annotations
# 加载文本描述
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()
# '''