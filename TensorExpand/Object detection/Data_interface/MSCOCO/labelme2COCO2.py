# -*- coding:utf-8 -*-

import json
import cv2
import numpy as np

labelme_json='./1.json'
data=json.load(open(labelme_json))

data_coco={}

# images
images=[]
image={}
file_name=data['imagePath'].split('/')[-1] # windows \\ ;linux /
image['file_name']=file_name
image['id']=0 # 每张图片对应的id都是唯一的

# img=cv2.imread(data['imagePath'])
img=cv2.imread('./1.jpg')
image['height']=img.shape[0]
image['width']=img.shape[1]
img=None

images.append(image)

data_coco['images']=images

# categories
categories=[]

categorie={}
categorie['supercategory']='cat'
categorie['id']=1 # id 唯一 0 默认为背景
categorie['name']='persian cat' # 波斯猫
categories.append(categorie)

categorie={}
categorie['supercategory']='cat'
categorie['id']=2
categorie['name']='garden Cat' # 田园猫
categories.append(categorie)

data_coco['categories']=categories

# annotations
annotations=[]
annotation={}

annotation['segmentation']=[list(np.asarray(data['shapes'][0]['points']).flatten())]   # data['shapes'][0]['points']
annotation['iscrowd']=0
annotation['image_id']=image['id']
annotation['bbox']=[] # 先空着，需要反算出定位框
annotation['category_id']=1
annotation['id']=1 # 第一个对象 这个ID也不能重复，如果下一张图，id不能取1，需从1 开始往下取
annotations.append(annotation)

annotation={}
annotation['segmentation']=[list(np.asarray(data['shapes'][1]['points']).flatten())]
annotation['iscrowd']=0
annotation['image_id']=image['id']
annotation['bbox']=[] # 先空着，需要反算出定位框
annotation['category_id']=2
annotation['id']=2 # 第一个对象 这个ID也不能重复，如果下一张图，id不能取1，需从1 开始往下取
annotations.append(annotation)

data_coco['annotations']=annotations

# 保存json文件
json.dump(data_coco,open('./new_instances_val2017.json','w'),indent=4) # indent=4 更加美观显示