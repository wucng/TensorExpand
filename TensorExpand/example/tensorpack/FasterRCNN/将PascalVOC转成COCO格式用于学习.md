<font size=4>

[toc]

# 将PascalVOC转成COCO格式用于学习
由于COCO 数据集太大很难下载，因此手动生成一些COCO格式的数据用于代码测试学习

# Pascal VOC 2007数据下载

```
# 共20个类别
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
# 将Pascal VOC转成COCO格式
使用下面的脚本生成`instances_train2014.json`
```python
# -*- coding:utf-8 -*-
# !/usr/bin/env python
# file name：PascalVOC2COCO.py
'''
https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/Object%20detection/Data_interface/MSCOCO/Pascal%20VOC/PascalVOC2COCO.py
'''

import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
# from labelme import utils
import numpy as np
import glob
import PIL.Image
import os,sys


class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.xml):

            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()

            self.json_file = json_file
            self.num = num
            path = os.path.dirname(self.json_file)
            path = os.path.dirname(path)
            # path=os.path.split(self.json_file)[0]
            # path=os.path.split(path)[0]
            obj_path = glob.glob(os.path.join(path, 'SegmentationObject', '*.png'))
            with open(json_file, 'r') as fp:
                for p in fp:
                    # if 'folder' in p:
                    #     folder =p.split('>')[1].split('<')[0]
                    if 'filename' in p:
                        self.filen_ame = p.split('>')[1].split('<')[0]

                        self.path = os.path.join(path, 'SegmentationObject', self.filen_ame.split('.')[0] + '.png')
                        if self.path not in obj_path:
                            break


                    if 'width' in p:
                        self.width = int(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])

                        self.images.append(self.image())

                    if '<object>' in p:
                        # 类别
                        d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                        self.supercategory = d[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)

                        # 边界框
                        x1 = int(d[-4]);
                        y1 = int(d[-3]);
                        x2 = int(d[-2]);
                        y2 = int(d[-1])
                        self.rectangle = [x1, y1, x2, y2]
                        self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]

                        self.annotations.append(self.annotation())
                        self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie

    @staticmethod
    def change_format(contour):
        contour2 = []
        length = len(contour)
        for i in range(0, length, 2):
            contour2.append([contour[i], contour[i + 1]])
        return np.asarray(contour2, np.int32)

    def annotation(self):
        annotation = {}
        # annotation['segmentation'] = [self.getsegmentation()]
        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        # annotation['bbox'] = list(map(float, self.bbox))
        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID

        # 计算轮廓面积
        contour = PascalVOC2coco.change_format(annotation['segmentation'][0])
        annotation['area']=abs(cv2.contourArea(contour,True))
        
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i;
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i;
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        '''从mask提取边界点'''
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox=[]
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox # list(contours[1][0].flatten())

    # '''
    def getbbox(self, points):
        '''边界点生成mask，从mask提取定位框'''
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        '''边界点生成mask'''
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    # '''
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


xml_file = glob.glob('./Annotations/*.xml')
# xml_file=['./Annotations/000032.xml']

PascalVOC2coco(xml_file, './instances_train2014.json')
```

# 根据上面生成的json复制出用到的图片
使用下面的脚本生成`train2004`文件佳
```python
# -*- coding:utf-8 -*-
# file name：copy_image.py

import json
import shutil,os

json_file='instances_train2014.json'
data=json.load(open(json_file))

# 图片名
img_name=[]
for img in data['images']:
    img_name.append(img['file_name'])

# mkdir train2014
for img in img_name:
    shutil.copy(os.path.join('./JPEGImages',img),'./train2014')

# print(len(img_name))
```
# 创建完整的COCO目录
目录格式如下：

```
mkdir COCO

<COCO>
├──minival2014
├──test2014
├──train2014
├──val2014
├──valminusminival2014
└──annotations

<COCO/annotations>
├── instances_minival2014.json
├── instances_test2014.json
├── instances_train2014.json
├── instances_val2014.json
└── instances_valminusminival2014.json

<COCO/*2014> (*为minival、test、train、val、valminusminival）
└── *.jpg
```
用于测试代码，将上面生成的`instances_train2014.json`，分别复制成其它的`instances_*2014.json *为(minival,test,val,valminusminival)`放至`COCO/annotations`目录

将上面的`train2014`分别复制`*2014 (*为minival、test、train、val、valminusminival）`放至`COCO`目录

# cocoDataFlow

```python
# -*- coding:utf-8 -*-
# file name:cocoDataflow.py

'''
将COCO格式数据转成以下格式，其中每一个{},代表一张图片的信息

[{'height': 281, 'boxes': array([[104.,  78., 375., 183.],[133.,  88., 197., 123.],[195., 180., 213., 229.],[ 26., 189.,  44., 238.]], dtype=float32), 
'segmentation': [[array([[234., 106.],[232., 108.],[230., 108.],[228., 110.],[259., 116.],[258., 115.]], dtype=float32)], [array([[191.,  88.],
[190.,  89.],[195.,  89.],[194.,  88.]], dtype=float32)], [array([[207., 181.],[206., 182.],[206., 185.],[207., 186.],[210., 186.],[202., 180.]], dtype=float32)], 
[array([[ 29., 189.],[ 34., 189.],[ 33., 190.],[ 32., 189.]], dtype=float32)]], 
'id': 14, 'width': 500, 'file_name': '/media/wucong/data2t/Exercise/test/FasterRCNN/data/COCO/train2014/000032.jpg', 
'is_crowd': array([0, 0, 0, 0], dtype=int8), 'class': array([1, 1, 2, 2], dtype=int32)},{}]
'''


import pickle
import os
import cv2
import numpy as np
import json
from tensorpack import *

file_name='train2014'
# json_path='./data/COCO/annotations/%s.json'%(file_name)
json_path='./data/COCO/annotations'

def cocoDataFlow(json_path,file_name,add_gt=True,add_mask=False):
    json_path=os.path.join(os.path.abspath(json_path),'annotations/instances_{}.json'.format(file_name))
    # save_json_path = os.path.join(os.path.dirname(json_path), file_name + '.json')
    data = json.load(open(json_path, 'r'))

    # 解析"categories"
    categories = {}
    for item in data["categories"]:
        categories.update({item['id']: item['name']})

    # 解析"images"
    img_path = os.path.dirname(os.path.dirname(json_path))
    images = {}
    for item in data["images"]:
        images.update({item['id']: {'width': item["width"], 'height': item['height'], "file_name":
            os.path.join(img_path, '{}'.format(file_name), item['file_name']), 'boxes': [],
                                    'segmentation': [], 'is_crowd': [], 'class': [], 'id': item['id']}})

    def change_format(contour):
        contour2 = []
        length = len(contour)
        for i in range(0, length, 2):
            contour2.append([contour[i], contour[i + 1]])
        return np.asarray(contour2, np.float32)

    # 解析 "annotations"
    for item in data['annotations']:
        image_dic = images[item['image_id']]
        image_dic['area'] = item['area']
        image_dic['is_crowd'].append(item['iscrowd'])
        image_dic['class'].append(item['category_id'])
        if add_mask:
            add_gt=True
            image_dic['segmentation'].append([change_format(item['segmentation'][0])])
        if add_gt:
            image_dic['boxes'].append(np.asarray(item['bbox'], np.float32))

    image_list = []
    for k, v in images.items():
        v['boxes']=np.asarray(v['boxes'],np.float32)
        v['is_crowd']=np.asarray(v['is_crowd'],np.int8)
        v['class']=np.asarray(v['class'],np.int32)
        image_list.append(v)

    # json.dump(image_list, open(save_json_path, 'w'), indent=4)  # indent=4 更加美观显示

    # pickle.dump(image_list, open(save_json_path, 'wb'))

    # return image_list

    ds = DataFromList(image_list, shuffle=True)

    return ds
```
