# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
# from labelme import utils
import numpy as np
import glob
import PIL.Image
import PIL.ImageDraw
import os,sys

parser=argparse.ArgumentParser()
parser.add_argument('-jf','--json_file',help="json file",type=str, default='new_instances_val2017.json')
args = parser.parse_args()

data=json.load(open(args.json_file))

height=data['images'][0]['height']
width=data['images'][0]['width']

segmentation=data['annotations'][0]['segmentation'] # 对象的边界点
# 注 从COCO JSON中获得的边界点格式[[x1,y1,x2,y2,x3,y3,...,xn,yn]]
# 从labelme JSON中获得的边界点格式[[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]


# 从边界点获得mask（边界多边形）
def polygons_to_mask(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式
    :return: 
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

# segmentation先转成labelme JSON 中的边界点格式
seg=segmentation[0]
polygons=[]
for i in range(0,len(seg),2):
    polygons.append([seg[i],seg[i+1]])

mask=polygons_to_mask([height,width],polygons)

# mask 转成0、1二值图片
mask=mask.astype(np.uint8)

plt.subplot(121)
plt.imshow(mask,'gray')
# plt.show()

# 从mask反算出定位框
def mask2box(mask):
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

    return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2] 对应Pascal VOC 2007 的bbox格式
    # return [left_top_c, left_top_r, right_bottom_c - left_top_c,
    #         right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

bbox=mask2box(mask)

# 使用opencv在mask图上画出
cv2.rectangle(mask,tuple(bbox[:2]),tuple(bbox[2:]),2)

plt.subplot(122)
plt.imshow(mask,'gray')
plt.show()

# mask反算polygons
def mask2polygons(mask):
    '''从mask提取边界点'''
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
    bbox=[]
    for cont in contours[1]:
        [bbox.append(i) for i in list(cont.flatten())]
        # map(bbox.append,list(cont.flatten()))
    return bbox # list(contours[1][0].flatten())