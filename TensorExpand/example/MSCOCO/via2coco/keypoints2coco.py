# -*- coding:utf-8 -*-
# filename: keypoints2coco.py
# time:2018.08.21
# author：Mr.wu
# description:使用Via标注数据集，转成coco数据格式

"""只需要按照实际改写images，annotations，categories另外两个字段其实可以忽略
在keypoints，categories内容是固定的不需修改
"""

import json
from tqdm import tqdm
import cv2
import os
import numpy as np
import PIL

class COCO(object):
    def info(self):
        return {"description":"Via custom data",
                "url":"http://www.robots.ox.ac.uk/~vgg/software/via/",
                "version":"1.0",
                "year":2018,
                "contributor":"Mr.wu",
                "date_created":"2018/08/21",
                "github":"https://github.com/wucng/",
                "bolg":"https://blog.csdn.net/wc781708249/article/details/79603522"}
    def licenses(self):
        return [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "name": "Attribution-NonCommercial-ShareAlike License",
            "id": 1
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "name": "Attribution-NonCommercial License",
            "id": 2
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
            "name": "Attribution-NonCommercial-NoDerivs License",
            "id": 3
        },
        {
            "url": "http://creativecommons.org/licenses/by/2.0/",
            "name": "Attribution License",
            "id": 4
        },
        {
            "url": "http://creativecommons.org/licenses/by-sa/2.0/",
            "name": "Attribution-ShareAlike License",
            "id": 5
        },
        {
            "url": "http://creativecommons.org/licenses/by-nd/2.0/",
            "name": "Attribution-NoDerivs License",
            "id": 6
        },
        {
            "url": "http://flickr.com/commons/usage/",
            "name": "No known copyright restrictions",
            "id": 7
        },
        {
            "url": "http://www.usa.gov/copyright.shtml",
            "name": "United States Government Work",
            "id": 8
        }
    ]
    def image(self):
        return {
            "license": 4,
            "file_name": "000000397133.jpg", # 图片名
            "coco_url":  "http://images.cocodataset.org/val2017/000000397133.jpg",# 网路地址路径
            "height": 427, # 高
            "width": 640, # 宽
            "date_captured": "2013-11-14 17:02:52", # 数据获取日期
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",# flickr网路地址
            "id": 397133 # 图片的ID编号（每张图片ID是唯一的）
        }

    def annotation(self):
        return {
            "segmentation": [ # 对象的边界点（边界多边形）
                [
                    224.24,297.18,# 第一个点 x,y坐标
                    228.29,297.18, # 第二个点 x,y坐标
                    234.91,298.29,
                    225.34,297.55
                ]
            ],
            "num_keypoints": 10,
            # keypoints是按照以下关键点来标记的，如果nose 没有标则为0,0,0(3个数字为一组，分别为x,y,v v=0表示为标记此时的x=y=0,
            # v=1表示标记了但是在图上是不可见，v=2表示标记了，在图上可见)
            # "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow",
            # "right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
            "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "area": 1481.3806499999994, # 区域面积
            "iscrowd": 0, #
            "image_id": 397133, # 对应的图片ID（与images中的ID对应）
            "bbox": [217.62,240.54,38.99,57.75], # 定位边框 [x,y,w,h]
            "category_id": 44, # 类别ID（与categories中的ID对应）
            "id": 82445 # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号（每个对象的ID是唯一的）
            }

    def categorie(self):
        return {
                "supercategory": "person", # 主类别
                "id": 1, # 类对应的id （0 默认为背景）
                "name": "person", # 子类别
                "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                              "left_knee", "right_knee", "left_ankle", "right_ankle"],
                "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                             [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
                }

def cocoPolygons(all_points_x,all_points_y):
    """
    :param all_points_x:list
    :param all_points_y: list
    :return: [[x1,y1,x2,y2,....]]
    """
    all_xy=[]
    for x,y in zip(all_points_x,all_points_y):
        all_xy.extend([x,y])

    return [all_xy]

def ploygons2box(all_points_x,all_points_y):
    xs=np.asarray(all_points_x)
    ys=np.asarray(all_points_y)

    x_min=np.min(xs)
    x_max=np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    # 左上角坐标
    x1,y1=x_min,y_min
    # 右下角坐标
    x2, y2 = x_max, y_max

    return [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]

def changePolygons(all_points_x,all_points_y):
    """
    :param all_points_x:list
    :param all_points_y: list
    :return: [[x1,y1],[x2,y2],....]
    """
    all_xy=[]
    for x,y in zip(all_points_x,all_points_y):
        all_xy.append([x,y])

    return np.asarray(all_xy, np.float32)

class Keypoints2COCO(COCO):
    def __init__(self,jsonfile,save_json_path,images_path):
        with open(jsonfile, 'r') as fp:
            self.data = json.load(fp)
        self.save_json_path = save_json_path # 最终保存的json文件
        self.images_path=images_path # 原始图片保存的位置
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.num=1
        self.keypoints=["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                          "left_knee", "right_knee", "left_ankle", "right_ankle"]
        self.num_keypoints=0

    def __call__(self):
        for key in tqdm(self.data.keys()):
            image=self.image()
            # annotation=self.annotation()
            # categorie=self.categorie()
            for k in self.data[key].keys():
                if k=="filename":
                    image["file_name"]=self.data[key][k]
                    image["id"]=self.num
                    shape=cv2.imread(os.path.join(self.images_path,image["file_name"]),0).shape[:2]
                    image["height"]=shape[0]
                    image["width"]=shape[1]

                if k=="regions":
                    annotation = self.annotation()
                    for da in self.data[key][k]: # list
                        if da["region_attributes"]["person"] in self.keypoints:
                            index = self.keypoints.index(da["region_attributes"]["person"])
                            x=da["shape_attributes"]["cx"]
                            y=da["shape_attributes"]["cy"]

                            annotation['keypoints'][index*3:(index+1)*3]=[x,y,2] # 默认为可见 v=2
                            self.num_keypoints+=1

                        else:
                            k2 = "shape_attributes"
                            all_points_x = da[k2]["all_points_x"]
                            all_points_y = da[k2]["all_points_y"]
                            annotation["segmentation"] = cocoPolygons(all_points_x, all_points_y)
                            annotation["image_id"] = self.num
                            annotation["id"] = self.annID
                            annotation["bbox"] = ploygons2box(all_points_x, all_points_y)
                            annotation["area"] = abs(cv2.contourArea(changePolygons(all_points_x, all_points_y), True))
                            annotation["category_id"] = 1
                            annotation['num_keypoints']=self.num_keypoints

                            self.annID += 1  # 对应对象
                            self.annotations.append(annotation)
                            # 到下一个对象
                            self.num_keypoints=0
                            annotation = self.annotation() # 计算下一个对象

            self.images.append(image)
            # self.categories.append(categorie)
            # self.annotations.append(annotation)
            self.num+=1 # 对应图像

        jsdata={"info":self.info(),"licenses":self.licenses(),"images":self.images,
                "categories":[self.categorie()],"annotations":self.annotations}
        json.dump(jsdata,open(self.save_json_path,'w'),default=float) # python3 需加上default=float 否则会报错

Keypoints2COCO('via_keypoints.json','via_keypoints_coco.json','./images')()
