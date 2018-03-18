参考：

- [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
- [philferriere/cocoapi](https://github.com/philferriere/cocoapi)- support Windows build and python3
- [COCO 标注详解](http://blog.csdn.net/yeyang911/article/details/78675942)
- [COCO数据集annotation内容](http://blog.csdn.net/qq_30401249/article/details/72636414)
- [Dataset - COCO Dataset 数据特点](http://blog.csdn.net/zziahgf/article/details/72819043)


----------
[toc]

----------
# JSON文件
json文件主要包含以下几个字段：
详细描述参考 [COCO 标注详解](http://blog.csdn.net/yeyang911/article/details/78675942)
```python
{
    "info": info, # dict
    "licenses": [license], # list ，内部是dict
    "images": [image], # list ，内部是dict
    "annotations": [annotation], # list ，内部是dict
    "categories": # list ，内部是dict
}
```

# 打开JSON文件查看数据特点

由于JSON文件太大，很多都是重复定义的，所以只提取一张图片，存储成新的JSON文件，便于观察。

```python
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
import json

json_file='./annotations/instances_val2017.json' # # Object Instance 类型的标注
# person_keypoints_val2017.json  # Object Keypoint 类型的标注格式
# captions_val2017.json  # Image Caption的标注格式

data=json.load(open(json_file,'r'))

data_2={}
data_2['info']=data['info']
data_2['licenses']=data['licenses']
data_2['images']=[data['images'][0]] # 只提取第一张图片
data_2['categories']=data['categories']
annotation=[]

# 通过imgID 找到其所有对象
imgID=data_2['images'][0]['id']
for ann in data['annotations']:
    if ann['image_id']==imgID:
        annotation.append(ann)

data_2['annotations']=annotation

# 保存到新的JSON文件，便于查看数据特点
json.dump(data_2,open('./new_instances_val2017.json','w'),indent=4) # indent=4 更加美观显示
```

# Object Instance 类型的标注格式
主要有以下几个字段：

![这里写图片描述](http://img.blog.csdn.net/20180318201116341?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## info

```python
"info": { # 数据集信息描述
        "description": "COCO 2017 Dataset", # 数据集描述
        "url": "http://cocodataset.org", # 下载地址
        "version": "1.0", # 版本
        "year": 2017, # 年份
        "contributor": "COCO Consortium", # 提供者
        "date_created": "2017/09/01" # 数据创建日期
    },
```
## licenses

```python
"licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        ……
        ……
    ],
```
## images

```python
"images": [
        {
            "license": 4,
            "file_name": "000000397133.jpg", # 图片名
            "coco_url":  "http://images.cocodataset.org/val2017/000000397133.jpg",# 网路地址路径
            "height": 427, # 高
            "width": 640, # 宽
            "date_captured": "2013-11-14 17:02:52", # 数据获取日期
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",# flickr网路地址
            "id": 397133 # 图片的ID编号
        },
        ……
        ……
    ],
```
## categories

```
"categories": [ # 类别描述
        {
            "supercategory": "person", # 主类别
            "id": 1, # 类对应的id （0 默认为背景）
            "name": "person" # 子类别
        },
        {
            "supercategory": "vehicle", 
            "id": 2,
            "name": "bicycle"
        },
        {
            "supercategory": "vehicle",
            "id": 3,
            "name": "car"
        },
        ……
        ……
    ],
```
<font size=5 color=#FF00FF>**注：** bicycle 与car都属于vehicle，但两者又属于不同的类别。例如：羊（主类别）分为山羊、绵羊、藏羚羊（子类别）等</font>

## annotations

```python
"annotation": [
        {
            "segmentation": [ # 对象的边界点（边界多边形）
                [
                    224.24,297.18,# 第一个点 x,y坐标
                    228.29,297.18, # 第二个点 x,y坐标
                    234.91,298.29,
                    ……
                    ……
                    225.34,297.55
                ]
            ],
            "area": 1481.3806499999994, # 区域面积
            "iscrowd": 0, # 
            "image_id": 397133, # 对应的图片ID（与images中的ID对应）
            "bbox": [217.62,240.54,38.99,57.75], # 定位边框 [x,y,w,h]
            "category_id": 44, # 类别ID（与categories中的ID对应）
            "id": 82445 # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号
        },
        ……
        ……
        ]
```
<font size=5 color=##00DD00>注意，单个的对象（iscrowd=0)可能需要多个polygon来表示，比如这个对象在图像中被挡住了。而iscrowd=1时（将标注一组对象，比如一群人）的segmentation使用的就是RLE格式。</font>


----------
## 可视化
现在调用`cocoapi`显示刚生成的JSON文件，并检查是否有问题。

```python
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

annFile='./new_instances_val2017.json'
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# imgIds = coco.getImgIds(imgIds = [324158])
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])[0]
dataDir = '.'
dataType = 'val2017'
I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))

plt.axis('off')
plt.imshow(I)
plt.show()


# load and display instance annotations
# 加载实例掩膜
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# catIds=coco.getCatIds()
catIds=[]
for ann in coco.dataset['annotations']:
    if ann['image_id']==imgIds[0]:
        catIds.append(ann['category_id'])

plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# load and display keypoints annotations
# 加载肢体关键点
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)

# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

# load and display caption annotations
# 加载文本描述
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()


```
![这里写图片描述](http://img.blog.csdn.net/20180318193443036?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```python
A man is in a kitchen making pizzas.
Man in apron standing on front of oven with pans and bakeware
A baker is working in the kitchen rolling dough.
A person standing by a stove in a kitchen.
A table with pies being made and a person standing near a wall with pots and pans hanging on the wall.
```


----------

## 仿照COCO JSON文件
仿照COCO的数据格式，将[labelme的JSON](http://blog.csdn.net/wc781708249/article/details/79595174)改造成COCO的JSON