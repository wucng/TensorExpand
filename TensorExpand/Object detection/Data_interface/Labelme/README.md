参考：

- github地址：https://github.com/wkentaro/labelme 


----------

# 安装

安装方式：详情参考[官网](https://github.com/wkentaro/labelme)安装

```python
# Ubuntu 14.04
sudo apt-get install python-qt4 pyqt4-dev-tools
sudo pip install labelme # python2 works

# Ubuntu 16.04
sudo apt-get install python-qt5 pyqt5-dev-tools
sudo pip3 install labelme
```

----------
# 启动命令 

终端或cmd输入`labelme` 开始标记，标记完成后保存得到一个json文件。

![这里写图片描述](http://img.blog.csdn.net/20180308085133453?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

注：每个对象对应一个mask（图中2个对象，对应2个mask）,左边的猫标记为`cat_1`，右边的标记为`cat_2`

# 分析json文件
首先看看标记完成的json文件长什么样子。

```python
{
  "imageData": "something too long", # 原图像数据 通过该字段可以解析出原图像数据
  "shapes": [ # 每个对象的形状
    { # 第一个对象
      "points": [ # 边缘是由点构成，将这些点连在一起就是对象的边缘多边形
        [
          165.90909090909093, # 第一个点 x 坐标
          36.909090909090935  # 第一个点 y 坐标
        ],
        ……
        [
          240.90909090909093,
          15.909090909090935
        ],
        [
          216.90909090909093, # 最后一个点 会闭合到第一个点完成封闭的多边形 x 坐标
          31.909090909090935 # y 坐标
        ]
      ],
      "fill_color": null,
      "label": "cat_1",  # 第一个对象的标签
      "line_color": null
    },
    {  # 第二个对象
      "points": [
        [
          280.90909090909093,
          31.909090909090935
        ],
        ……
        [
          362.90909090909093,
          20.909090909090935
        ],
        [
          339.90909090909093,
          32.909090909090935
        ]
      ],
      "fill_color": null,
      "label": "cat_2", # 第二个对象的标签
      "line_color": null
    }
  ],
  "fillColor": [
    255,
    0,
    0,
    128
  ],
  "imagePath": "/home/wu/1.jpg", # 原始图片的路径
  "lineColor": [
    0,
    255,
    0,
    128
  ]
}
```

# 通过json文件提取信息
通过解析json文件提取每个对象的**边界多边形（mask）**，**labels**，以及原图像的**地址路径**

参考：[labelme_draw_json](https://github.com/wkentaro/labelme/blob/master/scripts/labelme_draw_json)

```python
# -*- coding:utf-8 -*-
#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt

from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    json_file = args.json_file

    data = json.load(open(json_file)) # 加载json文件

    img = utils.img_b64_to_array(data['imageData']) # 解析原图片数据
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes']) # 解析'shapes'中的字段信息，解析出每个对象的mask与对应的label   lbl存储 mask，lbl_names 存储对应的label
    # lal 像素取值 0、1、2 其中0对应背景，1对应第一个对象，2对应第二个对象
    # 使用该方法取出每个对象的mask mask=[] mask.append((lbl==1).astype(np.uint8)) # 解析出像素值为1的对象，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
    # lbl_names  ['background','cat_1','cat_2']

    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw_label(lbl, img, captions)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl_viz)
    plt.show()


if __name__ == '__main__':
    main()
 
''' 其他
data['imageData'] # 原图数据 str
data['shapes'] # 每个对像mask及label  list
len(data['shapes']) # 返回对象个数 int
data['shapes'][0]['label'] # 返回第一个对象的标签 str
data['shapes'][0]['points'] # 返回第一个对象的边界点 list
data['shapes'][0]['points'][0] # 返回第一个对象的边界点第一个点 list

data['imagePath'] # 原图路径 str
data['fillColor'] # 填充颜色（边界内部） list
data['lineColor'] # 边界线颜色  list
'''
```

![这里写图片描述](http://img.blog.csdn.net/20180317113223356?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```python
#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    json_file = args.json_file

    data = json.load(open(json_file))

    img = utils.img_b64_to_array(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw_label(lbl, img, captions)

    # lbl_names[0] 默认为背景，对应的像素值为0
    # 解析图片中的对象 像素值不为0（0 对应背景）
    mask=[]
    class_id=[]
    for i in range(1,len(lbl_names)): # 跳过第一个class（默认为背景）
        mask.append((lbl==i).astype(np.uint8)) # 解析出像素值为1的对应，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
        class_id.append(i) # mask与clas 一一对应

    mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0]) # 转成[h,w,instance count]
    class_id=np.asarray(class_id,np.uint8) # [instance count,]
    class_name=lbl_names[1:] # 不需要包含背景

    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(lbl_viz)

    plt.subplot(223)
    plt.imshow(mask[:,:,0],'gray')
    plt.title(class_name[0]+'\n id '+str(class_id[0]))
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(mask[:,:,1],'gray')
    plt.title(class_name[1] + '\n id ' + str(class_id[1]))
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
```
![这里写图片描述](http://img.blog.csdn.net/20180308102447751?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 模仿labelme的json文件
仿照lablme的json文件改写自己的数据，然后便可以调用labelme的数据接口解析自己的数据

```python
# -*- coding:utf-8 -*-

'''
仿照labelme的json文件写入自己的数据
'''
import cv2
import json

# json_file = './1.json'

# data = json.load(open(json_file))

# 参考labelme的json格式重新生成json文件，
# 便可以使用labelme的接口解析数据

def dict_json(imageData,shapes,imagePath,fillColor=None,lineColor=None):
    '''

    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"imageData":imageData,"shapes":shapes,"fillColor":fillColor,
            'imagePath':imagePath,'lineColor':lineColor}

def dict_shapes(points,label,fill_color=None,line_color=None):
    return {'points':points,'fill_color':fill_color,'label':label,'line_color':line_color}

# 注以下都是虚拟数据，仅为了说明问题
imageData="image data"
shapes=[]
# 第一个对象
points=[[10,10],[120,10],[120,120],[10,120]] # 数据模拟
# fill_color=null
label='cat_1'
# line_color=null
shapes.append(dict_shapes(points,label))

# 第二个对象
points=[[150,200],[200,200],[200,250],[150,250]] # 数据模拟
label='cat_2'
shapes.append(dict_shapes(points,label))

fillColor=[255,0,0,128]

imagePath='E:/practice/1.jpg'

lineColor=[0,255,0,128]

data=dict_json(imageData,shapes,imagePath,fillColor,lineColor)

# 写入json文件
json_file = 'E:/practice/2.json'
json.dump(data,open(json_file,'w'))
```
生成的json文件便可以使用labelme提供的接口解析。