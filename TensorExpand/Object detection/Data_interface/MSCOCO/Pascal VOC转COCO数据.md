参考：

-  [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
- [philferriere/cocoapi](https://github.com/philferriere/cocoapi)- support Windows build and python3
- [COCO 标注详解](http://blog.csdn.net/yeyang911/article/details/78675942)
-  [labelme标注的数据分析](http://blog.csdn.net/wc781708249/article/details/79595174) 
-  [MSCOCO数据标注详解](http://blog.csdn.net/wc781708249/article/details/79603522) 
- [labelme数据转成COCO数据](http://blog.csdn.net/wc781708249/article/details/79611536) 


----------
详细代码点击这里

----------
# Pascal VOC 2007数据下载

```python
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

```python
# 执行以下命令将解压到一个名为VOCdevkit的目录中
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```


----------
## 数据预览
1、VOC2007/Annotations

类别名与对象的矩形框位置

```xml
<annotation>
	<folder>VOC2007</folder> # 所在文件夹
	<filename>000032.jpg</filename> # 对应的图片名
	<source>
		<database>The VOC2007 Database</database> # 数据集
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>311023000</flickrid>
	</source>
	<owner>
		<flickrid>-hi-no-to-ri-mo-rt-al-</flickrid>
		<name>?</name>
	</owner>
	<size>
		<width>500</width> # 图片宽
		<height>281</height> # 图片高
		<depth>3</depth> # 图片 通道数
	</size>
	<segmented>1</segmented> 
	<object>
		<name>aeroplane</name> # 对象名
		<pose>Frontal</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>104</xmin> # 对象框左上角 x
			<ymin>78</ymin> # 对象框左上角 y
			<xmax>375</xmax> # 对象框右下角 x
			<ymax>183</ymax> # 对象框右下角 y
		</bndbox>
	</object>
	<object>
		<name>aeroplane</name> # 对象名
		<pose>Left</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>133</xmin>
			<ymin>88</ymin>
			<xmax>197</xmax>
			<ymax>123</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Rear</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>195</xmin>
			<ymin>180</ymin>
			<xmax>213</xmax>
			<ymax>229</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Rear</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>26</xmin>
			<ymin>189</ymin>
			<xmax>44</xmax>
			<ymax>238</ymax>
		</bndbox>
	</object>
</annotation>
```

2、VOC2007/JPEGImages

![这里写图片描述](http://img.blog.csdn.net/20180312111000760?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

3、VOC2007/SegmentationClass

![这里写图片描述](http://img.blog.csdn.net/20180312111028031?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

4、VOC2007/SegmentationObject

![这里写图片描述](http://img.blog.csdn.net/20180312111039831?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 将Pascal VOC转COCO的JSON数据
- XML数据解析参考[这里](http://blog.csdn.net/wc781708249/article/details/79542655#t3)

- [labelme数据转成COCO数据](http://blog.csdn.net/wc781708249/article/details/79611536) 


----------
参考 [labelme数据转成COCO数据](http://blog.csdn.net/wc781708249/article/details/79611536) 

通过解析xml文件得到images、categories、annotations字段中的内容
详情参考：`PascalVOC2COCO.py`

```python
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
        return annotation
```

# 可视化结果

参考：`visualization.py`

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

annFile='./new.json'
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
dataType = 'JPEGImages'
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
```

![这里写图片描述](http://img.blog.csdn.net/20180319175124155?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180319175133475?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 存在的问题
- 1、从SegmentationObject文件夹中提取对象的mask效果不好
- 2、从mask反算出mask边界点上的坐标，使用opencv提取对象边缘，感觉效果不好

上面问题导致`annotation['segmentation']` 得到的点不是很准确
