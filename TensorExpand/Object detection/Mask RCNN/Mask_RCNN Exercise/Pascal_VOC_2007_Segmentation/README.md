参考：

- [使用Keras和Tensorflow设置和安装Mask RCNN](http://blog.csdn.net/wc781708249/article/details/79438972) 
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN) for object detection and instance segmentation on Keras and TensorFlow
- [MaskRCNN识别Pascal VOC 2007](http://blog.csdn.net/wc781708249/article/details/79542655) 使用非COCO数据格式
- [labelme数据转成COCO数据](http://blog.csdn.net/wc781708249/article/details/79611536) 
- [Pascal VOC转COCO数据](http://blog.csdn.net/wc781708249/article/details/79615210) 

----------
# MaskRCNN识别Pascal VOC 2007



------

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

## 数据预览
1、VOC2007/Annotations

类别名与对象的矩形框位置

2、VOC2007/JPEGImages

![这里写图片描述](http://img.blog.csdn.net/20180312111000760?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

3、VOC2007/SegmentationClass

![这里写图片描述](http://img.blog.csdn.net/20180312111028031?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

4、VOC2007/SegmentationObject

![这里写图片描述](http://img.blog.csdn.net/20180312111039831?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 将Pascal VOC2017转COCO数据 
- 参考[Pascal VOC转COCO数据](http://blog.csdn.net/wc781708249/article/details/79615210) 

----------

```python
# 新建文件夹存放数据
mkdir ./Mask_RCNN/COCO
```

参考[RFCN识别Pascal VOC 2007](http://blog.csdn.net/wc781708249/article/details/79624990#t6) 中的设置

这里只是为了演示方便

- 将`train2014` 复制`val2014` 都存放在`COCO`目录下
- 将`instances_train2014.json` 复制`instances_val2014.json`、`instances_valminusminival2014.json`、`instances_minival2014.json` 都存放在`COCO/annotations`目录

- 最终`./Mask_RCNN/COCO`目录结构如下

```python
<COCO>
├─  annotations
│     ├─  instances_train2014.json
|     |——  instances_val2014.json
│     |——  instances_valminusminival2014.json
|     └─  instances_minival2014.json
|
|─  train2014
|       └─ *.jpg
└─  val2014
       └─ *.jpg

```

----------


# 修改coco.py
- [coco.py](https://github.com/matterport/Mask_RCNN/blob/master/coco.py)


----------
- 改写class CocoConfig(Config)

参考[MaskRCNN识别Pascal VOC 2007](http://blog.csdn.net/wc781708249/article/details/79542655#t4) 

```python
class CocoConfig(Config):
    # 命名配置
    NAME = "coco"

    # 输入图像resing
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # 使用的GPU数量。 对于CPU训练，请使用1
    GPU_COUNT = 1

    IMAGES_PER_GPU = int(1024 * 1024 * 4 // (IMAGE_MAX_DIM * IMAGE_MAX_DIM * 12))+1

    batch_size = GPU_COUNT * IMAGES_PER_GPU
    # STEPS_PER_EPOCH = int(train_images / batch_size * (3 / 4))
    STEPS_PER_EPOCH=800

    VALIDATION_STEPS = STEPS_PER_EPOCH // (1000 // 50)
	
    NUM_CLASSES = 1 + 20  # 必须包含一个背景（背景作为一个类别） Pascal VOC 2007有20个类，前面加1 表示加上背景

    scale = 1024 // IMAGE_MAX_DIM
    RPN_ANCHOR_SCALES = (32 // scale, 64 // scale, 128 // scale, 256 // scale, 512 // scale)  # anchor side in pixels

    RPN_NMS_THRESHOLD = 0.6  # 0.6

    RPN_TRAIN_ANCHORS_PER_IMAGE = 256 // scale

    MINI_MASK_SHAPE = (56 // scale, 56 // scale)

    TRAIN_ROIS_PER_IMAGE = 200 // scale

    DETECTION_MAX_INSTANCES = 100 * scale * 2 // 3

    DETECTION_MIN_CONFIDENCE = 0.6
```

# train
```python
# 这里 --dataset=./COCO  数据路径
# Train a new model starting from pre-trained COCO weights
python3 coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 coco.py train --dataset=/path/to/coco/ --model=last

# You can also run the COCO evaluation code with
# Run COCO evaluation on the last trained model
python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
```


```python
# （可选）如果imagenet模型参数下载不了 可以按下面的方式修改
# 使用 --model=imagenet 需先下载下面的模型 到 ./Mask_RCNN/ 目录下
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

# 再修改coco.py 463行
model_path = model.get_imagenet_weights()
# 改成
model_path = os.path.join(ROOT_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
```
## 运行以下命令
```python
python3 coco.py train --dataset=./COCO --model=imagenet
```

<font size=5 color=##00DD00>注：</font>执行 `python3 coco.py train --dataset=./COCO --model=coco` 会报错，因为我是使用Pascal VOC 2007转成COCO格式数据，而不是真实的COCO数据，`--model=coco` 是按原本的COCO数据得到的模型参数

# evaluation 

```python
# Run COCO evaluation on the last trained model
python3 coco.py evaluate --dataset=./COCO --model=last --limit=2 # 只评估2张，为了节省时间
# 这里缺失'area'内容会报错 因为Pascal VOC 2007在转COCO数据时 没有加上每个对象的 area，所以要想评估转的时候需加上每个对象的'area' 即对象掩膜的面积（并不影响模型训练与推理）
```

# 推理

参考 [demo.md](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/Object%20detection/Mask%20RCNN/matterport-Mask_RCNN/demo.md)

```python
# -*- coding:utf8 -*-

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

# %matplotlib inline

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model_path = model.find_last()[1]
print("Loading weights ", model_path)
model.load_weights(model_path, by_name=True)

# 改成Pascal VOC 2007实际类别，按顺序列出
# 参考 http://blog.csdn.net/wc781708249/article/details/79624990#t9
class_names=['BG','diningtable', 'person', 'bottle', 'boat', 'train', 'bird', 'dog', 'cat', 'tvmonitor',
             'cow', 'car', 'sofa', 'horse', 'chair', 'pottedplant', 'bicycle', 'motorbike', 'aeroplane',
             'sheep', 'bus'] # 第一个默认为背景


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
```

直接执行该脚本即可。

![这里写图片描述](http://img.blog.csdn.net/20180320171857129?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

<font size=5 color=#FF00FF>注： 由于运行的时间不长，因此精度并不高，可以自行尝试多运行一段时间，调试参数</font>

