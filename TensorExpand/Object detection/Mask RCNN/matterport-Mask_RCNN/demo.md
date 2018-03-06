参考：

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [使用Keras和Tensorflow设置和安装Mask RCNN](http://blog.csdn.net/wc781708249/article/details/79438972)


----------
[demo.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/demo.ipynb)是最简单的开始。 它展示了一个使用MS COCO预先训练的模型来分割自己图像中的对象的例子。 它包括在任意图像上运行对象检测和实例分割的代码。

# Mask R-CNN Demo
快速介绍如何使用预先训练的模型来检测和分割对象。

```python
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

%matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
# 存放logs和训练模型的目录
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# 下载COCO权重
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# 需检查的图像目录
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
```

# 配置
我们将使用在MS-COCO数据集上训练的模型。 该模型的配置位于coco.py中的`CocoConfig`类中。

对于推理，稍微修改一下配置以适应任务。 为此，子类CocoConfig类并覆盖您需要更改的属性。

```python
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # 将批处理大小设置为1，因为我们将开始运行推理
    # 一次一个图像。 批量大小= GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
```
# 创建模型并加载训练后的权重

```python
# Create model object in inference mode.
# 在推理模式下创建模型对象。
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# 加载在MS-COCO上训练的重量
model.load_weights(COCO_MODEL_PATH, by_name=True)
```
# 类名称
该模型对对象进行分类并返回类ID，这是识别每个类的整数值。 有些数据集将整数值分配给它们的类，有些则不会。 例如，在MS-COCO数据集中，“人”类为1，“泰迪熊”为88.这些ID通常是连续的，但并非总是如此。 例如，COCO数据集具有与类别ID 70和72相关联的类别，但不具有类别71。

为了提高一致性，同时支持来自多个来源的数据的培训，我们的数据集类为每个类指定了自己的顺序整数ID。 例如，如果您使用我们的数据集类加载COCO数据集，那么'person'类将获得类ID = 1（就像COCO），'teddy bear'类是78（与COCO不同）。 在将类ID映射到类名时请记住这一点。

要获取类名称列表，您需要加载数据集，然后像这样使用class_names属性。

```python
# Load COCO dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
print(dataset.class_names)
```

我们**不想要求你下载COCO数据集**来运行这个演示，所以我们在下面列出了类名的列表。 列表中类名称的索引表示其ID（第一类为0，第二类为1，第三类为2，...等）。

```python
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# 列表中类的索引是它的ID。 例如，要获取ID
# teddy bear（泰迪熊）class，请使用：class_names.index（'teddy bear'）
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
```
# Run Object Detection

```python
# Load a random image from the images folder
# 从图像文件夹加载一个随机图像
file_names = next(os.walk(IMAGE_DIR))[2] # 获取图片文件
# next(os.walk(IMAGE_DIR)) 返回3个值，0为根目录，1为子文件夹，2为子文件（根目录下的所有图片文件）是个列表
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

'''
# 或者
import glob
file_paths=glob.glob('IMAGE_DIR/*')
image=skimage.io.imread(random.choice(file_paths))
'''

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
```
![这里写图片描述](http://img.blog.csdn.net/20180306091629013?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 完整代码

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

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

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

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

'''
# Load COCO dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
print(dataset.class_names)
'''
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

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
