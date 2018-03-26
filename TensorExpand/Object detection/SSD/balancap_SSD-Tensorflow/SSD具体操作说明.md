参考：

- [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)
- [pascalvoc_to_tfrecords.py](https://github.com/balancap/SSD-Tensorflow/blob/master/datasets/pascalvoc_to_tfrecords.py)


----------
# pascalvoc_to_tfrecords

使用Example protos将Pascal VOC数据转换为TFRecords文件格式。

Pascal VOC数据集中JPEG文件在目录`JPEGImages`。 同样，边界框注释应该是
存储在`Annotation directory`

此TensorFlow脚本将训练和评估数据转换为分别由1024和128个TFRecord文件组成的分片数据集。

每个验证TFRecord文件包含〜500条记录。 每次训练TFREcord文件包含〜1000条记录。 TFRecord文件中的每条记录都是一个序列化。 示例proto包含以下字段：

```python
image/encoded: str RGB色彩空间中包含JPEG编码图像的字符串
image/height:integer,高
image/width:integer，宽
image/channels: integer ，指定通道数量，始终为3
image/format: string, 指定格式, 总是'JPEG'

image/object/bbox/xmin:人标注的0+（浮点数）边界框列表
image/object/bbox/xmax:人标注的0+（浮点数）边界框列表
image/object/bbox/ymin:人标注的0+（浮点数）边界框列表
image/object/bbox/ymax:人标注的0+（浮点数）边界框列表
image/object/bbox/label:指定分类索引的整数列表。
image/object/bbox/label_text:字符串描述列表。

# 请注意，每个例子的xmin的长度与xmax，ymin和ymax的长度相同
```

```python
#  参考pascalvoc_to_tfrecords.py 重写以下字段，便可以实现SSD数据接口重载
# 例如 将labelme格式数据转成tfrecords，便可使用该项目训练，测试
'image/height': int64_feature(shape[0]),
'image/width': int64_feature(shape[1]),
'image/channels': int64_feature(shape[2]), # 3
'image/shape': int64_feature(shape),
'image/object/bbox/xmin': float_feature(xmin),
'image/object/bbox/xmax': float_feature(xmax),
'image/object/bbox/ymin': float_feature(ymin),
'image/object/bbox/ymax': float_feature(ymax),
'image/object/bbox/label': int64_feature(labels),
'image/object/bbox/label_text': bytes_feature(labels_text),
'image/object/bbox/difficult': int64_feature(difficult), # 0
'image/object/bbox/truncated': int64_feature(truncated), # 0
'image/format': bytes_feature(image_format), # JPEG
'image/encoded': bytes_feature(image_data)}))
```


# Pascal VOC 2007
- Pascal VOC 2007数据下载与数据预览参考[这里](https://blog.csdn.net/wc781708249/article/details/79615210#t0)

- VOC2007目录结构

这里只需要`Annotations` 与`JPEGImages` 
```python
<VOC2007>
|---- test
|       |--- Annotations # 需要
|       	└─  *.xml # 存放图片的 类别与边框
|       |———  JPEGImages # 需要
|       	└─ *.jpg # 存放图片
|---- train
|       |--- Annotations # 需要
|       	└─  *.xml # 存放图片的 类别与边框
|       |———  JPEGImages # 需要
|       	└─ *.jpg # 存放图片
|
| ……
| ……
```

修改[pascalvoc_to_tfrecords.py](https://github.com/balancap/SSD-Tensorflow/blob/master/datasets/pascalvoc_to_tfrecords.py) 84行 

```python
image_data = tf.gfile.FastGFile(filename, 'r').read()
# 修改成
image_data = tf.gfile.FastGFile(filename, 'rb').read()
```

- 执行命令：
```python
mkdir tfrecords
python3 tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=./VOC2007/ \
    --output_name=pascalvoc \
    --output_dir=./tfrecords
```


# Evaluation on Pascal VOC 2007
```python
python3 tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=./VOC2007/test/ \
    --output_name=voc_2007_test \
    --output_dir=./VOC2007/test
```
```python
python3 eval_ssd_network.py \
    --eval_dir=./logs/ \
    --dataset_dir=./VOC2007/test/ \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=./checkpoints/ssd_300_vgg.ckpt \
    --batch_size=1
```


----------


# Training
```python
# mkdir tfrecords
python3 tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=./VOC2007/train/ \
    --output_name=voc_2007_train \ # voc_2012_train
    --output_dir=./VOC2007/train/
```


```python
python3 train_ssd_network.py \
    --train_dir=./logs/ \
    --dataset_dir=./VOC2007/train/ \
    --dataset_name=pascalvoc_2007 \ # pascalvoc_2012
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=./checkpoints/ssd_300_vgg.ckpt \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32
```

请注意，除了训练脚本标志之外，还可能需要在`ssd_vgg_preprocessing.py`或/和网络参数（要素图层，锚点框......）中试验数据增强参数（随机裁剪，分辨率......） 在`ssd_vgg_300 / 512.py`中

# Evaluation 

```python
python3 tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=./VOC2007/test/ \
    --output_name=voc_2007_test \ # voc_2012_test
    --output_dir=./VOC2007/test/
```

``` 
python3 eval_ssd_network.py \
    --eval_dir=./logs/eval \
    --dataset_dir=./VOC2007/test/ \
    --dataset_name=pascalvoc_2007 \ # pascalvoc_2012
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=./logs/ \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```
# test

```python
# -*- coding:utf-8 -*-

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import sys
# sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './logs/'
# ckpt_filename = './logs/model.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# saver.restore(isess, ckpt_filename)

# 验证之前是否已经保存了检查点文件
ckpt = tf.train.get_checkpoint_state(ckpt_filename)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(isess, ckpt.model_checkpoint_path)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = './demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-5])
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

```



# 微调在ImageNet上训练的网络

```python
python3 train_ssd_network.py \
    --train_dir=./log/ \
    --dataset_dir=./VOC2007/train/ \
    --dataset_name=pascalvoc_2007 \ # pascalvoc_2012
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=./checkpoints/vgg_16.ckpt \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```

```python
python train_ssd_network.py \
    --train_dir=./log_finetune/ \
    --dataset_dir=./VOC2007/train/ \
    --dataset_name=pascalvoc_2007 \ # pascalvoc_2012
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=./log/model.ckpt-N \
    --checkpoint_model_scope=vgg_16 \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.00001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```