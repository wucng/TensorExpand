参考：

- [MarvinTeichmann/tensorflow-fcn](https://github.com/MarvinTeichmann/tensorflow-fcn)
- [shekkizh/FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow)
- [aurora95/Keras-FCN](https://github.com/aurora95/Keras-FCN)
- [wkentaro/pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
- [CSAILVision/sceneparsing](https://github.com/CSAILVision/sceneparsing)
- [hangzhaomit/semantic-segmentation-pytorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch)

------
用于语义分割的全卷积网络的Tensorflow实现（http://fcn.berkeleyvision.org）

----------
[完全卷积网络的语义分割](http://arxiv.org/pdf/1605.06211v1.pdf)（FCNs）的Tensorflow实现。

实施主要基于论文[链接](https://github.com/shelhamer/fcn.berkeleyvision.org)作者提供的参考代码。 该模型应用于由MIT http://sceneparsing.csail.mit.edu/提供的场景解析挑战数据集。


----------


# 先决条件
- 在12GB TitanX上训练约6-7小时后获得结果。
- 代码最初是用tensorflow0.11和python2.7编写和测试的。 tf.summary调用已更新为与tensorflow版本0.12一起使用。 要使用旧版本的tensorflow，请使用分支[tf.0.11_compatible](https://github.com/shekkizh/FCN.tensorflow/tree/tf.0.11_compatible)。
- 在[问题＃9](https://github.com/shekkizh/FCN.tensorflow/issues/9)中讨论了在tensorflow1.0和windows下工作时的一些问题。
- 要训练模型，只需执行`python FCN.py`
- 要显示随机批次图像的结果，请使用flag `--mode = visualize`
- `debug`标志可以在训练期间设置以添加关于激活，梯度，变量等的信息。
- 日志文件夹中的[IPython笔记本](https://github.com/shekkizh/FCN.tensorflow/blob/master/logs/images/Image_Cmaped.ipynb)可以用来查看以下颜色的结果。


----------
# Results

结果通过用256x256的调整大小的图像分批地训练模型而获得。 请注意，尽管训练是在这个图像大小下完成的 - 没有什么能够防止模型在任意大小的图像上工作。 对预测图像没有进行后期处理。 训练已完成9个epoch - 较短的训练时间解释了为什么某些概念在模型中似乎在语义上被理解，而另一些则不是。 以下结果来自验证数据集随机选择的图像。

与caffe中的论文参考模型实现相差无几。 添加的新图层的权重初始化为小值，并且使用Adam Optimizer（学习率= 1e-4）完成学习。

![这里写图片描述](https://github.com/shekkizh/FCN.tensorflow/raw/master/logs/images/inp_1.png)![这里写图片描述](https://github.com/shekkizh/FCN.tensorflow/raw/master/logs/images/gt_c1.png)![这里写图片描述](https://github.com/shekkizh/FCN.tensorflow/raw/master/logs/images/pred_c1.png)


----------

# 意见

- 小批量大小对于将训练模型适用于内存是必要的，但解释了学习速度缓慢
- 有许多例子的概念似乎被正确识别和分割 - 在上面的例子中，你可以看到汽车和人被识别得更好。 我相信这可以通过训练更长的时代来解决。
- 此外，**图像的大小调整**会导致信息的丢失 - 您可以注意到这一点，因为较小的对象的分割精度较低。

![这里写图片描述](https://github.com/shekkizh/FCN.tensorflow/raw/master/logs/images/sparse_entropy.png)

现在对于梯度，

- 如果仔细观察渐变，您会注意到初始训练几乎完全位于新添加的图层上 - 只有在这些图层经过合理训练之后，才能看到VGG图层获得某种渐变流。 这是可以理解的，因为改变新的层次会在一开始就影响损失目标。
- netowrk的早期层使用VGG权重进行初始化，因此概念上需要较少的调整，除非训练数据变化很大 - 在这种情况下不是这样。
- 第一层卷积模型捕获低层次信息，并且由于这个依赖于数据集的依赖关系，您会注意到梯度会调整第一层权重以使模型适用于数据集。
- 来自VGG的其他conv层具有非常小的梯度，因为这里捕获的概念对我们的最终目标足够好 - 分割。
- 这是 **Transfer Learning**如此完美的核心原因。 只是想到在这里指出这一点。

![这里写图片描述](http://img.blog.csdn.net/20180313173909131?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


# Useful Links
作者在论文[链接](http://techtalks.tv/talks/fully-convolutional-networks-for-semantic-segmentation/61606/)上给出的演示视频


# 流程图
![这里写图片描述](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/Object%20detection/FCN/shekkizh_FCN.tensorflow/FCN-tensorflow/FCN.png)

# 实践
1、git项目

```python
git clone https://github.com/shekkizh/FCN.tensorflow.git
```
2、传入自己的数据
shape_data.py
```python
# -*- coding:utf-8 -*-
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import cv2
import math
import sys


class ShapesDataset(object):

    def __init__(self, class_map=None):
        self.image_info = []
        # self.height=224
        # self.width = 224

    def add_image(self, f, annotation_file, **kwargs):
        image_info = {
            "image": f, # [h,w,3]
            "annotation": annotation_file, # [h,w]  BatchDatsetReader会转成[h,w,1]
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_shapes(self, count, height, width):
        for i in range(count):
            comb_image, _, _, mask_ = self.random_image(height, width)
            self.add_image(f=comb_image, annotation_file=mask_)

    def random_shape(self, height, width):
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

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
        return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]

    def random_image(self, height, width):
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # images=[]
        mask = []
        class_id = []

        bg_color = bg_color.reshape([1, 1, 3])
        image = np.ones([height, width, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)

        N = random.randint(1, 4)
        for _ in range(N):
            image_mask = np.zeros([height, width], dtype=np.uint8)
            shape, color, dims = self.random_shape(height, width)
            x, y, s = dims
            if shape == 'square':
                cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
                cv2.rectangle(image_mask, (x - s, y - s), (x + s, y + s), 1, -1)  # 0、1组成的图片
                # images.append(img)
                mask.append(image_mask)
                class_id.append(1)  # 对应class ID 1

            elif shape == "circle":
                cv2.circle(image, (x, y), s, color, -1)
                cv2.circle(image_mask, (x, y), s, 1, -1)
                # images.append(img)
                mask.append(image_mask)
                class_id.append(2)  # 对应class ID 2

            elif shape == "triangle":
                points = np.array([[(x, y - s),
                                    (x - s / math.sin(math.radians(60)), y + s),
                                    (x + s / math.sin(math.radians(60)), y + s),
                                    ]], dtype=np.int32)
                cv2.fillPoly(image, points, color)
                cv2.fillPoly(image_mask, points, 1)
                # images.append(img)
                mask.append(image_mask)
                class_id.append(3)  # 对应class ID 3

        # images=np.asarray(images,np.float32) # [h,w,c]
        mask = np.asarray(mask, np.uint8).transpose([1, 2, 0])  # [h,w,instance count]
        class_id = np.asarray(class_id, np.uint8)  # [instance count,]

        # Handle occlusions 处理遮挡情况
        count = mask.shape[-1]
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            # 如果mask 全为0 也就是意味着完全被遮挡，需丢弃这种mask，否则训练会报错
            # （而实际标准mask时不会出现这种情况的，因为完全遮挡了没办法标注mask）
            if np.sum(mask[:, :, i]) < 1:  # 完全被遮挡
                mask = np.delete(mask, i, axis=-1)
                class_id = np.delete(class_id, i)  # 对应的mask的class id 也需删除

        count = mask.shape[-1]  # 完全覆盖的mask会被删除，重新计算mask个数
        bboxes = []  # [instance count,4]
        [bboxes.append(self.mask2box(mask[:, :, i])) for i in range(count)]
        bboxes = np.asarray(bboxes)
        gt_boxes = np.hstack((bboxes, class_id[:, np.newaxis]))  # [instance count,5] 前4列为boxs，最后一列为 class id

        mask_ = np.zeros((height, width), dtype=np.float32)  # [h,w]
        for i in range(count):
            mask_ += mask[:, :, i] * class_id[i]

        mask = np.asarray(mask, np.uint8).transpose([2, 0, 1])  # [instance count,h,w]

        return image, gt_boxes, mask, mask_
```


# 3、数据显示
##1、image 

shape [224,224,3] 像素值0~255 

![这里写图片描述](http://img.blog.csdn.net/20180315111720695?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##2、mask 

shape[224,224] 像素值0、1、2、3

0为背景 1为矩形 2为圆 3为三角形

![这里写图片描述](http://img.blog.csdn.net/20180315111946039?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


# 4、改写FCN.py

FCN_my_data.py

```python

'''
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    '''
    # -----------------------------------------------#
    # 传入自己的数据
    # train_records，valid_records [{'image': f, 'annotation': annotation_file, 'filename': filename},……]
    # f shape[IMAGE_SIZE, IMAGE_SIZE, 3]   annotation_file shape [IMAGE_SIZE, IMAGE_SIZE]
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(train_images_num, IMAGE_SIZE, IMAGE_SIZE)
    train_records=dataset_train.image_info

    dataset_val = ShapesDataset()
    dataset_val.load_shapes(val_images_num, IMAGE_SIZE, IMAGE_SIZE)
    valid_records=dataset_val.image_info

    # ---------------------------------------------#
```

# 5、运行
```python
# 执行
python3 FCN_my_data.py
```

可视化结果执行：`python3 FCN_my_data.py --mode visualize`
