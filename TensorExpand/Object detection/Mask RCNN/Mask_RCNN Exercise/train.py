# -*- coding: utf8 -*-

'''
注意：
0、如果训练新的数据只需修改ShapesConfig与ShapesDataset类，其他不动

1、传入的图片大小必须是2^6的倍数，如：64,128,256……(64*n n=1,2,3,……)，且必须是3波段，如果进入`config.py`修改波段，后面运算会报错（不解）

2、图片对应的掩膜图片大小为对应原图的大小，背景像素为0，对象内的像素为1 即0、1像素图片，图像shape[h,w,3] ;mask shape [h,w,instance count]

3、mnist图片大小28x28x1，类别数0~9 10类 加上背景 共11类（这种模式[图像分割]必须算上背景作为一个类别）**注**：背景默认class为0，0~9 数字对应的标签为1~10

4、先将mnist图片从28x28 resize到32x32 ，再组成4x4的图片，新的图片大小32x4=128

5、masks:[height, width, instance count]  这里height=width=128，instance count=16（4x4每个格都有对象）

6、class_ids: a 1D array of class IDs of the instance masks. [instance count,] 要与mask对应起来

7、输入：图片[h,w,3]，图片每个对象的掩膜[h,w,instance count]，每个对象的class_id[instance count,] (从1开始标记，0默认为背景)
'''

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
# from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import pickle

train = 1 # 1 train;0 test

# mnist = read_data_sets('./MNIST_data', one_hot=False)

train_images=200

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    # 命名配置
    NAME = "shapes"

    # 输入图像resing
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # 使用的GPU数量。 对于CPU训练，请使用1
    GPU_COUNT = 1

    IMAGES_PER_GPU = int(1024 * 1024 * 4 // (IMAGE_MAX_DIM * IMAGE_MAX_DIM * 12))+1

    batch_size = GPU_COUNT * IMAGES_PER_GPU
    STEPS_PER_EPOCH = int(train_images / batch_size * (3 / 4))

    VALIDATION_STEPS = STEPS_PER_EPOCH // (1000 // 50)

    NUM_CLASSES = 1 + 20  # 必须包含一个背景（背景作为一个类别）

    scale = 1024 // IMAGE_MAX_DIM
    RPN_ANCHOR_SCALES = (32 // scale, 64 // scale, 128 // scale, 256 // scale, 512 // scale)  # anchor side in pixels

    RPN_NMS_THRESHOLD = 0.6  # 0.6

    RPN_TRAIN_ANCHORS_PER_IMAGE = 256 // scale

    MINI_MASK_SHAPE = (56 // scale, 56 // scale)

    TRAIN_ROIS_PER_IMAGE = 200 // scale

    DETECTION_MAX_INSTANCES = 100 * scale * 2 // 3

    DETECTION_MIN_CONFIDENCE = 0.6


config = ShapesConfig()
# config.display() # 显示细节信息


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    返回要用于的Matplotlib轴数组
     笔记本中的所有可视化。 提供一个
     控制图形大小的中心点。

     更改默认大小属性以控制大小
     的渲染图像
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class ShapesDataset(utils.Dataset):

    def __init__(self,pkl_path,class_map=None):
        super(ShapesDataset, self).__init__(class_map)
        self.pkl_path=pkl_path

    def load_shapes(self, count, height, width):
        data=self.load_pkl()
        class_dict=data[0][0] # 所有类别字典 从1开始
        # self.add_class("shapes", 0, 'BG') # 标签0默认为背景 utils中已经添加了 这里不用再添加

        # self.add_class("shapes", 1, "square")
        # self.add_class("shapes", 2, "circle")
        # self.add_class("shapes", 3, "triangle")
        # 必须从1、2、3、……开始添加，否则会出错 ，下面这种情况会导致标签对不上
        # [self.add_class('shapes',class_dict[i],i) for i in list(class_dict.keys())] # 无序添加会导致标签对应不上

        # class id反算出class name
        class_name_dict = dict(zip(class_dict.values(), class_dict.keys()))

        [self.add_class('shapes',i,class_name_dict[i]) for i in range(1,21)] # 共20类


        '''
        for i in range(count):
            comb_image, mask, class_ids = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           image=comb_image, mask=mask, class_ids=class_ids)
        '''

        for i in range(1,len(data)):
            comb_image=data[i][0]
            mask=data[i][1]
            class_ids=data[i][2]
            self.add_image("shapes", image_id=i-1, path=None,
                           width=height, height=width,
                           image=comb_image, mask=mask, class_ids=class_ids)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = info['image']
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = info['mask']
        class_ids = info['class_ids']
        return mask, class_ids.astype(np.int32)

    '''
    def random_shape(self,height, width):
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

    def random_image(self,height, width):
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
            # （而实际标注mask时不会出现这种情况的，因为完全遮挡了没办法标注mask）
            if np.sum(mask[:, :, i]) < 1:  # 完全被遮挡
                mask = np.delete(mask, i, axis=-1)
                class_id = np.delete(class_id, i)  # 对应的mask的class id 也需删除
        return image, mask, class_id
    '''
    def load_pkl(self):
        with open(self.pkl_path, 'rb') as fp:
            data = pickle.load(fp)
        return data


# if train == 1:
# Training dataset
pkl_path='./data/data_200.pkl'
dataset_train = ShapesDataset(pkl_path)
dataset_train.load_shapes(train_images, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# if train==0:
# Validation dataset
pkl_path='./data/data_631.pkl'
dataset_val = ShapesDataset(pkl_path)
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

if train == 1:

    model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    '''
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 10)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    exit(-1)
    # '''


    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        if os.path.exists(model_path):
            model.load_weights(model_path, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])

    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Training - Stage 1
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    # '''
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')
    # '''

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers="all")
    # '''

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    '''
    保存重量
    ＃通常不需要，因为回调在每个epoch后都会保存
    ＃取消注释以手动保存
    '''
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    model.keras_model.save_weights(model_path)

if train == 0:
    class InferenceConfig(ShapesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")

    # model_path = model.find_last()[1]
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    # original_image = dataset_val.load_image(image_id)


    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))
