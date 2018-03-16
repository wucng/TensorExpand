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
import glob
import scipy.misc
import scipy.ndimage


class ShapesDataset(object):

    def __init__(self, SegmentationClass='./VOCdevkit/VOC2007/SegmentationClass/*.png',
                 JPEGImages='./VOCdevkit/VOC2007/JPEGImages'):
        '''
        :param SegmentationClass: Mask 路径
        :param JPEGImages: JPG路径 
        '''
        self.image_info = []
        self.SegmentationClass=SegmentationClass
        self.JPEGImages=JPEGImages
        # self.height=224
        # self.width = 224

    def add_image(self, f, annotation_file, **kwargs):
        image_info = {
            "image": f, # JPG 路径
            "annotation": annotation_file, # 对应mask路径
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_shapes(self,count):
        '''加载图像和对应的Mask的路径（没有事先解析成numpy）'''
        Class_path = glob.glob(self.SegmentationClass)
        np.random.shuffle(Class_path)  # 数据随机打乱
        for num, path in enumerate(Class_path):
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(Class_path)))
            sys.stdout.flush()

            file_name = path.split('/')[-1].split('.')[0]
            # 对应JPG
            image_path = os.path.join(self.JPEGImages, file_name + '.jpg')

            self.add_image(f=image_path, annotation_file=path)
            if (num+1)==count:
                break

        sys.stdout.write('\n')
        sys.stdout.flush()


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask



