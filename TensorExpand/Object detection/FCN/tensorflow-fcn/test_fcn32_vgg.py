#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

import fcn32_vgg
import utils

from tensorflow.python.framework import ops

img1 = scp.misc.imread("./test_data/tabby_cat.png")

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn32_vgg.FCN32VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up] # [1,ceil(h)//32,ceil(w)//32]  [1,h,w]
    down, up = sess.run(tensors, feed_dict=feed_dict)
    # down[0]  [ceil(h)//32,ceil(w)//32]
    down_color = utils.color_image(down[0]) # [ceil(h)//32,ceil(w)//32,4]
    up_color = utils.color_image(up[0])

    scp.misc.imsave('fcn32_downsampled.png', down_color)
    scp.misc.imsave('fcn32_upsampled.png', up_color)
