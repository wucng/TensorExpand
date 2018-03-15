# -*- coding:utf-8 -*-
#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

import fcn32_vgg
import utils

from tensorflow.python.framework import ops
from shape_data import ShapesDataset
import BatchDatsetReader as dataset


NUM_CLASS=4 # 1+3 (包括背景)
train=1 # 1 train ;0 test
MAX_ITERATION=int(1e5+1)
batch_size=2
logdir='./logs'
IMAGE_SIZE=224
train_images_num=500
val_images_num=50
lr=1e-6
# img1 = scp.misc.imread("./test_data/tabby_cat.png")

# 传入自己的数据
dataset_train = ShapesDataset()
dataset_train.load_shapes(train_images_num, IMAGE_SIZE, IMAGE_SIZE)
train_records=dataset_train.image_info

dataset_val = ShapesDataset()
dataset_val.load_shapes(val_images_num, IMAGE_SIZE, IMAGE_SIZE)
valid_records=dataset_val.image_info

# ---------------------------------------------#
print(len(train_records))
print(len(valid_records))

print("Setting up dataset reader")
image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
if train==1:
    train_dataset_reader = dataset.BatchDatset(train_records, image_options)
validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)


with tf.Session() as sess:
    images = tf.placeholder("float",[None,IMAGE_SIZE,IMAGE_SIZE,3])
    labels=tf.placeholder(tf.int32,[None,IMAGE_SIZE,IMAGE_SIZE,1])
    # feed_dict = {images: img1,labels:}
    # batch_images = tf.expand_dims(images, 0)
    batch_images = images
    if train==1:
        vgg_fcn = fcn32_vgg.FCN32VGG()
        with tf.name_scope("content_vgg"):
            vgg_fcn.build(batch_images, train=True,num_classes=NUM_CLASS,debug=True)
    else:
        vgg_fcn = fcn32_vgg.FCN32VGG()
        with tf.name_scope("content_vgg"):
            vgg_fcn.build(batch_images,num_classes=NUM_CLASS, debug=True)

    print('Finished building Network.')

    # loss
    loss= tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=vgg_fcn.upscore,
                                                                         labels=tf.squeeze(labels, squeeze_dims=[3]),
                                                                         name="entropy")))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)


    # init = tf.global_variables_initializer()
    # sess.run(init)

    saver = tf.train.Saver()
    # 验证之前是否已经保存了检查点文件
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        tf.global_variables_initializer().run()


    if train==1:
        for itr in range(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(batch_size=batch_size)
            # feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            feed_dict = {images: train_images, labels:train_annotations}
            sess.run(train_op,feed_dict)

            if itr%20==0:
                print('itr',itr,'|','loss',loss.eval(feed_dict))
            if itr%200==0:
                saver.save(sess,logdir+'/model.ckpt')

    else:
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(1)
        feed_dict = {images:valid_images} #  np.expand_dims(valid_images,0)
        print('Running the Network')
        tensors = [vgg_fcn.pred, vgg_fcn.pred_up]  # [1,ceil(h)//32,ceil(w)//32]  [1,h,w]
        down, up = sess.run(tensors, feed_dict=feed_dict)
        # down[0]  [ceil(h)//32,ceil(w)//32]
        down_color = utils.color_image(down[0],num_classes=NUM_CLASS)  # [ceil(h)//32,ceil(w)//32,4]
        up_color = utils.color_image(up[0],num_classes=NUM_CLASS)

        scp.misc.imsave('fcn32_downsampled.png', down_color)
        scp.misc.imsave('fcn32_upsampled.png', up_color)