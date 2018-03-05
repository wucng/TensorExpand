# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from glob import glob
from PIL import Image
import os


# 设置超参数
num_images=895035
num_class=3755
image_size=32 # 64
batch_size=256
learning_rate=1e-4
global_step = tf.Variable(0, name="global_step", trainable=False)
# learning_rate=tf.train.polynomial_decay(1e-4,global_step,num_images*2//batch_size,1e-8)
epochs=100
train=1 # 1 train,0 test
keep=0.8
logdir='./checkpoint/'

def tfrecord_to_numpy(file_path,img_pixel=32,batch_size=64):
    '''
    加载tfrecord文件
    :param file_path: tfrecord文件路径
    :param img_pixel: 样本像素大小
    :param batch_size: 每批次训练的样本数
    :return: 
    '''
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(file_path))  # 加载多个Tfrecode文件
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            # 'label': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        })

    record_image = tf.decode_raw(features['image'], tf.float16)
    image = tf.reshape(record_image, [img_pixel, img_pixel])
    label = tf.cast(features['label'], tf.int64)
    # label = tf.cast(features['label'], tf.string)

    min_after_dequeue =num_images//batch_size #1000
    capacity = min_after_dequeue + batch_size
    data, label = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    # 将图像转换为灰度值位于[0,1)的浮点类型，
    # float_image_batch = tf.image.convert_image_dtype(data, tf.float32)
    # return float_image_batch, label
    return data,label

x=tf.placeholder(tf.float32,[None,image_size,image_size])
y_=tf.placeholder(tf.int64,[None,])
keep_rate=tf.placeholder(tf.float32)
is_training=tf.placeholder(tf.bool)

# 搭建网络
net=tf.expand_dims(x,-1) # [n,32,32,1]

net=tf.layers.conv2d(net,64,7,1,'same',activation=tf.nn.leaky_relu) # [n,32,32,64]
net=slim.batch_norm(net,is_training=is_training)
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,16,16,64]

branch1=tf.layers.conv2d(net,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,16,16,64]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,16,16,64]
branch2=slim.batch_norm(branch2,is_training=is_training)

net=tf.nn.leaky_relu(net+branch2)
branch1=tf.layers.conv2d(net,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,16,16,64]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,16,16,64]
branch2=slim.batch_norm(branch2,is_training=is_training)

net=tf.nn.leaky_relu(net+branch2)
branch1=tf.layers.conv2d(net,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,16,16,64]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,16,16,64]
branch2=slim.batch_norm(branch2,is_training=is_training)

net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,8,8,64]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
# net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.conv2d(net,128,1,1,'same',activation=tf.nn.leaky_relu) # [n,8,8,128]
net=tf.nn.leaky_relu(net+branch2) # [n,8,8,128]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2) # [n,8,8,128]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2) # [n,8,8,128]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2) # [n,8,8,128]
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,4,4,128]

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
# net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.conv2d(net,256,1,1,'same',activation=tf.nn.leaky_relu) # [n,4,4,256]
net=tf.nn.leaky_relu(net+branch2) # [n,4,4,256]

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,2,2,256]

branch1=tf.layers.conv2d(net,512,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,512]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,512,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,512]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.layers.conv2d(net,512,1,1,'same',activation=tf.nn.leaky_relu) # [n,2,2,512]
net=tf.nn.leaky_relu(net+branch2) # [n,2,2,512]
# net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,512,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,512]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,512,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,512]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,512,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,512]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,512,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,512]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,1,1,512]

# net=tf.squeeze(net) # [n,256]
net=tf.reshape(net,[-1,512])
print(net.shape)
fc1=tf.layers.dense(net,1024,activation=tf.nn.leaky_relu)
fc1=slim.dropout(fc1,keep_rate,is_training=is_training)
prediction=tf.layers.dense(fc1,num_class)

'''
net=tf.expand_dims(x,-1) # [n,32,32,1]

net=tf.layers.conv2d(net,64,7,2,'same',activation=tf.nn.leaky_relu) # [n,16,16,64]
net=slim.batch_norm(net,is_training=is_training)
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,8,8,64]

branch1=tf.layers.conv2d(net,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,64]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,64]
branch2=slim.batch_norm(branch2,is_training=is_training)

net=tf.nn.leaky_relu(net+branch2)
branch1=tf.layers.conv2d(net,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,64]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,64]
branch2=slim.batch_norm(branch2,is_training=is_training)

net=tf.nn.leaky_relu(net+branch2)
branch1=tf.layers.conv2d(net,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,64]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,64,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,8,8,64]
branch2=slim.batch_norm(branch2,is_training=is_training)

net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,4,4,64]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
# net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.conv2d(net,128,1,1,'same',activation=tf.nn.leaky_relu) # [n,4,4,128]
net=tf.nn.leaky_relu(net+branch2) # [n,4,4,128]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2) # [n,4,4,128]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2) # [n,4,4,128]

branch1=tf.layers.conv2d(net,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,128,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,4,4,128]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2) # [n,4,4,128]
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,2,2,128]

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
# net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.conv2d(net,256,1,1,'same',activation=tf.nn.leaky_relu) # [n,2,2,256]
net=tf.nn.leaky_relu(net+branch2) # [n,2,2,256]

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)

branch1=tf.layers.conv2d(net,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(branch1,256,3,1,padding='same',activation=tf.nn.leaky_relu) # [n,2,2,256]
branch2=slim.batch_norm(branch2,is_training=is_training)
net=tf.nn.leaky_relu(net+branch2)
net=tf.layers.max_pooling2d(net,2,2,'same') # [n,1,1,256]
# net=tf.squeeze(net) # [n,256]
net=tf.reshape(net,[-1,256])
print(net.shape)
fc1=tf.layers.dense(net,1024,activation=tf.nn.leaky_relu)
fc1=slim.dropout(fc1,keep_rate,is_training=is_training)
prediction=tf.layers.dense(fc1,num_class)
'''

'''
net=tf.expand_dims(x,-1) # [n,32,32,1]
branch1=tf.layers.conv2d(net,32,7,1,padding='valid',activation=tf.nn.relu) # [n,26,26,32]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(net,64,7,1,padding='valid',activation=tf.nn.relu) # [n,26,26,64]
branch2=slim.batch_norm(branch2,is_training=is_training)
branch3=tf.layers.conv2d(net,128,7,1,padding='valid',activation=tf.nn.relu) # [n,26,26,128]
branch3=slim.batch_norm(branch3,is_training=is_training)

net1=tf.concat([branch1, branch2, branch3],axis=-1) #[n,26,26,224]
net1=tf.layers.max_pooling2d(net1,2,2,'same') # [n,13,13,224]

branch1=tf.layers.conv2d(net1,256,5,1,padding='valid',activation=tf.nn.relu) # [n,9,9,256]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(net1,512,5,1,padding='valid',activation=tf.nn.relu) # [n,9,9,512]
branch2=slim.batch_norm(branch2,is_training=is_training)
branch3=tf.layers.conv2d(net1,1024,5,1,padding='valid',activation=tf.nn.relu) # [n,9,9,1024]
branch3=slim.batch_norm(branch3,is_training=is_training)

net2=tf.concat([branch1, branch2, branch3],axis=-1) #[n,9,9,1792]
net2=tf.layers.max_pooling2d(net2,2,2,'same') # [n,5,5,1792]

branch1=tf.layers.conv2d(net2,1252,3,1,padding='valid',activation=tf.nn.relu) # [n,3,3,1252]
branch1=slim.batch_norm(branch1,is_training=is_training)
branch2=tf.layers.conv2d(net2,1252,3,1,padding='valid',activation=tf.nn.relu) # [n,3,3,1252]
branch2=slim.batch_norm(branch2,is_training=is_training)
branch3=tf.layers.conv2d(net2,1251,3,1,padding='valid',activation=tf.nn.relu) # [n,3,3,1251]
branch3=slim.batch_norm(branch3,is_training=is_training)

net3=tf.concat([branch1, branch2, branch3],axis=-1) #[n,3,3,3755]
net3=tf.layers.max_pooling2d(net3,2,2,'valid') # [n,1,1,3755]

prediction=tf.squeeze(net3) # [n,3755]
'''
'''
net=tf.layers.conv2d(net,32,7,padding='same',name='conv3')
net=tf.layers.conv2d(net,128,7,padding='same',name='conv4')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,16,16,128]

net=tf.layers.conv2d(net,128,7,padding='same',name='conv5')
net=tf.layers.conv2d(net,256,5,padding='same',name='conv6')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,8,8,256]

net=tf.layers.conv2d(net,256,5,padding='same',name='conv7')
net=tf.layers.conv2d(net,512,5,padding='same',name='conv8')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,4,4,512]

net=tf.layers.conv2d(net,512,5,padding='same',name='conv9')
net=tf.layers.conv2d(net,1024,3,padding='same',name='conv10')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,2,2,1024]

net=tf.layers.conv2d(net,1024,3,padding='same',name='conv11')
net=tf.layers.conv2d(net,num_class,3,padding='same',name='conv12')
# net=tf.layers.batch_normalization(net,is_training=is_training)
net=slim.batch_norm(net,is_training=is_training)
net=tf.nn.leaky_relu(net)
net=slim.max_pool2d(net,2,2,'same') # [n,1,1,3755]

net=tf.reshape(net,[-1,num_class])
net=slim.dropout(net,keep_rate,is_training=is_training)

prediction=slim.softmax(net) # [n,3755]
# '''

'''
def cnn_net(images,num_classes=3755,is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CNNNet'):
    """
    Creates a variant of the CNN-Net model.
    :param images: 输入图像数据 形状[n,h,w,c]
    :param num_classes: 类别数
    :param is_training: 是否训练 模型训练设置为True，测试、推理设置为False
    :param dropout_keep_prob: droupout保持率
    :param prediction_fn: 输出层的激活函数
    :param scope: 节点名
    :return: 
        net：2D Tensor ,logits （pre-softmax激活）如果num_classes
            是非零整数，或者如果num_classes为0或None输入到逻辑层           
        end_points：从网络组件到相应的字典激活。
    """

    # 如果使用mnist images的shape为[n,64,64,1]
    with tf.variable_scope(scope, 'CNNNet', [images]): # 其中[images]为传入的数据
        net = slim.conv2d(images, 64, [5,5], scope='conv1') # 5x5卷核，输出节点64 默认stride为1  ;shape [n,64,64,64]
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1') # 本地响应规范化  一般可以不使用
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')  # 2x2池化核  stride为2  ;shape [n,32,32,64]

        net = slim.conv2d(net, 64, [5, 5], scope='conv2') # 5x5卷核，输出节点64 默认stride为1  ;shape [n,32,32,64]
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2') # 2x2池化核  stride为2  ;shape [n,16,16,64]

        net = slim.conv2d(net, 128, [5, 5], scope='conv3')  # 5x5卷核，输出节点64 默认stride为1  ;shape [n,16,16,128]
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')  # 2x2池化核  stride为2  ;shape [n,8,8,128]

        net = slim.conv2d(net, 256, [5, 5], scope='conv4')  # 5x5卷核，输出节点64 默认stride为1  ;shape [n,8,8,256]
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool4')  # 2x2池化核  stride为2  ;shape [n,4,4,256]

        net = slim.conv2d(net, 512, [5, 5], scope='conv5')  # 5x5卷核，输出节点64 默认stride为1  ;shape [n,4,4,512]
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm5')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool5')  # 2x2池化核  stride为2  ;shape [n,2,2,512]

        net = slim.conv2d(net, 1024, [5, 5], scope='conv6')  # 5x5卷核，输出节点64 默认stride为1  ;shape [n,4,4,512]
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm6')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool6')  # 2x2池化核  stride为2  ;shape [n,1,1,1024]
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout')  # droupout 层 ；shape [n,3755]

        net=slim.conv2d(net,num_classes,[1,1],scope='conv7') #
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm7') # [n,1,1,3755]

        net=tf.squeeze(net) # [n,3755]

    return net

cnn_net.default_image_size = 64  # 这里使用mnist数据  如果使用cifar 改成32

def arg_scope(weight_decay=0.00004,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=0.001,
            activation_fn=tf.nn.relu):
  """Returns the scope with the default parameters for inception_resnet_v2.
  使用了batch_norm，相对来说要比cifarnet_arg_scope效果更佳，推荐使用该方式进行各层参数配置
  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Activation function for conv2d.
  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'fused': None,  # Use fused batch norm if possible.
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope

image_shaped_input = tf.expand_dims(x, -1)  # [n,64,64,1]

with slim.arg_scope(arg_scope()):
    prediction= cnn_net(images=image_shaped_input, num_classes=num_class, is_training=is_training, dropout_keep_prob=keep)
'''

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__=="__main__":
    sess = tf.InteractiveSession()
    if train==1:
        x_train_batch, y_train_batch = tfrecord_to_numpy(glob('./data/train-*.tfrecords'),batch_size=batch_size)
        # acc_last=0.0
        steps = num_images // batch_size
    if train==0:
        x_test_batch, y_test_batch = tfrecord_to_numpy(glob('./data/test-*.tfrecords'),batch_size=batch_size)
        acc_list = []
    # print(x_train_batch.shape)
    # print(y_train_batch.shape)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    saver=tf.train.Saver(tf.global_variables())

    if not os.path.exists(logdir): os.mkdir(logdir)
    # 验证之前是否已经保存了检查点文件
    #ckpt = tf.train.get_checkpoint_state(logdir)
    #if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)

    # print(sess.run(y_train_batch))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        if train == 1:  # 训练
            while not coord.should_stop():

                for epoch in range(epochs):
                    acc_last=0.0
                    ckpt = tf.train.get_checkpoint_state(logdir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)

                    for step in range(steps):
                        # 验证之前是否已经保存了检查点文件
                        # if step%500==0:
                        #     ckpt = tf.train.get_checkpoint_state(logdir)
                        #     if ckpt and ckpt.model_checkpoint_path:
                        #         saver.restore(sess, ckpt.model_checkpoint_path)

                        # Run training steps or whatever
                        curr_x_train_batch,curr_y_train_batch = sess.run([x_train_batch,y_train_batch])
                        sess.run([optimizer,global_step],{x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:keep,is_training:True})
                        # optimizer.run({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:keep,is_training:True})

                        acc = accuracy.eval({x: curr_x_train_batch, y_: curr_y_train_batch, keep_rate: 1., is_training: True})
                        if step %20==0:
                            # acc=accuracy.eval({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:1.,is_training:True})
                            loss=cost.eval({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:1.,is_training:True})
                            print('epoch',epoch,'|','step',step,'|','acc',acc,'|','loss',loss)

                        # if acc>acc_last and acc>0.6:
                        if acc>0.5:
                            #print('acc',acc)
                            #ckpt = tf.train.get_checkpoint_state(logdir)
                            #if ckpt and ckpt.model_checkpoint_path:
                            #    saver.restore(sess, ckpt.model_checkpoint_path)
                            # acc_last=acc
                            # print('acc_last',acc_last)
                            # saver.save(sess, logdir + 'model.ckpt',global_step=step)
                            saver.save(sess, logdir + 'model.ckpt')

                break

        if train==0: # 测试
            while not coord.should_stop():

                for step in range(223991//batch_size):
                    curr_x_test_batch, curr_y_test_batch = sess.run([x_test_batch, y_test_batch])

                    if step % 20 == 0:
                        acc = accuracy.eval({x: curr_x_test_batch, y_: curr_y_test_batch, keep_rate: 1., is_training: True})
                        acc_list.append(acc)
                        print('step', step, '|', 'acc', acc)
                print('test acc', np.mean(acc_list))
                break

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
