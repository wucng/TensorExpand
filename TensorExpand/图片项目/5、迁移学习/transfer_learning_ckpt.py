# -*- coding:utf-8 -*-

'''
迁移学习：
只更新添加层的参数，冻结原始模型
初始化参数时 只对对添加层的参数初始化

模型微调：
将原来的模型参数作为初始参数，也进行更新
初始化参数时 只对对添加层的参数初始化

# 附加 ckpt模型做迁移思路
saver = tf.train.Saver(max_to_keep=1)  # 最多保留一个版本
var_list = tf.global_variables()
print(var_list)
var_list_1=[]
for var in var_list:  # 不加载 最后两层的参数，即重新训练
    if 'fc1' in var.name  or 'ouput' in var.name:
        # var_list_1.remove(var)
        continue
    var_list_1.append(var)
print(var_list_1)

saver = tf.train.Saver(max_to_keep=1,var_list=var_list_1)

saver.restore(sess,'save_path')
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2
import os
import argparse
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

model='../../20170512-110547' # 存放ckpt的文件夹,或者pb文件

parse = argparse.ArgumentParser()
parse.add_argument('--mode', type=int, default=1, help='1 train ,0 valid')
parse.add_argument('-ta', '--trainable', type=bool, default=False, help='trainable or not')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parse.add_argument('--epochs', type=int, default=10, help='epochs')
arg = parse.parse_args()

# Hyperparameter
train = arg.mode  # 1 train ,0 test,2 输出编码测试
flag = arg.trainable
lr = arg.lr  # 1e-3~1e-6
batch_size = 128  # 逐步改变 128~128*4
img_Pixels_h = 128
img_Pixels_w = 64
img_Pixels_c = 3
num_classes = 458
epochs = arg.epochs
log_dir = './model'
keep_rata = 0.7

first=True


if not os.path.exists(log_dir):os.mkdir(log_dir)

# 编码列数
m=48

class Data():
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.read()
        self.num_images = len(self.data)
        self.start = 0
        self.end = 0

    def read(self):
        data = []
        with open(self.data_path, 'r') as fp:
            for f in fp:
                data.append(f.strip('\n'))
        self.data = data

    def shuttle_data(self):
        '''打乱数据'''
        np.random.shuffle(self.data)
        self.start = 0
        self.end = 0

    def Next_batch(self):
        self.end = min((self.start + self.batch_size, self.num_images))
        data = self.data[self.start:self.end]

        self.start = self.end
        if self.start == self.num_images:
            self.start = 0

        images = []
        labels = []

        for da in data:
            da = da.strip('\n').split(',')
            labels.append(int(da[1]))
            images.append(cv2.imread(da[0]))

        # 特征归一化处理
        imgs = np.asarray(images, np.float32)
        imgs = (imgs - np.min(imgs, 0)) * 1. / (np.max(imgs, 0) - np.min(imgs, 0))
        # imgs = (imgs - np.mean(imgs, 0)) * 1. / np.std(imgs,0)

        return imgs, np.asarray(labels, np.int64)  # [batch_size,128,64,3],[batch_size,]

# # 数据增强
# https://blog.csdn.net/medium_hao/article/details/79227056
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
    if color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    if color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height=None, width=None, bbox=None):
    # if bbox is None:
    #     bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # if image.dytpe != tf.float32:
    #    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # distorted_image = tf.image.resize_images(image, (height, width), method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))
    return distorted_image


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

with tf.Graph().as_default():
    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用GPU70%的显存

    with tf.Session(config=config) as sess:
        # 第一次执行时执行
        if first:
            # Load the model
            print('Loading feature extraction model')
            # load_model(model)
            saver = tf.train.import_meta_graph(os.path.join(model, 'model-20170512-110547.meta'))  # 导入图表结构
        else:
            # 加载ckpt模型图表（执行完成第一次后，会把ckpt/pb中的图表和参数存储到ckpt文件中）
            saver = tf.train.import_meta_graph(os.path.join(log_dir, 'model.ckpt.meta')) # 导入图表结构

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") # 输入 shape [batch_size,149,149,3]
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") # 输出 shape [batch_size,128]
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0") # is_training
        embedding_size = embeddings.get_shape()[1]

        if not flag:
            embeddings=tf.stop_gradient(embeddings) # 做迁移 不更新这层以及之前所有层的参数

        if train == 1:
            x = tf.placeholder(tf.float32, (batch_size, img_Pixels_h, img_Pixels_w, img_Pixels_c))
        else:
            x = tf.placeholder(tf.float32, (None, img_Pixels_h, img_Pixels_w, img_Pixels_c))

        y_true = tf.placeholder(tf.int64, shape=[None,], name='y_true')
        # y_true_cls = tf.argmax(y_true, dimension=1)
        with tf.variable_scope('D'):
            fc=tf.layers.dense(embeddings,48,activation=tf.nn.sigmoid,name='coding_layer')
            y_pred=tf.layers.dense(fc,num_classes,activation=tf.nn.softmax,name='output')

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

        y_pred_cls = tf.argmax(y_pred, axis=1)
        # 创建一个布尔向量，表示每张图像的真实类别是否与预测类别相同。
        correct_prediction = tf.equal(y_pred_cls, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 只更新coding_layer与output这两层参数，其他的参数不更新
        tvars = tf.trainable_variables() # 获取所有可以更新的变量
        d_params = [v for v in tvars if v.name.startswith('D/')]

        # 注 默认是更新所以参数 var_list=None
        train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost,var_list=d_params)

        init = tf.global_variables_initializer() # 默认初始化所以变量参数
        # 只初始化coding_layer与output这两层参数, global_step 也需初始化
        # init=tf.variables_initializer(var_list=d_params)
        sess.run(init)

        # 加载数据
        if train==1:
            data = Data('train.data', batch_size)
            # 训练时做数据增强，其他情况没必要做
            # 数据增强
            x1 = []
            for i in range(x.shape[0]):
                # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
                x2 = tf.squeeze(tf.slice(x, [i, 0, 0, 0], [1, img_Pixels_h, img_Pixels_w, img_Pixels_c]), 0)
                x1.append(preprocess_for_train(x2, img_Pixels_h, img_Pixels_w, None))

            x1 = tf.convert_to_tensor(x1, tf.float32)
            x1=tf.image.resize_image_with_crop_or_pad(x1,160,160)

        if train==0:
            data = Data('valid.data', batch_size)
            x1 = tf.image.resize_image_with_crop_or_pad(x, 160, 160)

        steps = data.num_images // batch_size
        index2 = np.arange(0, batch_size)

        if first:
            saver.restore(sess, os.path.join(model, 'model-20170512-110547.ckpt-250000'))
        else:
            # 验证之前是否已经保存了检查点文件
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path) # 只能导入参数

        if train==1:
            for epoch in range(epochs):
                data.shuttle_data()
                epoch_loss = 0
                for step in range(steps):
                    batch_x, batch_y = data.Next_batch()

                    # 每个step 打乱数据
                    np.random.shuffle(index2)
                    batch_x = batch_x[index2]
                    batch_y = batch_y[index2]

                    images=sess.run(x1,{x:batch_x})

                    # feed_dict = {images_placeholder: batch_x, phase_train_placeholder: False}
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False, y_true: batch_y}
                    _,c=sess.run([train_op,cost],feed_dict)
                    epoch_loss += c
                    if step % 50==0:
                        # feed_dict = {images_placeholder: batch_x, phase_train_placeholder: False,y_true:batch_y}
                        acc=sess.run(accuracy,feed_dict)
                        print('epoch',epoch,'step',step,'|','acc',acc,'|','loss',c)
                print(epoch, ' : ', epoch_loss / steps)

                # 保存所有变量 var_list指定要保持或者是提取的变量，默认是所有变量
                saver = tf.train.Saver(var_list=tf.global_variables())  # var_list=None 也是默认保持所以变量
                saver.save(sess, os.path.join(log_dir, 'model.ckpt'))
