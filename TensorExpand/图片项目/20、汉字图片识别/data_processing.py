# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import glob
import os,sys

'''
1、图片二值化处理，缩放成统一大小

2、将处理后的图片数据转成tfrecord文件
'''


# 打印二值化处理过程
def plot_binary_image(image_path):
    img = cv2.imread(image_path, 0)

    ret2, th2 = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th=cv2.bitwise_not(th2)

    # ret, th = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY_INV)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img,cmap='gray')
    plt.title('origin')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(th2,cmap='gray')
    plt.title('binarization')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(th,cmap='gray')
    plt.title('binary inversion')
    plt.axis('off')

    plt.show()

# 帮助功能打印转换进度。
def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def Image_to_tfrecord(images_path,record_location='./data/train',save_num=10000,img_pixel=64):
    '''
    直接将图像转成tfrecord
    :param record_location: tfr文件保存位置
    :param img_pixel: 设置图像固定大小
    :param save_num: 每隔多少张图片保存成一个tfrecord文件
    :return: 
    '''
    writer = None
    current_index = 0
    num_images=len(images_path)

    if save_num==None:save_num=num_images

    for i,image_path in enumerate(images_path):
        print_progress(count=i, total=num_images - 1)

        if current_index % save_num == 0:  # 每隔100幅图像，训练样本的信息就被写入到一个新的Tfrecode文件中，以加快操作的进程
            if writer:
                writer.close()

            record_filename = "{record_location}-{current_index}.tfrecords".format(
                record_location=record_location,
                current_index=current_index)

            writer = tf.python_io.TFRecordWriter(record_filename)
        current_index += 1

        img = cv2.imread(image_path, 0)

        # ret2, th2 = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # th = cv2.bitwise_not(th2)
        # img = cv2.resize(th, dsize=(img_pixel, img_pixel))  # 缩放到统一大小,像素值转成0、1
        ret, img = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY_INV)
        img = cv2.resize(img, dsize=(img_pixel, img_pixel))  # 缩放到统一大小,像素值转成0、1

        image_label = int(image_path.split('/')[-2])  # 对应的标签值

        image_bytes = np.array(img, np.float16).tobytes()  # 将图片转化为bytes
        # image_label = file_name.encode("utf-8")  # 这里直接使用文件名作标签

        example = tf.train.Example(features=tf.train.Features(feature={
            # 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_label])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        }))
        writer.write(example.SerializeToString())
        # exit(-1)
    if writer:
        writer.close()

if __name__=="__main__":
    train_images_path = glob.glob('./HWDB1/train/*/*.png')  # 获取图片路径
    np.random.shuffle(train_images_path)  # 打乱路径，启动打乱数据的目的
    print(len(train_images_path))  # 895035

    # plot_binary_image(train_images_path[10])

    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists('./data/train-0.tfrecords'):
        Image_to_tfrecord(train_images_path,save_num=50000,img_pixel=32)

    # test数据同样处理
    test_images_path = glob.glob('./HWDB1/test/*/*.png')  # 获取图片路径
    print(len(test_images_path))  # 223991

    if not os.path.exists('./data/test-0.tfrecords'):
        Image_to_tfrecord(test_images_path, record_location='./data/test',save_num=50000,img_pixel=32)

    exit(-1)