#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
try:
  from osgeo import gdal,ogr
except:
  import gdal,ogr

import os
from os import path
import datetime

from osgeo.gdalconst import *
import glob
import argparse


def create_pickle_train(image_path, mask_path, img_pixel=9, channels=4):
    m = 0

    image_data = Multiband2Array(image_path,channels)
    print("data_matrix_max= ", image_data.max())
    print("data_matrix_min= ", image_data.min())
    # mask_data = cv2.split(cv2.imread(mask_path))[0] / 255
    mask_data=Multiband2Array(mask_path,channels)/255

    x_size, y_size = image_data.shape[:2]

    data_list = []

    for i in range(0, x_size - img_pixel + 1, img_pixel // 2):  # 文件夹下的文件名
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel // 2):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            train_label = mask_data[i+img_pixel//2,j+img_pixel//2]
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签

            data_list.append(data2)

            m += 1

            if m % 10000 == 0:
                print(datetime.datetime.now(), "compressed {number} images".format(number=m))
                data_matrix = np.array(data_list, dtype=int)

                data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels + 1)))
                return data_matrix

    print(len(data_list))
    print(m)

    data_matrix = np.array(data_list, dtype=int)

    data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels+1)))

    """
    with gzip.open('D:/train_data_64.pkl', 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
    """
    return data_matrix # shape [none,9*9*4+1]

def create_pickle_train2(image_path, mask_path, img_pixel=400, channels=4):
    m = 0
    compress_count=0  #增加一个值，来返回压缩了几次  by bxjxf
    # num_img+=1
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # mask_img= gdal.Open(mask_path, gdal.GA_ReadOnly)  # 只读方式打开原始影像
    # srcXSize = mask_img.RasterXSize  # 宽度
    # srcYSize = mask_img.RasterYSize  # 高度
    # band=mask_img.GetRasterBand(1)
    # mask_data=band.ReadAsArray(0,0,srcYSize,srcXSize)
    step=img_pixel//3
    '''
    mask_img=gdal.Open(mask_path)
    mask_data_temp=mask_img.ReadAsArray()
    # print(mask_data.shape)
    row_num,col_num=mask_data_temp.shape
    mask_data=np.zeros([row_num,col_num])
    mask_data=mask_data_temp
    '''
    mask_data=Multiband2Array(mask_path,1)
    if np.max(mask_data)==255:mask_data=mask_data/255
    # mask_data=mask_data_temp
    # #将图像中的编码映射成序列数   2017.10.16,by xjxf  __start
    # for i_1 in range(row_num):
    #     for j_1 in range(col_num):
    #         if mask_data_temp[i_1,j_1] ==2 or mask_data_temp[i_1,j_1]==50:
    #             mask_data[i_1,j_1]=1
    #         elif mask_data_temp[i_1,j_1]==3 or mask_data_temp[i_1,j_1]==54:
    #             mask_data[i_1,j_1]=2
    #         else:
    #             mask_data[i_1, j_1] = 0
    # mask_data_new=mask_data.reshape([row_num,col_num])
    # cv2.imwrite("xjxf.tif",mask_data_new)
    # print(num_img)
    # # 将图像中的编码映射成序列数   2017.10.16,by xjxf  __end

    image_data = Multiband2Array(image_path,channels)

    # mask_data = cv2.split(cv2.imread(mask_path))[0]

    x_size, y_size = image_data.shape[:2]

    data_list = []
    flag_x=True
    flag_y=True
    # print(len(data_list))

    for i in range(0, x_size - img_pixel + 1, step):  # 文件夹下的文件名
        i_end=i+img_pixel
        if i + img_pixel > x_size:
            # i = x_size - img_pixel - 1
            i_end=x_size
            flag_x=False

        flag_y=False
        for j in range(0, y_size - img_pixel + 1,step):
            j_end=j+img_pixel
            if j + img_pixel > y_size:
            #     j = y_size - img_pixel - 1
                j_end=y_size
                flag_y=False

            cropped_data_temp = image_data[i:i_end, j:j_end]
            #对截取的样本做扩充, 2017.10.24, by xjxf __start

            cropped_data=np.lib.pad(cropped_data_temp,((0,img_pixel-(i_end-i)),(0,img_pixel-(j_end-j)),(0,0)),'constant',constant_values=0)
            # 对截取的样本做扩充, 2017.10.24, by xjxf __end


            data_1 = cropped_data.reshape((-1, img_pixel * img_pixel*channels ))  # 展成一行
            # cropped_data_2 = image_data[i:i + img_pixel, j:j + img_pixel, 1]
            # data_2 = cropped_data_2.reshape((-1, img_pixel * img_pixel ))  # 展成一行
            # cropped_data_3 = image_data[i:i + img_pixel, j:j + img_pixel, 2]
            # data_3 = cropped_data_3.reshape((-1, img_pixel * img_pixel ))  # 展成一行
            cropped_mask_data_temp=mask_data[i:i_end,j:j_end]
            # 对截取的样本做扩充, 2017.10.24, by xjxf __start
            cropped_mask_data=np.lib.pad(cropped_mask_data_temp,((0,img_pixel-(i_end-i)),(0,img_pixel-(j_end-j))),'constant',constant_values=0)
            # 对截取的样本做扩充, 2017.10.24, by xjxf __end
            train_label = cropped_mask_data.reshape((-1,img_pixel*img_pixel))

            # data2 = np.append(data_1[np.newaxis,:], data_2[np.newaxis,:])
            # data2=np.append(data2,data_3[np.newaxis,:])

            data2=np.append(data_1,train_label)[np.newaxis,:]


            # if train_label==0 or train_label==1 or train_label==3 or train_label==5:    #去除标签是其他的样本
            # if train_label==1:    #去除标签是其他的样本
            #     print("hello")
            data_list.append(data2)
            m += 1


            if m % 10000 == 0:
                print(datetime.datetime.now(), "compressed {number} images".format(number=m))
                data_matrix = np.array(data_list, dtype=int)

                data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))
                return data_matrix

    print(len(data_list))
    print(m)

    data_matrix = np.array(data_list, dtype=int)

    data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))

    return data_matrix

def Multiband2Array(path,channels):

    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s'% path)
        sys.exit(1)

    xcount=src_ds.RasterXSize # 宽度
    ycount=src_ds.RasterYSize # 高度
    ibands=src_ds.RasterCount # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    # if ibands==4:ibands=3
    ibands=min(channels,ibands)
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band) # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float16)
        if ibands==1:return dataraster.reshape((ycount,xcount))
        if band==1:
            data=dataraster.reshape((ycount,xcount,1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data=np.append(data,dataraster.reshape((ycount,xcount,1)),axis=2)

    return data

def next_batch(data, batch_size, flag, img_pixel=3, channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    if 1==flag:
        start_index = 0
    # start_index = 0
    second_index = start_index + batch_size

    if second_index > len(data):
        second_index = len(data)

    data1 = data[start_index:second_index]
    # print('start_index', start_index, 'second_index', second_index)

    start_index = second_index
    if start_index >= len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提取出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    img = img.astype(np.float32)  # 类型转换

    label = data1[:, img_pixel * img_pixel * channels:]
    label = label.reshape([-1, 1])
    label = label.astype(int)  # 类型转换

    return img, label

start_index=0
def next_batch(data,batch_size,img_pixel=9,channels=3):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    second_index=start_index+batch_size
    if second_index>len(data):
        second_index=len(data)
    data1=data[start_index:second_index]
    # lab=labels[start_index:second_index]
    start_index=second_index
    if start_index>=len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提起出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    # img=img.astype(float) # 类型转换

    label = data1[:, -1]
    label = label.astype(int)  # 类型转换

    return img,label


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # 从标量类标签转换为一个one-hot向量
    num_labels = labels_dense.shape[0]        #label的行数
    index_offset = np.arange(num_labels) * num_classes
    # print index_offset
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def dense_to_one_hot2(labels_dense, num_classes):
    labels_dense = np.array(labels_dense, dtype=np.uint8).reshape([-1, 1])  # [160000,1]
    num_labels = labels_dense.shape[0]  # 标签个数 160000
    labels_one_hot = np.zeros((num_labels, num_classes), np.uint8)  # [160000,3]
    for i, itenm in enumerate(labels_dense):
        labels_one_hot[i, itenm] = 1
        # 如果labels_dense不是int类型，itenm就不是int，此时做数组的切片索引就会报错，
        # 数组索引值必须是int类型，也可以 int(itenm) 强制转成int
        # labels_one_hot[i, :][itenm] = 1
    return labels_one_hot
