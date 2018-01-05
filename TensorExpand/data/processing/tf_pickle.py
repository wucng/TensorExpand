#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


"""
image存储成pickle
"""
import numpy as np
import pickle
import os
import zlib
import gzip
import gdal
import sys
# import cv2
import m1
# from PIL import Image
import datetime

def create_pickle_train(image_path, mask_path, pkl_path, img_pixel=10, channels=3):
    m = 0
    n=0
    # image_data = Multiband2Array(image_path)
    image_data=m1.Multiband2Array(image_path)
    # mask_data = cv2.split(cv2.imread(mask_path))[0] / 255
    # mask_data=np.asarray(Image.open(mask_path))//255

    mask_data=m1.Multiband2Array(mask_path)//255


    x_size, y_size = image_data.shape[:2]

    data_list = []
    flag=True
    for i in range(0, x_size - img_pixel + 1, img_pixel // 2):  # 文件夹下的文件名
        if not flag:break
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel // 2):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            train_label = mask_data[i:i + img_pixel, j:j + img_pixel].max()
            # train_label = 1
            # train_label = mask_data[i:i + img_pixel, j:j + img_pixel].min()
            # train_label = int(mask_data[i:i + img_pixel, j:j + img_pixel].sum() / (img_pixel*img_pixel/2+1))
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签
            data_list.append(data2)

            m += 1
            if m >=10000000:
                data_matrix = np.array(data_list, dtype=np.float32)
                data_matrix = data_matrix.reshape((-1, 301))
                with gzip.open(pkl_path+'_'+str(n)+'.pkl', 'wb') as writer:  # 以压缩包方式创建文件，进一步压缩文件
                    pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
                data_list=[]
                m=0
                n+=1
                flag=False
                break
            # if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))
    # print(m)
    # data_matrix = np.array(data_list, dtype=int)
    if data_list!=[]:
        data_matrix = np.array(data_list, dtype=np.float32)
        data_matrix = data_matrix.reshape((-1, 301))
        data_matrix=data_matrix.astype(np.float32)
        # data_matrix = data_matrix.tostring()  # 转成byte，缩小文件大小
        with gzip.open(pkl_path+'.pkl', 'wb') as writer:  # 以压缩包方式创建文件，进一步压缩文件
            pickle.dump(data_matrix, writer)  # 数据存储成pickle文件


# def read_and_decode(filename, img_pixel=isize, channels=img_channel):
def read_and_decode(filename):
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        data = pickle.load(pkl_file)  # 加载数据

    return data


'''
其他工具
'''


# ---------------生成多列标签 如：0,1 对应为[1,0],[0,1]------------#
# 单列标签转成多列标签
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # 从标量类标签转换为一个one-hot向量
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    # print index_offset
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def dense_to_one_hot2(labels_dense,num_classes):
    labels_dense=np.array(labels_dense,dtype=np.uint8)
    num_labels = labels_dense.shape[0] # 标签个数
    labels_one_hot=np.zeros((num_labels,num_classes),np.uint8)
    for i,itenm in enumerate(labels_dense):
        labels_one_hot[i,itenm]=1
        # 如果labels_dense不是int类型，itenm就不是int，此时做数组的切片索引就会报错，
        # 数组索引值必须是int类型，也可以 int(itenm) 强制转成int
        # labels_one_hot[i, :][itenm] = 1
    return labels_one_hot
# ------------next_batch------------#
'''
注：
每次 data传入next_batch()完成，进行下一次传入时，先进行打乱
如下面的做法：

total_batch = int(img_nums / batch_size)
data=read_and_decode(filename,img_pixel=isize,channels=3)

for epoch in range(training_epochs):
    # 将数据按行打乱
    index = [i for i in range(len(data))]  # len(data)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data = data[index]
    for i in range(total_batch):
        img, label=next_batch(data,batch_size,img_pixel=isize,channels=img_channel)
        ......
'''


# 按batch_size提取数据
# batch_size为每次批处理样本数
# data包含特征+标签 每一行都是 特征+标签

start_index = 0
def next_batch(data, batch_size, img_pixel=3, channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    second_index = start_index + batch_size

    if second_index > len(data):
        second_index = len(data)
    data1 = data[start_index:second_index]
    # lab=labels[start_index:second_index]
    start_index = second_index
    if start_index >= len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提起出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    img = img.astype(np.float32)  # 类型转换

    label = data1[:, -1]
    label = label.astype(int)  # 类型转换

    return img, label

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("startTime: ", start_time)

    print('找到对应的data与label图像')
    data_images=[]
    label_images=[]

    data_path=r'E:\耕地样本数据\test'

    for root, dirs, files in os.walk(data_path):
        for file in files:
            code_file = os.path.join(root, file)
            if code_file.endswith('.tif'): # 找出所有.tif文件
                if 'mask' in file:
                    label_images.append(code_file)
                    file1=file.replace('_mask','')
                    data_images.append(os.path.join(root, file1))


    for i in range(len(data_images)):
        # 生成训练集
        # image_path = r"L15-3322E-2473N_01.tif"
        # mask_path = r"L15-3322E-2473N_01_mask.tif"
        image_path=data_images[i]
        mask_path=label_images[i]
        pkl_path = image_path[0:-4] #+ ".pkl"
        print("影像路径：", image_path)
        print("掩模路径：", mask_path)
        print("序列化文件：", pkl_path)

        create_pickle_train(image_path, mask_path, pkl_path, img_pixel=10, channels=3)

    end_time = datetime.datetime.now()
    print("endTime: ", end_time)
    print("seconds used: ", (end_time - start_time).seconds)








