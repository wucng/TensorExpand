# -*- coding=utf8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas
import glob
import os,sys,pickle

'''
1、图片二值化处理，缩放成统一大小

2、将处理后的图片数据转成pickle文件
'''

train_images_path=glob.glob('./HWDB1/train/*/*.png') # 获取图片路径
np.random.shuffle(train_images_path) # 打乱路径，启动打乱数据的目的
print(len(train_images_path)) # 895035

# 打印二值化处理过程
def plot_binary_image(image_path):
    img = cv2.imread(image_path, 0)

    ret2, th2 = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th=cv2.bitwise_not(th2)

    # ret2, th2 = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY_INV)

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

# plot_binary_image(train_images_path[10])

# 帮助函数 实现图像二值化,统一图像尺寸，像素值转成0、1
def binary_image(image_path):
    img = cv2.imread(image_path, 0)
    '''
    ret2, th2 = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th=cv2.bitwise_not(th2)

    img_=cv2.resize(th,dsize=(64,64))/255. # 缩放到统一大小,像素值转成0、1
    '''
    ret, img = cv2.threshold(img, np.mean(img), 1, cv2.THRESH_BINARY_INV)
    img_ = cv2.resize(img, dsize=(32, 32))  # 缩放到统一大小,像素值转成0、1
    label=int(image_path.split('/')[-2]) # 对应的标签值

    return [img_,label]

# %%time
def image2pickle(images_path,save_path):
    datas=[]
    [datas.append(data) for path in images_path for data in binary_image(path)]

    # 保存成pickle文件
    with open(save_path, 'wb') as file:
        pickle.dump(datas, file)

if not os.path.exists('data'):
    os.mkdir('data')

save_size=50000
for i in range(len(train_images_path)//save_size):
    start=i*save_size
    end=min(((i+1)*save_size,len(train_images_path)))
    image2pickle(train_images_path[start:end],'./data/train_'+str(i)+'.pickle')

# test数据同样处理
test_images_path=glob.glob('./HWDB1/test/*/*.png') # 获取图片路径
print(len(test_images_path)) # 223991
for i in range(len(test_images_path)//save_size):
    start=i*save_size
    end=min(((i+1)*save_size,len(test_images_path)))
    image2pickle(test_images_path[start:end],'./data/test_'+str(i)+'.pickle')

exit(-1)
