# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
import os

class Data_interface(object):
    def __init__(self,data_path='cifar.data',hight=28,width=28,batch_size=128):
        self.data_path=data_path
        self.hight=hight
        self.width=width
        self.batch_size=batch_size
        self.start_index=0
        self.end_index=0

        self.read_file()

    def read_file(self):
        with open(self.data_path,'r') as fp:
            for p in fp:
                p=p.strip('\n').strip('\r').strip().replace(' ','')
                if 'classes' in p:
                    self.classes=int(p.split('=')[-1])
                if 'train' in p:
                    self.train = str(p.split('=')[-1])
                if 'valid' in p:
                    self.valid = str(p.split('=')[-1])

                if 'labels' in p:
                    self.labels = str(p.split('=')[-1])

                if 'backup' in p:
                    self.backup = str(p.split('=')[-1])

        with open(self.labels,'r') as fp:
            data = fp.read().splitlines()  # 删除换行符

            self.classes2id=dict(zip(data,range(self.classes))) # 类名与id对应

            self.train_data=self.data_select(self.train)

            self.valid_data = self.data_select(self.valid)

    def data_select(self,path):
        with open(path,'r') as fp:
            data = fp.read().splitlines()  # 删除换行符
            return data

    def data_shuffle(self,data):
        np.random.shuffle(data)

    def data_next_batch(self,data):
        # steps=len(data)//self.batch_size
        self.end_index=min((self.batch_size+self.start_index,len(data)))
        batch_data=data[self.start_index:self.end_index]

        self.batch_data=self.to_img(batch_data)

        self.start_index=self.end_index
        if self.start_index==len(data):
            self.start_index=0


    def to_img(self,data):
        imgs=[]
        labels=[]
        for da in data:
            img=cv2.imread(da)
            img=cv2.resize(img,(self.width,self.hight))
            img=img/255.-0.5 # 归一化处理
            # img = tf.image.resize_image_with_crop_or_pad(img, self.hight, self.width)
            # img=tf.image.random_hue(img,0.1)
            # img=tf.image.random_flip_left_right(img)
            # img=tf.image.random_contrast(img,0.5,0.75)
            # img=tf.image.random_brightness(img,0.75)

            label=os.path.basename(da).split('_')[-1].split('.')[0]
            label=int(self.classes2id[label]) # 换成 id

            imgs.append(img)
            labels.append(label)

        return np.asarray(imgs,np.float32),np.asarray(labels,np.int64) #

if __name__=='__main__':
    data=Data_interface('cfg/cifar.data')

    data.data_shuffle(data.train_data)

    data.data_next_batch(data.train_data)

    print(data.batch_data[0].shape)

    print(data.batch_data[1].shape)