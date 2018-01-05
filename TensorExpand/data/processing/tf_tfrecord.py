#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
tensorflow 操作tfrecord数据
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image
import os

class tf_tfrecord():
    def __init__(self,file_path=None,record_location=None,
                 data=None,label=None,h=28,w=28,c=1,batch_size=32,save_num=100):
        '''    
        :param file_path: 图像文件路径
        :param record_location: tfrecord文件保存路径
        :param data: 样本数据
        :param labels: 标签 非one_hot
        :param h: 样本的高度
        :param w: 样本的宽度
        :param c: 样本的通道数（波段数）
        :param batch_size: 每批次训练的样本数
        :param save_num: 每隔多少张图片保存成一个tfrecord文件
        '''
        self.file_path=file_path
        self.record_location = record_location
        self.data=data
        self.label=label
        self.h=h
        self.w=w
        self.c=c
        self.batch_size=batch_size
        self.save_num=save_num

    def numpy_to_tfrecord(self):
        writer = tf.python_io.TFRecordWriter(self.record_location)

        for i in range(len(self.data)):
            image = np.reshape(self.data[i], [self.h, self.w, self.c]).astype(np.float32)
            image_bytes = image.astype(np.float16).tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                # 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.uint8(self.label[i])])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())
        writer.close()

    def tfrecord_to_numpy(self):
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(self.record_location))  # 加载多个Tfrecode文件
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
        image = tf.reshape(record_image, [self.h, self.w, self.c])
        label = tf.cast(features['label'], tf.int64)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + self.batch_size
        data,label = tf.train.shuffle_batch(
            [image,label], batch_size=self.batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )
        return [data,label]

    def Image_to_tfrecord(self):
        '''
        直接将图像转成tfrecord
        :param record_location: tfr文件保存位置
        :param img_pixel: 设置图像固定大小
        :param save_num: 每隔多少张图片保存成一个tfrecord文件
        :return: 
        '''
        images_path=glob(self.file_path) #获取所有图像路径

        writer = None
        current_index = 0
        for image_path in images_path:

            if current_index % self.save_num == 0: # 每隔100幅图像，训练样本的信息就被写入到一个新的Tfrecode文件中，以加快操作的进程
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=self.record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1

            file_name=image_path.split('\\')[-2] # 获取文件名，将文件名作为标签  Linux 使用 '/'
            img=Image.open(image_path) # 这里可以使用opencv，skimage，scipy，gdal等对图像操作
            """这里可以加入对图像的预处理"""
            img=img.resize([self.h,self.w])
            image_bytes=np.array(img,np.float16).tobytes() # 将图片转化为bytes
            image_label = file_name.encode("utf-8") # 这里直接使用文件名作标签

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())

        if writer:
            writer.close()

    def tfrecord_to_numpy(self):
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(self.file_path))  # 加载多个Tfrecode文件
        reader = tf.TFRecordReader()
        _, serialized = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                # 'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
            })

        record_image = tf.decode_raw(features['image'], tf.float16)
        image = tf.reshape(record_image, [self.h, self.w,self.c])
        # label = tf.cast(features['label'], tf.int64)
        label = tf.cast(features['label'], tf.string)

        # label string-->int 0,1 标签
        label = tf.case({tf.equal(label, tf.constant('ants')): lambda: tf.constant(0),
                         tf.equal(label, tf.constant('bees')): lambda: tf.constant(1),
                         }, lambda: tf.constant(-1), exclusive=True)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + self.batch_size
        data,label = tf.train.shuffle_batch(
            [image,label], batch_size=self.batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )

        # 将图像转换为灰度值位于[0,1)的浮点类型，
        float_image_batch = tf.image.convert_image_dtype(data, tf.float32)
        return float_image_batch, label
        # return data,label

    def Image_processing(self,reshaped_image):
        '''
        图像预处理
        参考：https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
        :return: 
        '''
        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [self.h, self.w, self.c])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        return float_image.set_shape([self.h, self.w, self.c])

    def Image_to_tfrecord_2(self):
        writer = tf.python_io.TFRecordWriter(self.record_location + "train.tfrecords")
        for _, dirs, _ in os.walk(self.file_path):
            for filename in dirs:  # 文件夹名 取文件名作为标签
                file_path = os.path.join(dir_name, filename)  # 文件夹路径
                # for _ , _,img in os.walk(file_path):
                for img_name in os.listdir(file_path):  # 文件夹下的文件名
                    imgae_path = os.path.join(file_path, img_name)  # 文件路径
                    img = Image.open(imgae_path)
                    img = img.resize((self.h, self.w))
                    img_raw = np.array(img,np.uint8).tobytes()  # 将图片转化为原生bytes
                    example = tf.train.Example(features=tf.train.Features(feature={
                        # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(filename)])),
                        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode("utf-8")])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
                    writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()

    def Image_to_tfrecord_3(self,sess):
        '''
        的decode_jpeg解析jpg图像转成tfrecord
        :param record_location: tfr文件保存位置
        :param img_pixel: 设置图像固定大小
        :return: 
        '''
        images_path = glob(self.file_path)  # 获取所有图像路径

        writer = None
        current_index = 0
        for image_path in images_path:

            if current_index % self.save_num == 0:  # 每隔100幅图像，训练样本的信息就被写入到一个新的Tfrecode文件中，以加快操作的进程
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=self.record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1

            file_name = image_path.split('\\')[-2]  # 获取文件名，将文件名作为标签  Linux 使用 '/'

            # 使用tf 的decode_jpeg解析jpg图像，只能解析jpg图像
            try:
                image_file = tf.read_file(image_path)
                image = tf.image.decode_jpeg(image_file)
                # grayscale_image = tf.image.rgb_to_grayscale(image)  # 转成灰度
                resized_image = tf.image.resize_images(image, (self.h, self.w))
                image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            except:
                print(image_filename)
                continue

            ''' 使用PIL
            img = Image.open(image_path)  # 这里可以使用opencv，skimage，scipy，gdal等对图像操作
            """这里可以加入对图像的预处理"""
            img = img.resize([img_pixel, img_pixel])
            image_bytes = np.array(img, np.uint8).tobytes()  # 将图片转化为bytes
            '''
            image_label = file_name.encode("utf-8")  # 这里直接使用文件名作标签

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())

        if writer:
            writer.close()


if __name__=="__main__":
    """
    data = np.array([[3, 4, 1], [2, 4, 0], [1, 3, 1]])
    d = tf_tfrecord('123.tfrecord',data)
    # d.numpy_to_tfrecord()

    x_train_batch,y_train_batch=d.tfrecord_to_numpy(10)
    print(x_train_batch.get_shape)
    print(y_train_batch.get_shape)
    """

    sess = tf.InteractiveSession()

    d=tf_tfrecord(r'C:\Users\Administrator\Desktop\test3\hymenoptera_data\train\*\*.jpg')
    record_location='./images/image'
    if not os.path.exists('images'):os.mkdir('./images')
    d.Image_to_tfrecord_3(record_location,224,sess)

    d2=tf_tfrecord('./images/*.tfrecords')

    x_train_batch, y_train_batch = d2.tfrecord_to_numpy(224,2)
    print(x_train_batch.get_shape)
    print(y_train_batch.get_shape)



    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # print(sess.run(y_train_batch))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            curr_x_train_batch,curr_y_train_batch = sess.run([x_train_batch,y_train_batch])
            print(curr_x_train_batch[0])
            exit(0)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
