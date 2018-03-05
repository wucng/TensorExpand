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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="The number of iterations", type=int, default=100) # 迭代次数
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=256) # 每步训练的样本数
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-4) # 学习效率
parser.add_argument("-k", "--keep", help="keep rate", type=float, default=0.8) # dropout out保持率
parser.add_argument("-m", "--mode", help="1 train,0 test", type=int, default=1) # 模式，训练or测试
parser.add_argument("-ld", "--logdir", help="model save path", type=str, default='./checkpoint/') # 模型参数保存位置

args = parser.parse_args()
print("args:",args)


# 设置超参数
train_num_images=895035 # 训练样本数
test_num_images=223991 # 测试样本数
num_class=3755 # 分类数
image_size=32 # 样本数据大小
batch_size=args.batch_size # 每步训练的样本数
learning_rate=args.learning_rate # 学习率
global_step = tf.Variable(0, name="global_step", trainable=False)
# learning_rate=tf.train.polynomial_decay(1e-4,global_step,train_num_images*2//batch_size,1e-8)
epochs=args.epochs # 数据迭代次数
train=args.mode # 选择模式 1 train,0 test
keep=args.keep # dropout out保持率
logdir=args.logdir # 模型参数存放位置

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

    min_after_dequeue =train_num_images//batch_size #1000
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
        steps = train_num_images // batch_size
    if train==0:
        x_test_batch, y_test_batch = tfrecord_to_numpy(glob('./data/test-*.tfrecords'),batch_size=batch_size)
        steps = test_num_images // batch_size
        acc_list = []

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    saver=tf.train.Saver(tf.global_variables())

    if not os.path.exists(logdir): os.mkdir(logdir)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        if train == 1:  # 训练
            while not coord.should_stop():

                for epoch in range(epochs):
                    # 每个epoch重新加载上一个epoch的模型参数训练
                    ckpt = tf.train.get_checkpoint_state(logdir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                    for step in range(steps):

                        # Run training steps or whatever
                        curr_x_train_batch,curr_y_train_batch = sess.run([x_train_batch,y_train_batch])
                        sess.run([optimizer,global_step],{x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:keep,is_training:True})
                        # optimizer.run({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:keep,is_training:True})

                        if step %20==0:
                            # acc=accuracy.eval({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:1.,is_training:True})
                            acc = accuracy.eval(
                                {x: curr_x_train_batch, y_: curr_y_train_batch, keep_rate: 1., is_training: True})
                            loss=cost.eval({x:curr_x_train_batch,y_:curr_y_train_batch,keep_rate:1.,is_training:True})
                            print('epoch',epoch,'|','step',step,'|','acc',acc,'|','loss',loss)

                    # 每个epoch保存下模型参数
                    saver.save(sess, logdir + 'model.ckpt')

                break

        if train==0: # 测试
            # 验证之前是否已经保存了检查点文件
            ckpt = tf.train.get_checkpoint_state(logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            while not coord.should_stop():
                for step in range(steps):
                    curr_x_test_batch, curr_y_test_batch = sess.run([x_test_batch, y_test_batch])
                    if step % 20 == 0:
                        acc = accuracy.eval({x: curr_x_test_batch, y_: curr_y_test_batch, keep_rate: 1., is_training: True})
                        acc_list.append(acc)
                        print('step', step, '|', 'acc', acc)

                # 取平均值
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
