# -*- coding: utf-8 -*-

'''
从pool_3开始重新训练参数
'''
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# import time
# from datetime import timedelta
import os

# Functions and classes for loading and using the Inception model.
import inception # 模型文件
import PIL.Image
import glob

# 加载数据
def load_image(image_path):
    img_path = glob.glob(image_path)  #
    imgs = []
    labels = []
    for path in img_path:
        image = np.array(PIL.Image.open(path).resize((299, 299))) #/255.
        imgs.append(image)
        if path.strip().split('/')[-2] == 'ants':
            labels.append([0, 1])  # 1
        else:
            labels.append([1, 0])  # 0

    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs,labels
imgs, labels=load_image('../../hymenoptera_data/train/*/*.jpg')

# 下载Inception模型
# 这是你保存数据文件的默认文件夹
# inception.data_dir = 'inception/'
inception.maybe_download()
# 载入Inception模型
model = inception.Inception()
# from inception import transfer_values_cache

transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x') # [n,2048]
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

fc=tf.layers.dense(x,1024,activation=tf.nn.relu,name='layer_fc1')
y_pred=tf.layers.dense(fc,2,activation=tf.nn.softmax)
loss=tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

y_pred_cls = tf.argmax(y_pred, dimension=1)

# 创建一个布尔向量，表示每张图像的真实类别是否与预测类别相同。
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_size = 32
total_batchs = len(imgs) // batch_size
for epoch in range(20):
    index = np.arange(0, len(imgs), dtype=np.int32)
    np.random.shuffle(index)
    imgs = imgs[index]
    labels = labels[index]

    for step in range(total_batchs):
        batch_x = imgs[step * batch_size:(step + 1) * batch_size]
        batch_y = labels[step * batch_size:(step + 1) * batch_size]

        input_x=inception.feed_data(model,images=batch_x)

        sess.run(train_op, {x: input_x, y_true: batch_y})

        if step % 10 == 0:
            acc = sess.run(accuracy, {x: input_x, y_true: batch_y})
            print('epoch', epoch, '|', 'step', step, '|', 'acc', acc)

# test
imgs_test, labels_test = load_image('../../hymenoptera_data/val/*/*.jpg')

input_x=inception.feed_data(model,images=imgs_test[:20])
pred_y=sess.run(y_pred_cls,{x:input_x})
print('pred:',pred_y,'\n','real:',np.argmax(labels_test[:20],1))

model.close()
sess.close()
