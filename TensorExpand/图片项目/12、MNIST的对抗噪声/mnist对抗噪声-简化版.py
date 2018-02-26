# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# 载入数据
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data/', one_hot=True)

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# TensorFlow图（Graph）
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# 对抗噪声的界限设为0.35，则噪声在正负0.35之间。
noise_limit = 0.35

# 下面的权重决定了与常规的损失度量相比，L2-loss的重要性。通常接近零的L2权重表现的更好。
noise_l2_weight = 0.02
# 当我们为噪声创建变量时，必须告知TensorFlow它属于哪一个变量集合，这样，后面就能通知两个优化器要更新哪些变量。
# 首先为变量集合定义一个名称。这只是一个字符串。
ADVERSARY_VARIABLES = 'adversary_variables'

collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]

x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise', trainable=False, # 不可训练，需手动更新
                      collections=collections)

# 对抗噪声会被限制在我们上面设定的噪声界限内。
# 注意此时并未在计算图表内进行计算，在优化步骤之后执行，详见下文。
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))

# 噪声图像只是输入图像和对抗噪声的总和。
x_noisy_image = x_image + x_noise
'''
x_pretty = pt.wrap(x_noisy_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\ # [n,28,28,16]
        max_pool(kernel=2, stride=2).\ [n,14,14,16]
        conv2d(kernel=5, depth=36, name='layer_conv2').\ [n,14,14,36]
        max_pool(kernel=2, stride=2).\ [n,7,7,36]
        flatten().\
        fully_connected(size=128, name='layer_fc1').\ 
        softmax_classifier(num_classes=num_classes, labels=y_true)
'''
conv1=tf.layers.conv2d(x_noisy_image,16,5,padding='SAME',activation=tf.nn.relu,name='layer_conv1')
pool1=tf.layers.max_pooling2d(conv1,2,2,padding='SAME')

conv2=tf.layers.conv2d(pool1,36,5,padding='SAME',activation=tf.nn.relu,name='layer_conv2')
pool2=tf.layers.max_pooling2d(conv2,2,2,padding='SAME')

fc1=tf.layers.dense(tf.layers.flatten(pool2),128,activation=tf.nn.relu,name='layer_fc1')
y_pred=tf.layers.dense(fc1,num_classes,activation=tf.nn.softmax,name='layer_output')
loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_true,logits=y_pred))

# 查看训练的变量
train_var=[var.name for var in tf.trainable_variables()]
print(train_var)
'''
['layer_conv1/kernel:0', 'layer_conv1/bias:0', 
'layer_conv2/kernel:0', 'layer_conv2/bias:0', 
'layer_fc1/kernel:0', 'layer_fc1/bias:0', 
'layer_output/kernel:0', 'layer_output/bias:0']
'''

# 优化除x_noise以外的可训练的变量
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# 对抗噪声的优化器
# 获取变量列表，这些是需要在第二个程序里为对抗噪声做优化的变量。
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)

# 展示变量名称列表。这里只有一个元素，是我们在上面创建的对抗噪声变量。
train_var2=[var.name for var in adversary_variables]
print(train_var2)
# ['x_noise:0']

# 我们会将常规优化的损失函数与所谓的L2-loss相结合。这将会得到在最佳分类准确率下的最小对抗噪声
# L2-loss由一个通常设置为接近零的权重缩放。
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)

# 将正常的损失函数和对抗噪声的L2-loss相结合。
loss_adversary = loss + l2_loss_noise

# 现在可以为对抗噪声创建优化器。由于优化器并不会更新神经网络的所有变量，
# 我们必须给出一个需要更新的变量的列表，即对抗噪声变量。注意，这里的学习率比上面的常规优化器要大很多。
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables) # 只更新x_noise:0

y_pred_cls = tf.argmax(y_pred, axis=1)

# 接着创建一个布尔数组，用来表示每张图像的预测类型是否与真实类型相同。
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 运行TensorFlow
session = tf.Session()

session.run(tf.global_variables_initializer())

# 帮助函数将对抗噪声初始化/重置为零。
def init_noise():
    session.run(tf.variables_initializer([x_noise]))

# 用来优化迭代的帮助函数
train_batch_size = 64

def optimize(num_iterations=1000,adversary_target_cls=None):
    for step in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        if adversary_target_cls is not None:
            # Set all the class-labels to zero.
            y_true_batch = np.zeros_like(y_true_batch)

            # Set the element for the adversarial target-class to 1.
            y_true_batch[:, adversary_target_cls] = 1.0  # 重设置目标标签

        session.run(optimizer,feed_dict={x:x_batch,y_true:y_true_batch})

        if step%50==0:
            train_acc=session.run(accuracy,{x:x_batch,y_true:y_true_batch})
            test_acc=session.run(accuracy,{x:data.test.images,y_true:data.test.labels})
            print('step',step,'|','train_acc',train_acc,'|','test_acc',test_acc)

    print('\n','true',np.argmax(data.test.labels[:10],axis=1),'pred',session.run(y_pred_cls,{x:data.test.images[:10]}))


'''
# 调用函数来初始化对抗噪声。
init_noise()
# 先更新loss网络变量，不更新x_noise变量
optimize(num_iterations=1000)
# 生成对抗噪声训练，仅更新x_noise变量
optimize(num_iterations=1000,adversary_target_cls=3) # 重置目标值为3
'''
# 免疫对抗噪声
# 调用函数来初始化对抗噪声。
init_noise()

optimize(num_iterations=100,adversary_target_cls=3) # 重置目标值为3

optimize(num_iterations=1000)
