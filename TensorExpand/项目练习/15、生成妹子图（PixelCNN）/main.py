# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import cv2

# 如果使用mnist数据集，把MNIST设置为True
MNIST = True

if MNIST == True:
    from tensorflow.examples.tutorials.mnist import input_data

    data = input_data.read_data_sets('./MNIST_data')
    image_height = 28
    image_width = 28
    image_channel = 1

    batch_size = 128
    n_batches = data.train.num_examples // batch_size
else:
    picture_dir = 'little_girls'
    picture_list = []
    # 建议不要把图片一次加载到内存，为了节省内存，最好边加载边使用
    for (dirpath, dirnames, filenames) in os.walk(picture_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                picture_list.append(os.sep.join([dirpath, filename]))

    print("图像总数: ", len(picture_list))

    # 图像大小和Channel
    image_height = 64
    image_width = 64
    image_channel = 3

    # 每次使用多少样本训练
    batch_size = 128
    n_batches = len(picture_list) // batch_size

    # 图片格式对应输入X
    img_data = []
    for img_file in picture_list:
        img_data.append(cv2.imread(img_file))
    img_data = np.array(img_data)
    img_data = img_data / 255.0
# print(img_data.shape)   # (44112, 64, 64, 3)


X = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel]) # [-1,28,28,1]


def gated_cnn(W_shape_, fan_in, gated=True, payload=None, mask=None, activation=True):
    W_shape = [W_shape_[0], W_shape_[1], fan_in.get_shape()[-1], W_shape_[2]]
    b_shape = W_shape_[2]

    def get_weights(shape, name, mask=None):
        weights_initializer = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(name, shape, tf.float32, weights_initializer)

        if mask:
            filter_mid_x = shape[0] // 2
            filter_mid_y = shape[1] // 2
            mask_filter = np.ones(shape, dtype=np.float32)
            mask_filter[filter_mid_x, filter_mid_y + 1:, :, :] = 0.
            mask_filter[filter_mid_x + 1:, :, :, :] = 0.

            if mask == 'a':
                mask_filter[filter_mid_x, filter_mid_y, :, :] = 0.

            W *= mask_filter
        return W

    if gated:
        W_f = get_weights(W_shape, "v_W", mask=mask)
        W_g = get_weights(W_shape, "h_W", mask=mask)

        b_f = tf.get_variable("v_b", b_shape, tf.float32, tf.zeros_initializer)
        b_g = tf.get_variable("h_b", b_shape, tf.float32, tf.zeros_initializer)

        conv_f = tf.nn.conv2d(fan_in, W_f, strides=[1, 1, 1, 1], padding='SAME') # [-1,28,28,32]
        conv_g = tf.nn.conv2d(fan_in, W_g, strides=[1, 1, 1, 1], padding='SAME') # [-1,28,28,32]
        if payload is not None:
            conv_f += payload
            conv_g += payload

        fan_out = tf.multiply(tf.tanh(conv_f + b_f), tf.sigmoid(conv_g + b_g)) # [-1,28,28,32]
    else:
        W = get_weights(W_shape, "W", mask=mask)
        b = tf.get_variable("b", b_shape, tf.float32, tf.zeros_initializer)
        conv = tf.nn.conv2d(fan_in, W, strides=[1, 1, 1, 1], padding='SAME') # [-1,28,28,32]
        if activation:
            fan_out = tf.nn.relu(tf.add(conv, b))
        else:
            fan_out = tf.add(conv, b)

    return fan_out # [-1,28,28,32]


def pixel_cnn(layers=12, f_map=32):
    v_stack_in, h_stack_in = X, X

    for i in range(layers):
        filter_size = 3 if i > 0 else 7
        mask = 'b' if i > 0 else 'a'
        residual = True if i > 0 else False
        i = str(i)

        with tf.variable_scope("v_stack" + i):
            v_stack = gated_cnn([filter_size, filter_size, f_map], v_stack_in, mask=mask) # [-1,28,28,32]
            v_stack_in = v_stack

        with tf.variable_scope("v_stack_1" + i):
            v_stack_1 = gated_cnn([1, 1, f_map], v_stack_in, gated=False, mask=mask) # [-1,28,28,32]

        with tf.variable_scope("h_stack" + i):
            h_stack = gated_cnn([1, filter_size, f_map], h_stack_in, payload=v_stack_1, mask=mask) # [-1,28,28,32]

        with tf.variable_scope("h_stack_1" + i):
            h_stack_1 = gated_cnn([1, 1, f_map], h_stack, gated=False, mask=mask) # [-1,28,28,32]
            if residual:
                h_stack_1 += h_stack_in
            h_stack_in = h_stack_1

    with tf.variable_scope("fc_1"):
        fc1 = gated_cnn([1, 1, f_map], h_stack_in, gated=False, mask='b') # [-1,28,28,32]

    color = 256
    with tf.variable_scope("fc_2"):
        fc2 = gated_cnn([1, 1, image_channel * color], fc1, gated=False, mask='b', activation=False) # [-1,28,28,1*256]
        fc2 = tf.reshape(fc2, (-1, color)) # [batch*28*28*1,256]

        return fc2


def train_pixel_cnn():
    output = pixel_cnn()

    loss = tf.reduce_mean(  # tf.reshape(X, [-1])  [batch*28*28*1,256]
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.cast(tf.reshape(X, [-1]), dtype=tf.int32)))
    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(loss)
    clipped_gradients = [(tf.clip_by_value(_[0], -1, 1), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(50):
            for batch in range(n_batches):

                if MNIST == True:
                    batch_X, _ = data.train.next_batch(batch_size)
                    batch_X = batch_X.reshape([batch_size, image_height, image_width, image_channel])
                else:
                    batch_X = img_data[batch_size * batch: batch_size * (batch + 1)]

                _, cost = sess.run([optimizer, loss], feed_dict={X: batch_X})
                print("epoch:", epoch, '  batch:', batch, '  cost:', cost)
            if epoch % 7 == 0:
                saver.save(sess, "girl.ckpt", global_step=epoch)

# 训练
train_pixel_cnn()


def generate_girl():
    output = pixel_cnn()  # [batch*28*28*1,256]
    # [batch*28*28*1,256]--> [batch*28*28*1,1]-->[batch,28,28,1]
    predict = tf.reshape(tf.multinomial(tf.nn.softmax(output), num_samples=1, seed=100), tf.shape(X)) # [batch,28,28,1]
    # predict_argmax = tf.reshape(tf.argmax(tf.nn.softmax(output), dimension=tf.rank(output) - 1), tf.shape(X))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, 'girl.ckpt-49')

        pics = np.zeros((1 * 1, image_height, image_width, image_channel), dtype=np.float32)

        for i in range(image_height):
            for j in range(image_width):
                for k in range(image_channel):
                    next_pic = sess.run(predict, feed_dict={X: pics})
                    pics[:, i, j, k] = next_pic[:, i, j, k]

        cv2.imwrite('girl.jpg', pics[0])
        print('生成妹子图: girl.jpg')

# 生成图像
generate_girl()