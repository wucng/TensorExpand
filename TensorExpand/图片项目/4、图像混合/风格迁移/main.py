#! /usr/bin/python
# -*- coding: utf8 -*-
'''
参考：https://zhuanlan.zhihu.com/p/27697553
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import PIL.Image
from IPython.display import Image, display
# 下载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data


parser = argparse.ArgumentParser()
#获得buckets路径
parser.add_argument('--buckets', type=str, default='./MNIST_data',
                    help='input data path')
#获得checkpoint路径
parser.add_argument('--checkpointDir', type=str, default='model',
                    help='output model path')
FLAGS, _ = parser.parse_known_args()


mnist = input_data.read_data_sets(FLAGS.buckets, one_hot=True)

n_output_layer = 10


# 定义待训练的神经网络
def convolutional_neural_network(data):
    weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'w_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_output_layer]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_output_layer]))}

    data = tf.reshape(data, [-1, 28, 28, 1])

    conv1 = tf.nn.relu(
        tf.add(tf.nn.conv2d(data, weights['w_conv1'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv1'])) # [-1,28,28,32]
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [-1,14,14,32]

    conv2 = tf.nn.relu(
        tf.add(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv2'])) # [-1,14,14,64]
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [-1,7,7,64]

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64]) # [-1,7*7*64]
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc'])) # [-1,1024]

    # dropout剔除一些"神经元"
    # fc = tf.nn.dropout(fc, 0.8)

    output = tf.add(tf.matmul(fc, weights['out']), biases['out']) # [-1,10]
    return conv1,conv2,output


# 每次使用100条数据进行训练
batch_size = 100

X = tf.placeholder('float', [None, 28,28,1])
Y = tf.placeholder('float')


# 使用数据训练神经网络
def train_neural_network(X, Y):
    _,_,predict = convolutional_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001

    epochs = 1
    saver=tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(mnist.train.num_examples // batch_size):
                x, y = mnist.train.next_batch(batch_size)
                x=np.reshape(x,[-1,28,28,1])
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)

            saver.save(session,os.path.join(FLAGS.checkpointDir,'model.ckpt'),global_step=epoch)

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: mnist.test.images.reshape([-1,28,28,1]), Y: mnist.test.labels}))

# ----------------风格迁移---------------------------------#

def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array.
    return np.float32(image)

def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

def plot_image_big(image):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert pixels to bytes.
    image = image.astype(np.uint8)

    # Convert to a PIL-image and display it.
    display(PIL.Image.fromarray(image))

def plot_images(content_image, style_image, mixed_image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation,cmap='gray')
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation,cmap='gray')
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation,cmap='gray')
    ax.set_xlabel("Style")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# J(C,G)
def create_content_loss(session, content_image, layers=[]):
    values = session.run(layers, feed_dict={X: content_image})  # C
    layer_losses = []
    for value, layer in zip(values, layers):  # layer 为网络中的随机矩阵 把他做G
        value_const = tf.constant(value)
        loss = tf.losses.mean_squared_error(predictions=layer, labels=value_const)

        layer_losses.append(loss)

    total_loss = tf.reduce_mean(layer_losses)

    return total_loss


def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram

# J(S,G)
def create_style_loss(session, style_image, layers=[]):
    gram_layers = [gram_matrix(layer) for layer in layers]
    values = session.run(gram_layers, feed_dict={X: style_image})  # S

    layer_losses = []

    for value, gram_layer in zip(values, gram_layers):
        value_const = tf.constant(value)

        loss = tf.losses.mean_squared_error(predictions=gram_layer, labels=value_const)

        layer_losses.append(loss)

    total_loss = tf.reduce_mean(layer_losses)

    return total_loss


# 给混合图像去噪的损失函数
def create_denoise_loss(inputs):
    loss = tf.reduce_sum(tf.abs(inputs[:, 1:, :, :] - inputs[:, :-1, :, :])) + \
           tf.reduce_sum(tf.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :]))

    return loss

# 风格迁移
def style_transfer(session,content_image, style_image, content_layers,style_layers,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0
                   ):

    loss_content = create_content_loss(session=session,
                                       content_image=content_image,
                                       layers=content_layers)

    loss_style = create_style_loss(session=session,
                                   style_image=style_image,
                                   layers=style_layers)

    loss_denoise = create_denoise_loss(X)

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    loss_combined = weight_content * adj_content * loss_content + \
                    weight_style * adj_style * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

    gradient = tf.gradients(loss_combined, X)

    run_list = [gradient, update_adj_content, update_adj_style, \
                update_adj_denoise]

    mixed_image = np.random.rand(*content_image.shape) + 128

    for i in range(num_iterations):
        grad, adj_content_val, adj_style_val, adj_denoise_val \
            = session.run(run_list, feed_dict={X:mixed_image})

        grad = np.squeeze(grad)
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        mixed_image -= (grad * step_size_scaled).reshape([-1,28,28,1])
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        # # Print a little progress-indicator.
        print(". ", end="")

        # Display status once every 10 iterations, and the last.
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            # Print adjustment weights for loss-functions.
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            # Plot the content-, style- and mixed-images.
            plot_images(content_image=content_image.reshape([28,28]),
                        style_image=style_image.reshape([28,28]),
                        mixed_image=mixed_image.reshape([28,28]))

    print()
    print("Final image:")
    plot_image_big(mixed_image)

    # Close the TensorFlow session to release its resources.
    session.close()

    # Return the mixed-image.
    return mixed_image

def run_style_transfer(X):
    content_image=mnist.test.images[0].reshape([-1,28,28,1]) # [1,28,28,1]
    style_image=mnist.test.images[5].reshape([-1,28,28,1])# [1,28,28,1]

    layer1, layer2, _ = convolutional_neural_network(X)
    content_layers=[layer1, layer2]
    style_layers = [layer1, layer2]

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpointDir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)

    style_transfer(session,content_image,style_image,content_layers,style_layers)

    session.close()


if __name__=="__main__":
    # train_neural_network(X, Y)
    #
    run_style_transfer(X)
