# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Functions and classes for loading and using the Inception model.
import inception

# Inception 模型

# 从网上下载Inception模型

# 从网上下载Inception模型。这是你保存数据文件的默认文件夹。如果文件夹不存在就自动创建。

# inception.data_dir = 'inception/'
# 如果文件夹中不存在Inception模型，就自动下载。 它有85MB。

inception.maybe_download()

# 卷积层的名称
# 这个函数返回Inception模型中卷积层的名称列表。
def get_conv_layer_names():
    # Load the Inception model.
    model = inception.Inception()

    # Create a list of names for the operations in the graph
    # for the Inception model where the operator-type is 'Conv2D'.
    names = [op.name for op in model.graph.get_operations() if op.type == 'Conv2D']

    # Close the TensorFlow session inside the model-object.
    model.close()

    return names
conv_names = get_conv_layer_names()

# 在Inception模型中总共有94个卷积层。
print(len(conv_names))
# 94

# 写出头5个卷积层的名称。
print(conv_names[:5])
'''
['conv/Conv2D',
'conv_1/Conv2D',
'conv_2/Conv2D',
'conv_3/Conv2D',
'conv_4/Conv2D']
'''
# 写出最后5个卷积层的名称。
print(conv_names[-5:])
'''
['mixed_10/tower_1/conv/Conv2D',
'mixed_10/tower_1/conv_1/Conv2D',
'mixed_10/tower_1/mixed/conv/Conv2D',
'mixed_10/tower_1/mixed/conv_1/Conv2D',
'mixed_10/tower_2/conv/Conv2D']
'''
# 找到输入图像的帮助函数
'''
这个函数用来寻找使网络内给定特征最大化的输入图像。
它本质上是用梯度法来进行优化。图像用小的随机值初始化，
然后用给定特征关于输入图像的梯度来逐步更新。
'''


def optimize_image(conv_id=None, feature=0,
                   num_iterations=30, show_progress=True):
    """
    Find an image that maximizes the feature
    given by the conv_id and feature number.

    Parameters:
    conv_id: Integer identifying the convolutional layer to
             maximize. It is an index into conv_names.
             If None then use the last fully-connected layer
             before the softmax output.
    feature: Index into the layer for the feature to maximize.
    num_iteration: Number of optimization iterations to perform.
    show_progress: Boolean whether to show the progress.
    """

    # Load the Inception model. This is done for each call of
    # this function because we will add a lot to the graph
    # which will cause the graph to grow and eventually the
    # computer will run out of memory.
    model = inception.Inception()

    # Reference to the tensor that takes the raw input image.
    resized_image = model.resized_image

    # Reference to the tensor for the predicted classes.
    # This is the output of the final layer's softmax classifier.
    y_pred = model.y_pred

    # Create the loss-function that must be maximized.
    if conv_id is None:
        # If we want to maximize a feature on the last layer,
        # then we use the fully-connected layer prior to the
        # softmax-classifier. The feature no. is the class-number
        # and must be an integer between 1 and 1000.
        # The loss-function is just the value of that feature.
        loss = model.y_logits[0, feature]
    else:
        # If instead we want to maximize a feature of a
        # convolutional layer inside the neural network.

        # Get the name of the convolutional operator.
        conv_name = conv_names[conv_id]

        # Get a reference to the tensor that is output by the
        # operator. Note that ":0" is added to the name for this.
        tensor = model.graph.get_tensor_by_name(conv_name + ":0")

        # Set the Inception model's graph as the default
        # so we can add an operator to it.
        with model.graph.as_default():
            # The loss-function is the average of all the
            # tensor-values for the given feature. This
            # ensures that we generate the whole input image.
            # You can try and modify this so it only uses
            # a part of the tensor.
            loss = tf.reduce_mean(tensor[:, :, :, feature])

    # Get the gradient for the loss-function with regard to
    # the resized input image. This creates a mathematical
    # function for calculating the gradient.
    gradient = tf.gradients(loss, resized_image)

    # Create a TensorFlow session so we can run the graph.
    session = tf.Session(graph=model.graph)

    # Generate a random image of the same size as the raw input.
    # Each pixel is a small random value between 128 and 129,
    # which is about the middle of the colour-range.
    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    # Perform a number of optimization iterations to find
    # the image that maximizes the loss-function.
    for i in range(num_iterations):
        # Create a feed-dict. This feeds the image to the
        # tensor in the graph that holds the resized image, because
        # this is the final stage for inputting raw image data.
        feed_dict = {model.tensor_name_resized_image: image}

        # Calculate the predicted class-scores,
        # as well as the gradient and the loss-value.
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        # Squeeze the dimensionality for the gradient-array.
        grad = np.array(grad).squeeze()

        # The gradient now tells us how much we need to change the
        # input image in order to maximize the given feature.

        # Calculate the step-size for updating the image.
        # This step-size was found to give fast convergence.
        # The addition of 1e-8 is to protect from div-by-zero.
        step_size = 1.0 / (grad.std() + 1e-8)

        # Update the image by adding the scaled gradient
        # This is called gradient ascent.
        image += step_size * grad

        # Ensure all pixel-values in the image are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        if show_progress:
            print("Iteration:", i)

            # Convert the predicted class-scores to a one-dim array.
            pred = np.squeeze(pred)

            # The predicted class for the Inception model.
            pred_cls = np.argmax(pred)

            # Name of the predicted class.
            cls_name = model.name_lookup.cls_to_name(pred_cls,
                                                     only_first_name=True)

            # The score (probability) for the predicted class.
            cls_score = pred[pred_cls]

            # Print the predicted score etc.
            msg = "Predicted class-name: {0} (#{1}), score: {2:>7.2%}"
            print(msg.format(cls_name, pred_cls, cls_score))

            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))

            # Print the loss-value.
            print("Loss:", loss_value)

            # Newline.
            print()

    # Close the TensorFlow session inside the model-object.
    model.close()
    session.close()

    return image.squeeze()

# 绘制图像和噪声的帮助函数
# 函数对图像做归一化，则像素值在0.0到1.0之间。
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm

# 这个函数绘制一张图像。
def plot_image(image):
    # Normalize the image so pixels are between 0.0 and 1.0
    img_norm = normalize_image(image)

    # Plot the image.
    plt.imshow(img_norm, interpolation='nearest')
    plt.show()

# 这个函数在坐标系内绘制6张图。
def plot_images(images, show_size=100):
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """

    # Create figure with sub-plots.
    fig, axes = plt.subplots(2, 3)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and only use the desired pixels.
        img = images[i, 0:show_size, 0:show_size, :]

        # Normalize the image so its pixels are between 0.0 and 1.0
        img_norm = normalize_image(img)

        # Plot the image.
        ax.imshow(img_norm, interpolation=interpolation)

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# 优化和绘制图像的帮助函数
# 这个函数优化多张图像并绘制它们。
def optimize_images(conv_id=None, num_iterations=30, show_size=100):
    """
    Find 6 images that maximize the 6 first features in the layer
    given by the conv_id.

    Parameters:
    conv_id: Integer identifying the convolutional layer to
             maximize. It is an index into conv_names.
             If None then use the last layer before the softmax output.
    num_iterations: Number of optimization iterations to perform.
    show_size: Number of pixels to show for each image. Max 299.
    """

    # Which layer are we using?
    if conv_id is None:
        print("Final fully-connected layer before softmax.")
    else:
        print("Layer:", conv_names[conv_id])

    # Initialize the array of images.
    images = []

    # For each feature do the following. Note that the
    # last fully-connected layer only supports numbers
    # between 1 and 1000, while the convolutional layers
    # support numbers between 0 and some other number.
    # So we just use the numbers between 1 and 7.
    for feature in range(1, 7):
        print("Optimizing image for feature no.", feature)

        # Find the image that maximizes the given feature
        # for the network layer identified by conv_id (or None).
        image = optimize_image(conv_id=conv_id, feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)

        # Squeeze the dim of the array.
        image = image.squeeze()

        # Append to the list of images.
        images.append(image)

    # Convert to numpy-array so we can index all dimensions easily.
    images = np.array(images)

    # Plot the images.
    plot_images(images=images, show_size=show_size)

# 为浅处的卷积层优化图像
# 举个例子，寻找让卷积层conv_names[conv_id]中的2号特征最大化的输入图像，其中conv_id=5。
image = optimize_image(conv_id=5, feature=2,
                       num_iterations=30, show_progress=True)

plot_image(image)

# 为卷积层优化多张图像
# 下面，我们为Inception模型中的卷积层优化多张图像，并绘制它们。
# 这些图像展示了卷积层“想看到的”内容。注意更深的层次里图案变得越来越复杂。
optimize_images(conv_id=0, num_iterations=10)

optimize_images(conv_id=3, num_iterations=30)

optimize_images(conv_id=4, num_iterations=30)

optimize_images(conv_id=5, num_iterations=30)

optimize_images(conv_id=6, num_iterations=30)

optimize_images(conv_id=7, num_iterations=30)

optimize_images(conv_id=8, num_iterations=30)

optimize_images(conv_id=9, num_iterations=30)

optimize_images(conv_id=10, num_iterations=30)

optimize_images(conv_id=20, num_iterations=30)

optimize_images(conv_id=30, num_iterations=30)

optimize_images(conv_id=40, num_iterations=30)

optimize_images(conv_id=50, num_iterations=30)

optimize_images(conv_id=60, num_iterations=30)

optimize_images(conv_id=70, num_iterations=30)

optimize_images(conv_id=80, num_iterations=30)

optimize_images(conv_id=90, num_iterations=30)

optimize_images(conv_id=93, num_iterations=30)

# Softmax前最终的全连接层
'''
现在，我们为Inception模型中的最后一层优化并绘制图像。这是在softmax分类器前的全连接层。该层特征对应了输出的类别。

我们可能希望在这些图像里看到一些可识别的图案，比如对应输出类别的猴子、鸟类等，但图像只显示了一些复杂的、抽象的图案。
'''

optimize_images(conv_id=None, num_iterations=30)

'''
上面只显示了100x100像素的图像，但实际上是299x299像素。如果我们执行更多的优化迭代并画出完整的图像，可能会有一些可识别的模式。那么，让我们再次优化第一张图像，并以全分辨率来绘制。

Inception模型以大约100%的确信度将结果图像分类成“敏狐”，但在人眼看来，图像只是一些抽象的图案。

如果你想测试另一个特征号码，要注意，号码必须介于0到1000之间，因为它对应了最终输出层的一个有效类别号。
'''
image = optimize_image(conv_id=None, feature=1,
                       num_iterations=100, show_progress=True)
plot_image(image=image)
