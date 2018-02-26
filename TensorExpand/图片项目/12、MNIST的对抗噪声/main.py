# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# We also need PrettyTensor.
import prettytensor as pt

# 载入数据
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
'''
Size of:
- Training-set:	55000
- Test-set:	10000
- Validation-set:	5000
'''
# 测试数据集类别数字的整型值，现在计算它
data.test.cls = np.argmax(data.test.labels, axis=1)

# 数据维度
# 在下面的源码中，有很多地方用到了数据维度。它们只在一个地方定义，
# 因此我们可以在代码中使用这些数字而不是直接写数字。

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

# 用来绘制图像的帮助函数
# 这个函数用来在3x3的栅格中画9张图像，然后在每张图像下面写出真实类别和预测类别。
# 如果提供了噪声，就将其添加到所有图像上。
def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.
        image = images[i].reshape(img_shape)

        # Add the adversarial noise to the image.
        image += noise

        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(image,
                  cmap='binary', interpolation='nearest')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# 绘制几张图像来看看数据是否正确
# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

# TensorFlow图（Graph）
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# 对抗噪声
# 输入图像的像素值在0.0到1.0之间。对抗噪声是在输入图像上添加或删除的数值。
# 对抗噪声的界限设为0.35，则噪声在正负0.35之间。
noise_limit = 0.35
# 对抗噪声的优化器会试图最小化两个损失度量：
# (1)神经网络常规的损失度量，因此我们会找到使得目标类型分类准确率最高的噪声；
# (2)L2-loss度量，它会保持尽可能低的噪声。

# 下面的权重决定了与常规的损失度量相比，L2-loss的重要性。通常接近零的L2权重表现的更好。
noise_l2_weight = 0.02
# 当我们为噪声创建变量时，必须告知TensorFlow它属于哪一个变量集合，这样，后面就能通知两个优化器要更新哪些变量。
# 首先为变量集合定义一个名称。这只是一个字符串。
ADVERSARY_VARIABLES = 'adversary_variables'

'''
接着，创建噪声变量所属集合的列表。如果我们将噪声变量添加到集合tf.GraphKeys.VARIABLES中，
它就会和TensorFlow图中的其他变量一起被初始化，但不会被优化。这里有点混乱。
'''
collections = [tf.GraphKeys.VARIABLES, ADVERSARY_VARIABLES]
'''
现在我们可以为对抗噪声添加新的变量。它会被初始化为零。
它是不可训练的，因此并不会与神经网络中的其他变量一起被优化。这让我们可以创建两个独立的优化程序。
'''
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise', trainable=False,
                      collections=collections)

# 对抗噪声会被限制在我们上面设定的噪声界限内。
# 注意此时并未在计算图表内进行计算，在优化步骤之后执行，详见下文。
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))

# 噪声图像只是输入图像和对抗噪声的总和。
x_noisy_image = x_image + x_noise

# 卷积神经网络
x_pretty = pt.wrap(x_noisy_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

'''
注意，在with代码块中，pt.defaults_scope(activation_fn=tf.nn.relu) 
把 activation_fn=tf.nn.relu当作每个的层参数，因此这些层都用到了 
Rectified Linear Units (ReLU) 。defaults_scope使我们能更方便地修改所有层的参数。
'''
# 正常训练的优化器
# 这是会在常规优化程序里被训练的神经网络的变量列表。
# 注意，'x_noise:0'不在列表里，因此这个程序并不会优化对抗噪声。 因为trainable=False

# [var.name for var in tf.trainable_variables()]
'''
['layer_conv1/weights:0',
'layer_conv1/bias:0',
'layer_conv2/weights:0',
'layer_conv2/bias:0',
'layer_fc1/weights:0',
'layer_fc1/bias:0',
'fully_connected/weights:0',
'fully_connected/bias:0']
'''
# 神经网络中这些变量的优化由Adam-optimizer完成，它用到上面PretyTensor构造的神经网络所返回的损失度量。

# 此时不执行优化，实际上这里根本没有计算，我们只是把优化对象添加到TensorFlow图表中，以便稍后运行。

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# 对抗噪声的优化器
# 获取变量列表，这些是需要在第二个程序里为对抗噪声做优化的变量。
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
# 展示变量名称列表。这里只有一个元素，是我们在上面创建的对抗噪声变量。
# [var.name for var in adversary_variables]
# ['x_noise:0']

# 我们会将常规优化的损失函数与所谓的L2-loss相结合。这将会得到在最佳分类准确率下的最小对抗噪声
# L2-loss由一个通常设置为接近零的权重缩放。
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)

# 将正常的损失函数和对抗噪声的L2-loss相结合。
loss_adversary = loss + l2_loss_noise

# 现在可以为对抗噪声创建优化器。由于优化器并不会更新神经网络的所有变量，
# 我们必须给出一个需要更新的变量的列表，即对抗噪声变量。注意，这里的学习率比上面的常规优化器要大很多。
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables) # 只更新x_noise:0

# 现在我们为神经网络创建了两个优化器，一个用于神经网络的变量，另一个用于对抗噪声的单个变量。

# 性能度量

# 在TensorFlow图表中，我们需要另外一些操作，以便在优化过程中向用户展示进度。

# 首先，计算出神经网络输出y_pred的预测类别号，它是一个包含10个元素的向量。类型号是最大元素的索引。
y_pred_cls = tf.argmax(y_pred, dimension=1)

# 接着创建一个布尔数组，用来表示每张图像的预测类型是否与真实类型相同。
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 运行TensorFlow
session = tf.Session()

session.run(tf.global_variables_initializer())
# 帮助函数将对抗噪声初始化/重置为零。
def init_noise():
    session.run(tf.variables_initializer([x_noise]))

# 调用函数来初始化对抗噪声。
init_noise()

# 用来优化迭代的帮助函数
train_batch_size = 64


def optimize(num_iterations, adversary_target_cls=None):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # If we are searching for the adversarial noise, then
        # use the adversarial target-class instead.
        if adversary_target_cls is not None:
            # The class-labels are One-Hot encoded.

            # Set all the class-labels to zero.
            y_true_batch = np.zeros_like(y_true_batch)

            # Set the element for the adversarial target-class to 1.
            y_true_batch[:, adversary_target_cls] = 1.0 # 重设置目标标签

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # If doing normal optimization of the neural network.
        if adversary_target_cls is None:
            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            session.run(optimizer, feed_dict=feed_dict_train)
        else:
            # Run the adversarial optimizer instead.
            # Note that we have 'faked' the class above to be
            # the adversarial target-class instead of the true class.
            session.run(optimizer_adversary, feed_dict=feed_dict_train)

            # Clip / limit the adversarial noise. This executes
            # another TensorFlow operation. It cannot be executed
            # in the same session.run() as the optimizer, because
            # it may run in parallel so the execution order is not
            # guaranteed. We need the clip to run after the optimizer.
            session.run(x_noise_clip)

        # Print status every 100 iterations.
        if (i % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# 获取及绘制噪声的帮助函数
def get_noise():
    # Run the TensorFlow session to retrieve the contents of
    # the x_noise variable inside the graph.
    noise = session.run(x_noise)

    return np.squeeze(noise)

# 这个函数绘制了对抗噪声，并打印一些统计信息。
def plot_noise():
    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()

    # Print statistics.
    print("Noise:")
    print("- Min:", noise.min())
    print("- Max:", noise.max())
    print("- Std:", noise.std())

    # Plot the noise.
    plt.imshow(noise, interpolation='nearest', cmap='seismic',
               vmin=-1.0, vmax=1.0)

# 用来绘制错误样本的帮助函数
# 函数用来绘制测试集中被误分类的样本。
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                noise=noise)

# 绘制混淆（confusion）矩阵的帮助函数
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

# 展示性能的帮助函数
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


optimize(num_iterations=1000)

print_test_accuracy(show_example_errors=True)

# 寻找对抗噪声
# 在我们开始优化对抗噪声之前，先将它初始化为零。上面已经完成了这一步，但这里再执行一次，以防你用其他目标类型重新运行代码。
init_noise()

# 现在执行对抗噪声的优化。这里使用对抗优化器而不是常规优化器，这说明它只优化对抗噪声变量，同时忽略神经网络中的其他变量。
optimize(num_iterations=1000, adversary_target_cls=3)

'''
现在对抗噪声已经被优化了，可以在一张图像中展示出来。红色像素显示了正噪声值，
蓝色像素显示了负噪声值。这个噪声模式将会被添加到每张输入图像中。正噪声值（红）使像素变暗，负噪声值（蓝）使像素变亮。如下所示。
'''
plot_noise()

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# 所有目标类别的对抗噪声
# 这是帮助函数用于寻找所有目标类别的对抗噪声。函数从类型号0遍历到9，执行上面的优化。然后将结果保存到一个数组中。

def find_all_noise(num_iterations=1000):
    # Adversarial noise for all target-classes.
    all_noise = []

    # For each target-class.
    for i in range(num_classes):
        print("Finding adversarial noise for target-class:", i)

        # Reset the adversarial noise to zero.
        init_noise()

        # Optimize the adversarial noise.
        optimize(num_iterations=num_iterations,
                 adversary_target_cls=i)

        # Get the adversarial noise from inside the TensorFlow graph.
        noise = get_noise()

        # Append the noise to the array.
        all_noise.append(noise)

        # Print newline.
        print()

    return all_noise


all_noise = find_all_noise(num_iterations=300)

# 绘制所有目标类型的对抗噪声
# 这个帮助函数用于在栅格中绘制所有目标类型（0到9）的对抗噪声。
def plot_all_noise(all_noise):
    # Create figure with 10 sub-plots.
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    # For each sub-plot.
    for i, ax in enumerate(axes.flat):
        # Get the adversarial noise for the i'th target-class.
        noise = all_noise[i]

        # Plot the noise.
        ax.imshow(noise,
                  cmap='seismic', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(i)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


plot_all_noise(all_noise)

# 红色像素显示正噪声值，蓝色像素显示负噪声值。

# 对抗噪声的免疫
# 现在试着让神经网络对对抗噪声免疫。我们重新训练神经网络，使其忽略对抗噪声。这个过程可以重复多次。
# 帮助函数创建了对对抗噪声免疫的神经网络
# 这是使神经网络对对抗噪声免疫的帮助函数。首先运行优化来找到对抗噪声。接着执行常规优化使神经网络对该噪声免疫。

def make_immune(target_cls, num_iterations_adversary=500,
                num_iterations_immune=200):
    print("Target-class:", target_cls)
    print("Finding adversarial noise ...")

    # Find the adversarial noise.
    optimize(num_iterations=num_iterations_adversary,
             adversary_target_cls=target_cls)

    # Newline.
    print()

    # Print classification accuracy.
    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)

    # Newline.
    print()

    print("Making the neural network immune to the noise ...")

    # Try and make the neural network immune to this noise.
    # Note that the adversarial noise has not been reset to zero
    # so the x_noise variable still holds the noise.
    # So we are training the neural network to ignore the noise.
    optimize(num_iterations=num_iterations_immune)

    # Newline.
    print()

    # Print classification accuracy.
    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)

# 对目标类型3的噪声免疫
make_immune(target_cls=3)

# 现在试着再次运行它。 现在更难为目标类别3找到对抗噪声。神经网络似乎已经变得对对抗噪声有些免疫。
make_immune(target_cls=3)

# 对所有目标类型的噪声免疫
# 现在，试着使神经网络对所有目标类型的噪声免疫。不幸的是，看起来并不太好。
for i in range(10):
    make_immune(target_cls=i)

    # Print newline.
    print()

# 对所有目标类别免疫（执行两次）
# 现在试着执行两次，使神经网络对所有目标类别的噪声免疫。不幸的是，结果也不太好。

# 使神经网络免受一个对抗目标类型的影响，似乎使得它对另外一个目标类型失去了免疫。
for i in range(10):
    make_immune(target_cls=i)

    # Print newline.
    print()

    make_immune(target_cls=i)

    # Print newline.
    print()

# 绘制对抗噪声
# 现在我们已经对神经网络和对抗网络都进行了很多优化。让我们看看对抗噪声长什么样。
plot_noise()
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


# 干净图像上的性能
# 现在将对抗噪声重置为零，看看神经网络在干净图像上的表现。
init_noise()
# 相比噪声图像，神经网络在干净图像上表现的要更差一点。
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# 关闭TensorFlow会话
session.close()

