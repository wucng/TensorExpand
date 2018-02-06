# -*- coding: utf8 -*-

'''
改变程序的几个不同地方，看看它如何影响性能：
在集成中使用更多神经网络。(多个神经网络集成，这里只使用同一个神经网络，只是每次权重的初始化不一样)
改变训练集的大小。
改变优化迭代的次数，试着增加或减少。
改变学习率
加上正则化，归一化，batch Normal等
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

print(combined_images.shape)
print(combined_labels.shape)

# (60000, 784)
# (60000, 10)
combined_size = len(combined_images) # 60000

train_size = int(0.8 * combined_size) # 48000
validation_size = combined_size - train_size # 12000

# 帮助函数将合并数组集划分成随机的训练集和验证集。
def random_training_set():
    # Create a randomized index into the full / combined training-set.
    idx = np.random.permutation(combined_size)

    # Split the random index into training- and validation-sets.
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training-set.
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    # Select the images and labels for the new validation-set.
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train, y_train, x_validation, y_validation

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

# 这个函数用来在3x3的栅格中画9张图像，然后在每张图像下面写出真实类别和预测类别
def plot_images(images,  # Images to plot, 2-d array.
                cls_true,  # True class-no for images.
                ensemble_cls_pred=None,  # Ensemble predicted class-no.
                best_cls_pred=None):  # Best-net predicted class-no.

    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for i, ax in enumerate(axes.flat):

        # There may not be enough images for all sub-plots.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

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

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# PrettyTensor实现卷积神经网络
x_pretty = pt.wrap(x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = (x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true))

accuracy = y_pred.evaluate_classifier(y_true)
optimizer_ = tf.train.GradientDescentOptimizer(0.1)  # learning rate
optimizer = pt.apply_optimizer(optimizer_, losses=[loss])

# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
# y_pred_cls = tf.argmax(y_pred, axis=1)
# correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 注意，如果在ensemble中有超过100个的神经网络，你需要根据情况来增加max_to_keep。
saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

session = tf.Session()

def init_variables():
    session.run(tf.global_variables_initializer())

train_batch_size = 64

def random_batch(x_train, y_train):
    # Total number of images in the training-set.
    num_images = len(x_train)

    # Create a random index into the training-set.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = x_train[idx, :]  # Images.
    y_batch = y_train[idx, :]  # Labels.

    # Return the batch.
    return x_batch, y_batch


def optimize(num_iterations, x_train, y_train):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch(x_train, y_train)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        if i % 100 == 0:
            # Calculate the accuracy on the training-batch.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Status-message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# 创建神经网络的集成（ensemble）
# 神经网络ensemble的数量
num_networks = 5
# 每个神经网络优化迭代的次数。
num_iterations = 10000

'''
创建神经网络的ensemble。所有网络都使用上面定义的那个TensorFlow图。
每个网络的TensorFlow权重和变量都用随机值初始化，然后进行优化。
接着将变量保存到磁盘中以便之后重载使用。
'''
if True:
    # For each of the neural networks.
    for i in range(num_networks):
        print("Neural network: {0}".format(i))

        # Create a random training-set. Ignore the validation-set.
        x_train, y_train, _, _ = random_training_set()

        # Initialize the variables of the TensorFlow graph.
        session.run(tf.global_variables_initializer())

        # Optimize the variables using this training-set.
        optimize(num_iterations=num_iterations,
                 x_train=x_train,
                 y_train=y_train)

        # Save the optimized variables to disk.
        saver.save(sess=session, save_path=get_save_path(i))

        # Print newline.
        print()

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

# 计算并且预测分类的帮助函数
def predict_labels(images):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)

    # Now calculate the predicted labels for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images between index i and j.
        feed_dict = {x: images[i:j, :]}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    return pred_labels

# 计算一个布尔值向量，代表图像的预测类型是否正确。
def correct_prediction(images, labels, cls_true):
    # Calculate the predicted labels.
    pred_labels = predict_labels(images=images)

    # Calculate the predicted class-number for each image.
    cls_pred = np.argmax(pred_labels, axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct

# 计算一个布尔数组，代表测试集中图像是否分类正确。
def test_correct():
    return correct_prediction(images = data.test.images,
                              labels = data.test.labels,
                              cls_true = data.test.cls)
# 计算一个布尔数组，代表验证集中图像是否分类正确。
def validation_correct():
    return correct_prediction(images = data.validation.images,
                              labels = data.validation.labels,
                              cls_true = data.validation.cls)

# 计算分类准确率的帮助函数
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    return correct.mean()

# 计算测试集的分类准确率。
def test_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the test-set.
    correct = test_correct()

    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)

# 计算原始验证集上的分类准确率。
def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    correct = validation_correct()

    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)

# 函数用来为ensemble中的所有神经网络计算预测标签。后面会将这些标签合并起来
def ensemble_predictions():
    # Empty list of predicted labels for each of the neural networks.
    pred_labels = []

    # Classification accuracy on the test-set for each network.
    test_accuracies = []

    # Classification accuracy on the validation-set for each network.
    val_accuracies = []

    # For each neural network in the ensemble.
    for i in range(num_networks):
        # Reload the variables into the TensorFlow graph.
        saver.restore(sess=session, save_path=get_save_path(i))

        # Calculate the classification accuracy on the test-set.
        test_acc = test_accuracy()

        # Append the classification accuracy to the list.
        test_accuracies.append(test_acc)

        # Calculate the classification accuracy on the validation-set.
        val_acc = validation_accuracy()

        # Append the classification accuracy to the list.
        val_accuracies.append(val_acc)

        # Print status message.
        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(i, val_acc, test_acc))

        # Calculate the predicted labels for the images in the test-set.
        # This is already calculated in test_accuracy() above but
        # it is re-calculated here to keep the code a bit simpler.
        pred = predict_labels(images=data.test.images)

        # Append the predicted labels to the list.
        pred_labels.append(pred)

    return np.array(pred_labels), \
           np.array(test_accuracies), \
           np.array(val_accuracies)

print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))

# pred_labels.shape # (5, 10000, 10)
# 这里用的方法是取ensemble中所有预测标签的平均
ensemble_pred_labels = np.mean(pred_labels, axis=0)
# ensemble_pred_labels.shape # (10000, 10)

ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
# ensemble_cls_pred.shape # (10000,)

ensemble_correct = (ensemble_cls_pred == data.test.cls)
# 对布尔数组取反，因此我们可以用它来查找误分类的图像
ensemble_incorrect = np.logical_not(ensemble_correct)

# 最佳的神经网络
# 现在我们找出在测试集上表现最佳的单个神经网络
# 首先列出ensemble中所有神经网络在测试集上的分类准确率。
# test_accuracies
# array([ 0.9893,  0.988 ,  0.9893,  0.9889,  0.9892])

# 准确率最高的神经网络索引。
best_net = np.argmax(test_accuracies)
# best_net # 0

# 最佳神经网络在测试集上的分类准确率。
# test_accuracies[best_net] # 0.98929999999999996

# 最佳神经网络的预测标签。
best_net_pred_labels = pred_labels[best_net, :, :]

# 预测的类别数字
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)

# 最佳神经网络在测试集上是否正确分类图像的布尔数组
best_net_correct = (best_net_cls_pred == data.test.cls)

# 图像是否被误分类的布尔数组。
best_net_incorrect = np.logical_not(best_net_correct)

# ensemble与最佳网络的比较
# 测试集中被ensemble正确分类的图像数量。
# np.sum(ensemble_correct) # 9916

# 测试集中被最佳网络正确分类的图像数量
# np.sum(best_net_correct) # 9893

# 布尔数组表示测试集中每张图像是否“被ensemble正确分类且被最佳网络误分类”。
ensemble_better = np.logical_and(best_net_incorrect,
                                 ensemble_correct)

# 测试集上ensemble比最佳网络表现更好的图像数量：
# ensemble_better.sum() # 39

# 布尔数组表示测试集中每张图像是否“被最佳网络正确分类且被ensemble误分类”。
best_net_better = np.logical_and(best_net_correct,
                                 ensemble_incorrect)
# 测试集上最佳网络比ensemble表现更好的图像数量：
# best_net_better.sum() # 16

# 绘制以及打印对比的帮助函数
# 函数用来绘制测试集中的图像，以及它们的真实类别与预测类别。
def plot_images_comparison(idx):
    plot_images(images=data.test.images[idx, :],
                cls_true=data.test.cls[idx],
                ensemble_cls_pred=ensemble_cls_pred[idx],
                best_cls_pred=best_net_cls_pred[idx])

# 打印预测标签的函数。
def print_labels(labels, idx, num=1):
    # Select the relevant labels based on idx.
    labels = labels[idx, :]

    # Select the first num labels.
    labels = labels[0:num, :]

    # Round numbers to 2 decimal points so they are easier to read.
    labels_rounded = np.round(labels, 2)

    # Print the rounded labels.
    print(labels_rounded)

# 打印神经网络ensemble预测标签的函数。
def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)

# 打印单个网络预测标签的函数。
def print_labels_best_net(idx, **kwargs):
    print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)

# 打印ensemble中所有神经网络预测标签的函数。只打印第一张图像的标签。
def print_labels_all_nets(idx):
    for i in range(num_networks):
        print_labels(labels=pred_labels[i, :, :], idx=idx, num=1)

# 样本：ensemble比最佳网络好
# 绘制出那些被集成网络正确分类，且被最佳网络误分类的样本。
plot_images_comparison(idx=ensemble_better)

# ensemble对第一张图像（左上）的预测标签：
print_labels_ensemble(idx=ensemble_better, num=1)
# [[ 0. 0. 0. 0.76 0. 0. 0. 0. 0.23 0. ]]

# 最佳网络对第一张图像的预测标签：
print_labels_best_net(idx=ensemble_better, num=1)
# [[ 0. 0. 0. 0.21 0. 0. 0. 0. 0.79 0. ]]

# ensemble中所有网络对第一张图像的预测标签：
print_labels_all_nets(idx=ensemble_better)
'''
[[ 0. 0. 0. 0.21 0. 0. 0. 0. 0.79 0. ]]
[[ 0. 0. 0. 0.96 0. 0.01 0. 0. 0.03 0. ]]
[[ 0. 0. 0. 0.99 0. 0. 0. 0. 0.01 0. ]]
[[ 0. 0. 0. 0.88 0. 0. 0. 0. 0.12 0. ]]
[[ 0. 0. 0. 0.76 0. 0.01 0. 0. 0.22 0. ]]
'''
# 样本：最佳网络比ensemble好
# 现在绘制那些被ensemble误分类，但被最佳网络正确分类的样本。
plot_images_comparison(idx=best_net_better)

# ensemble对第一张图像（左上）的预测标签：
print_labels_ensemble(idx=best_net_better, num=1)
# [[ 0.5 0. 0. 0. 0. 0.05 0.45 0. 0. 0. ]]

# 最佳网络对第一张图像的预测标签：
print_labels_best_net(idx=best_net_better, num=1)
# [[ 0.3 0. 0. 0. 0. 0.15 0.56 0. 0. 0. ]]

# ensemble中所有网络对第一张图像的预测标签：
print_labels_all_nets(idx=best_net_better)
'''
[[ 0.3 0. 0. 0. 0. 0.15 0.56 0. 0. 0. ]]
[[ 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[[ 0.19 0. 0. 0. 0. 0. 0.81 0. 0. 0. ]]
[[ 0.15 0. 0. 0. 0. 0.12 0.72 0. 0. 0. ]]
[[ 0.85 0. 0. 0. 0. 0. 0.14 0. 0. 0. ]]
'''

# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
session.close()


