import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model


'''
注：我们将使用Keras保存和加载模型，因此您需要安装h5py。 您还需要安装scikit-optimization以进行超参数优化。
您应该能够在终端中运行以下命令来安装它们：pip install h5py scikit-optimize

注意：本笔记本需要scikit-optimize中的功能，这些功能在撰写本文时尚未合并到正式版本中。 
如果本笔记本无法运行上述命令安装的scikit-optimize版本，则可能需要通过运行以下命令来从开发分支安装scikit-optimize：

pip install git+git://github.com/Hvass-Labs/scikit-optimize.git@610ce8d3e3e82d76f798ad90984c5888a204884e
'''
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

'''
超参数
在本教程中，我们希望找到使得简单的卷积神经网络在手写数字的MNIST数据集分类中表现最佳的超参数。
对于这个演示，我们希望找到以下超参数：
优化器的学习率。
完全连接/密集层的数量。
每个密集层的节点数量。
是否在所有图层中使用“sigmoid”或“relu”激活。

我们将使用Python包scikit-optimize（或skopt）来查找这些超参数的最佳选择。
在开始实际搜索超参数之前，我们首先需要为每个参数定义有效的搜索范围或搜索维度。
这是学习率的搜索维度。它是一个实数（浮点），下限为1e-6，上限为1e-2。
但是，不是直接在这些边界之间进行搜索，而是使用对数变换，所以我们将搜索1ek中的数字k，它仅在-6和-2之间有界。这比搜索整个指数范围要好。
'''
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')

'''
这是神经网络中稠密层数的搜索维数。 这是一个整数，我们希望神经网络中至少有1个致密层和至多5个致密层。
'''
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')

# 这是每个密集层的节点数量的搜索维度。 这也是一个整数，我们希望在神经网络的每一层至少有5个，最多512个节点。
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')

# 这是激活函数的搜索维度。 这是一个组合或分类参数，可以是'relu'或'sigmoid'。

dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')

# 然后，我们将所有这些搜索维度组合到一个列表中。
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation]

'''
用我们通过手动调整找到的合适选择开始搜索超参数是有帮助的。 
但是，我们将使用以下参数的表现不佳，以便更好地展示超参数优化的有用性：1e-5的学习率，
具有16个节点的单个致密层以及relu激活函数。

请注意，这些超参数打包在一个列表中。 这是skopt如何在超参数内部工作的。 因此，您需要确保尺寸的顺序与上述尺寸中给出的顺序一致。
'''
default_parameters = [1e-5, 1, 16, 'relu']

'''
日志目录名称的辅助函数
我们将记录所有参数组合的训练进度，以便使用TensorBoard查看和比较它们。 
这是通过设置一个通用的父目录来完成的，然后为每个具有适当名称的参数组合使用一个子目录。
'''
def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation)

    return log_dir

# 加载数据
# MNIST数据集约为12 MB，如果它不在给定路径中，它将自动下载
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data/', one_hot=True)


data.test.cls = np.argmax(data.test.labels, axis=1)
validation_data = (data.validation.images, data.validation.labels)

# 数据维度
# 数据维度在以下源代码的几个地方使用。 它们被定义一次，因此我们可以在下面的源代码中使用这些变量而不是数字。
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_size, img_size)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# 函数用于在3x3网格中绘制9个图像，并在每个图像下面写入真实和预测的类。
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

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

# 绘制几张图片，看看数据是否正确

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# 用于绘制已被错误分类的来自测试集的图像示例的功能。
def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.test.cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

'''
超参数优化
执行超参数优化需要几个步骤。
创建模型
我们首先需要一个函数，它接受一组超参数并创建对应于这些参数的卷积神经网络。 
我们使用Keras在TensorFlow中构建神经网络，参见教程＃03-C了解更多细节。
'''


def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Start construction of a Keras Sequential model.
    model = Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=(img_size_flat,)))

    # The input from MNIST is a flattened array with 784 elements,
    # but the convolutional layers expect images with shape (28, 28, 1)
    model.add(Reshape(img_shape_full))

    # First convolutional layer.
    # There are many hyper-parameters in this layer, but we only
    # want to optimize the activation-function in this example.
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation=activation, name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Second convolutional layer.
    # Again, we only want to optimize the activation-function here.
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation=activation, name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i + 1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))

    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)

    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Train and Evaluate the Model
# 具有最佳超参数的神经网络保存到磁盘，以便以后可以重新加载。 这是模型的文件名。
path_best_model = '19_best_model.keras'

# 这是保存到磁盘的模型的分类准确性。 它是一个全局变量，它将在优化超参数期间进行更新。
best_accuracy = 0.0

# 这是使用给定的超参数创建和训练神经网络的函数，然后评估验证集上的性能。
# 该函数然后返回所谓的适应度值（又称目标值），这是验证集上的负分类准确度。
# 这是负面的，因为skopt执行最小化而不是最大化。
# 请注意函数装饰器@use_named_args，它包装适应函数，以便可以将所有参数作为单个列表调用，
# 例如：fitness（x = [1e-4,3,256，'relu']）。 这是内部使用的调用方式skopt。
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()

    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation)

    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)

    # Use Keras to train the model.
    history = model.fit(x=data.train.images,
                        y=data.train.labels,
                        epochs=3,
                        batch_size=128,
                        validation_data=validation_data,
                        callbacks=[callback_log])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)

        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()

    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

'''
测试运行
在我们运行超参数优化之前，让我们首先检查上面提到的各种函数实际上是否工作，当我们传递默认的超参数时。
'''
fitness(x=default_parameters)

'''
运行超参数优化
现在我们已经准备好使用scikit-optimize包中的贝叶斯优化运行实际的超参数优化。 
请注意，它首先使用default_parameters调用fitness（）作为我们通过手动调整找到的起点，
这应该有助于优化器更快地找到更好的超参数。

您可以在这里尝试更多的参数，包括调用fitness（）函数的次数，
我们将其设置为40.但fitness（）评估的代价非常高，因此不应该运行太多次，特别是对于 更大的神经网络和数据集。
您也可以尝试使用所谓的采集功能，它可以确定如何从贝叶斯优化器的内部模型中找到一组新的超参数。 
您也可以尝试使用另一个贝叶斯优化器，例如随机森林。
'''

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

'''
优化进度
超参数优化的进度可以很容易地绘制出来。 找到的最佳适应度值绘制在y轴上，请记住，这是验证集上的否定分类精度。
请注意，在发现实质性改进之前，只有很少的超参数需要尝试。
'''
plot_convergence(search_result)

'''
最佳超参数
贝叶斯优化器找到的最佳超参数被打包为一个列表，因为这是它在内部使用的。
'''

print(search_result.x)
# [0.0023584457378584664, 4, 144, 'relu']
'''
我们可以将这些参数转换为具有搜索空间维度的专有名称的字典。
首先，我们需要对搜索空间对象的引用。
'''
space = search_result.space

# 然后我们可以使用它来创建一个字典，其中超参数具有搜索空间维度的专有名称。 这有点尴尬。
print(space.point_to_dict(search_result.x))
'''
{'activation': 'relu',
 'learning_rate': 0.0023584457378584664,
 'num_dense_layers': 4,
 'num_dense_nodes': 144}
'''
# 这是与这些超参数相关的适应值。 这是一个负数，因为贝叶斯优化器执行最小化，
# 所以我们不得不否定作为最大化问题提出的分类精度。
print(search_result.fun)
# -0.98799999999999999

'''
我们还可以看到贝叶斯优化器尝试的所有超参数及其相关的适应度值（否定分类精度）。 对这些进行排序以便首先显示最高的分类精度。
看起来'relu'激活通常比'sigmoid'更好。 否则，很难看到哪种参数选择良好的模式。 我们真的需要绘制这些结果。
'''
print(sorted(zip(search_result.func_vals, search_result.x_iters)))
'''
[(-0.98799999999999999, [0.00057102338020535671, 1, 246, 'relu']),
 (-0.98799999999999999, [0.0023584457378584664, 4, 144, 'relu']),
 (-0.98699999999999999, [0.0043924439217142824, 3, 311, 'relu']),
 (-0.98680000000000001, [0.00025070302453255417, 2, 435, 'relu']),
 (-0.98640000000000005, [0.0020904801989242469, 5, 436, 'relu']),
 (-0.98560000000000003, [0.00017567744133971055, 4, 453, 'relu']),
 (-0.98560000000000003, [0.00018871091218374878, 3, 441, 'relu']),
 (-0.98560000000000003, [0.0010013922052631494, 3, 496, 'relu']),
 (-0.98519999999999996, [0.006752254693985822, 2, 105, 'relu']),
 (-0.98499999999999999, [0.0001905308801138268, 4, 418, 'relu']),
 (-0.98460000000000003, [0.0073224617473678331, 3, 166, 'relu']),
 (-0.98440000000000005, [0.0020143982003767271, 4, 512, 'relu']),
 (-0.98419999999999996, [0.0014193250864683331, 2, 62, 'relu']),
 (-0.97960000000000003, [0.00023735076383216567, 1, 164, 'relu']),
 (-0.97860000000000003, [0.0026064900033469073, 1, 126, 'sigmoid']),
 (-0.97660000000000002, [0.0037123587226393501, 5, 512, 'relu']),
 (-0.9758, [0.0027230837381696737, 2, 364, 'sigmoid']),
 (-0.97340000000000004, [0.0016597651372777609, 1, 512, 'sigmoid']),
 (-0.97260000000000002, [0.0022460993827137423, 2, 326, 'sigmoid']),
 (-0.96919999999999995, [0.00060563429543890952, 2, 474, 'sigmoid']),
 (-0.96879999999999999, [7.5808558985641429e-05, 1, 241, 'relu']),
 (-0.96179999999999999, [0.0014963322170155162, 5, 285, 'sigmoid']),
 (-0.96120000000000005, [0.00013559943302194881, 2, 29, 'relu']),
 (-0.95699999999999996, [0.00056441093780360571, 5, 13, 'relu']),
 (-0.94679999999999997, [0.00036704404112128516, 4, 338, 'sigmoid']),
 (-0.92679999999999996, [1.3066947342663859e-05, 2, 512, 'relu']),
 (-0.90900000000000003, [0.00023277413216549582, 4, 512, 'sigmoid']),
 (-0.90139999999999998, [0.001544493082361837, 1, 5, 'sigmoid']),
 (-0.85440000000000005, [0.00016937303683800523, 4, 252, 'sigmoid']),
 (-0.85140000000000005, [6.1458838378363633e-06, 2, 333, 'relu']),
 (-0.74039999999999995, [2.4847514577863683e-06, 1, 409, 'relu']),
 (-0.72899999999999998, [1.7068698743151031e-06, 4, 512, 'relu']),
 (-0.61660000000000004, [1e-05, 1, 16, 'relu']),
 (-0.2898, [6.1011365846453456e-05, 2, 209, 'sigmoid']),
 (-0.129, [9.9999999999999995e-07, 2, 5, 'relu']),
 (-0.11260000000000001, [5.4599879082087208e-06, 4, 186, 'sigmoid']),
 (-0.11260000000000001, [3.1218037895598157e-05, 3, 427, 'sigmoid']),
 (-0.11260000000000001, [0.00033099542158994725, 5, 5, 'sigmoid']),
 (-0.11260000000000001, [0.01, 5, 352, 'sigmoid']),
 (-0.11260000000000001, [0.01, 5, 512, 'relu'])]
'''

'''
Plots
在skopt库中有几个绘图功能可用。 例如，我们可以绘制激活参数的直方图，其中显示了超参数优化期间样本的分布。
'''
fig, ax = plot_histogram(result=search_result,
                         dimension_name='activation')

'''
我们还可以制作搜索空间两个维度的估计适应度值的横向图，这里将其视为learning_rate和num_dense_layers。
贝叶斯优化器通过构建搜索空间的替代模型，然后搜索此模型而不是真正的搜索空间来工作，因为它速度更快。
该图显示了贝叶斯优化器建立的最后一个代理模型，其中黄色区域更好，蓝色区域更差。黑点显示优化器对搜索空间进行采样的位置，
红色星号显示找到的最佳参数。
这里应该注意几件事情。首先，这种搜索空间的替代模型可能不准确。它仅由40个调用fitness（）函数的样本构建，
用于通过给定的超参数选择来训练神经网络。模拟的健身景观可能与其真实值显着不同，尤其是在搜索空间很少的样本区域。
其次，由于神经网络的训练过程中的随机噪声，每次超参数优化运行时曲线可能会改变。
第三，该图显示了在搜索空间中的所有其他维度上求平均时，改变这两个参数num_dense_layers和learning_rate的效果，
这也称为部分依赖性图，并且是仅在二维中可视化高维空间的方式。
'''
fig = plot_objective_2D(result=search_result,
                        dimension_name1='learning_rate',
                        dimension_name2='num_dense_layers',
                        levels=50)

'''
我们不能为激活超参数创建景观图，因为它是一个分类变量，
可以是两个字符串relu或sigmoid中的一个。 如何编码取决于贝叶斯优化器，
例如，它是使用高斯过程还是随机森林。 但目前无法使用skopt的内置功能进行绘图。
相反，我们只想使用我们通过名称标识的搜索空间的实数和整数值维度。
'''

dim_names = ['learning_rate', 'num_dense_nodes', 'num_dense_layers']
'''
然后，我们可以制作这些维度的所有组合的矩阵图。
对角线显示单个维度对健身的影响。这是该维度的所谓的部分依赖图。它显示近似适应度值如何随着该维度中的不同值而变化。
对角线下方的图表显示了两个维度的部分相关性。这显示了当我们同时改变两个维度时近似适应度值的变化。
这些局部相关性图只是模拟的适应度函数的近似值 - 而这恰好是适应度（）中真实适应度函数的近似值。这可能有点难以理解。
例如，通过为learning_rate固定一个值，然后为搜索空间中的剩余维度采取大量随机样本，计算部分相关性。
然后对所有这些点的估计适应性进行平均。然后针对learning_rate的其他值重复此过程，以显示它对平均适应度的影响。
对于显示两个维度的部分依赖性图的图进行类似的程序。
'''
fig, ax = plot_objective(result=search_result, dimension_names=dim_names)

'''
我们也可以显示另一种类型的矩阵图。 这里对角线显示了贝叶斯优化期间每个超参数样本分布的直方图。 
对角线下面的图表显示了搜索空间中样本的位置，颜色编码显示了样本被采集的顺序。 对于大量的样本，
您可能会发现样本最终会集中在搜索空间的某个区域。
'''
fig, ax = plot_evaluations(result=search_result, dimension_names=dim_names)

# 评估测试集的最佳模型
# 我们现在可以在测试装置上使用最佳模型。 使用Keras重新加载模型非常简单。
model = load_model(path_best_model)

result = model.evaluate(x=data.test.images,
                        y=data.test.labels)

for name, value in zip(model.metrics_names, result):
    print(name, value)
'''
loss 0.0363312054525
acc 0.9888
'''

# 或者我们可以打印分类精确度。
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
# acc: 98.88%

# 预测新数据
images = data.test.images[0:9]

cls_true = data.test.cls[0:9]
y_pred = model.predict(x=images)

cls_pred = np.argmax(y_pred,axis=1)
plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)
'''
错视图像的例子
我们可以绘制测试集中一些错误分类图像的例子。
首先我们得到测试集中所有图像的预测类别：
'''
y_pred = model.predict(x=data.test.images)

cls_pred = np.argmax(y_pred,axis=1)

plot_example_errors(cls_pred)
