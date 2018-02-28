# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 加载数据
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 数据维度
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


data.train.cls = np.argmax(data.train.labels, axis=1) # 转成非one_hot编码
data.test.cls = np.argmax(data.test.labels, axis=1)

# 估计器的输入函数
'''
我们必须提供返回数据的函数，而不是直接向Estimator提供原始数据。 这使数据源具有更大的灵活性，以及数据如何随机混洗和迭代。
请注意，我们将使用DNNClassifier创建一个Estimator，它假定类数是整数，所以我们使用data.train.cls而不是data.train.labels，它们是一个热门编码数组。
该函数还具有用于batch_size，queue_capacity和num_threads的参数，以更好地控制数据读取。 在我们的例子中，我们直接从内存中的numpy数组中获取数据，所以不需要。'''

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.train.images)},
    y=np.array(data.train.cls),
    num_epochs=None,
    shuffle=True)

# 这实际上返回一个函数：
# train_input_fn
# <function tensorflow.python.estimator.inputs.numpy_io.numpy_input_fn.<locals>.input_fn>


# 调用这个函数返回一个带有TensorFlow操作的元组，用于返回输入和输出数据：
# print(train_input_fn())
'''
({'x': <tf.Tensor 'random_shuffle_queue_DequeueMany:1' shape=(128, 784) dtype=float32>},
 <tf.Tensor 'random_shuffle_queue_DequeueMany:2' shape=(128,) dtype=int64>)
'''
'''
同样，我们需要创建一个读取测试集数据的函数。 请注意，我们只想处理这些图像一次，因此num_epochs = 1，我们不希望图像混乱，因此shuffle = False。
'''
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.test.images)},
    y=np.array(data.test.cls),
    num_epochs=1,
    shuffle=False)

# 预测新数据的类别也需要输入函数。 作为一个例子，我们只是使用测试集中的一些图像。
some_images = data.test.images[0:9]
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": some_images},
    num_epochs=1,
    shuffle=False)

'''
类别号码实际上不用于输入函数，因为它不需要进行预测。 但是，当我们在下面绘制图像时，需要真正的类别编号。
'''
some_images_cls = data.test.cls[0:9]

# 使用预先制作的估算器时，我们需要指定数据的输入要素。 在这种情况下，我们希望从我们的数据集中输入图像，这些图像是给定形状的数值数组。

feature_x = tf.feature_column.numeric_column("x", shape=img_shape)
# 您可以有几个输入功能，然后将它们合并到一个列表中：
feature_columns = [feature_x]
# 在这个例子中，我们想要分别使用512,256和128个单元的3层DNN。
num_hidden_units = [512, 256, 128]
'''
DNNClassifier然后为我们构建神经网络。 我们也可以指定激活函数和其他各种参数（请参阅文档）。 
这里我们只是指定类的数量和检查点将被保存的目录。
'''
'''
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=num_hidden_units,
                                   activation_fn=tf.nn.relu,
                                   n_classes=num_classes,
                                   model_dir="./checkpoints_tutorial17-1/")
'''
# -------自定义DNNClassifier-----------------
def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn,
    #           see e.g. train_input_fn for these two.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.

    # Reference to the tensor named "x" in the input-function.
    x = features["x"]

    net=tf.layers.dense(x,512,activation=tf.nn.relu,name='fc1')
    net=tf.layers.dense(net,256,activation=tf.nn.relu,name='fc2')

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc3',
                          units=128, activation=tf.nn.relu)

    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc4',
                          units=10)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.

        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec

# 创建估算器的实例
# 我们可以指定超参数，例如 用于优化器的学习率。
params = {"learning_rate": 1e-4}

# 然后，我们可以创建新的估算器的实例。
# 请注意，我们不提供功能列，因为它是在调用model_fn（）时从数据函数自动推断的。
# 从TensorFlow文档中不清楚为什么有必要在上面的示例中使用DNNClassifier时指定要素列，在此不需要时。

model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./checkpoints_tutorial17-2/")

# ------------------------



'''
model=tf.estimator.DNNLinearCombinedClassifier(model_dir="./checkpoints_tutorial17-1/",
                                               linear_feature_columns=feature_columns,
                                               dnn_feature_columns=feature_columns,
                                               dnn_activation_fn=tf.nn.relu,
                                               dnn_hidden_units=num_hidden_units,
                                               n_classes=num_classes)
'''

# ------自定义优化器---------------------#

'''
如果你不能使用内置的Estimators之一，那么你可以自己创建一个任意的TensorFlow模型。 为此，您首先需要创建一个定义以下内容的函数：
TensorFlow模型，例如 卷积神经网络。
模型的输出。
优化过程中用于改善模型的损失函数。
优化方法。
性能指标。
估算器可以以三种模式运行：训练，评估或预测。 代码基本相同，但在预测模式下，我们不需要设置丢失函数和优化器。
这是Estimator API的另一个方面，设计不佳，与我们过去使用结构进行ANSI C编程的方式类似。 将它分解为几个函数并对Estimator类进行细分可能会更加优雅。
'''
'''
def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn,
    #           see e.g. train_input_fn for these two.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.

    # Reference to the tensor named "x" in the input-function.
    x = features["x"]

    # The convolutional layers expect 4-rank tensors
    # but x is a 2-rank tensor, so reshape it.
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=16, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=36, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Flatten to a 2-rank tensor.
    net = tf.contrib.layers.flatten(net)
    # Eventually this should be replaced with:
    # net = tf.layers.flatten(net)

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=10)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.

        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec

# 创建估算器的实例
# 我们可以指定超参数，例如 用于优化器的学习率。
params = {"learning_rate": 1e-4}

# 然后，我们可以创建新的估算器的实例。
# 请注意，我们不提供功能列，因为它是在调用model_fn（）时从数据函数自动推断的。
# 从TensorFlow文档中不清楚为什么有必要在上面的示例中使用DNNClassifier时指定要素列，在此不需要时。

model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./checkpoints_tutorial17-2/")
'''
# ----------------------------#


# Training
# 现在我们可以训练模型进行给定次数的迭代。 这会自动加载并保存检查点，以便稍后继续培训。
# 请注意，文本INFO：tensorflow：每行都会打印出来，并且很难快速读取实际进度。 它应该已经打印在一行代替。
model.train(input_fn=train_input_fn, steps=2000)

# Evaluation
# 一旦模型被训练完毕，我们可以评估它在测试集上的表现。
result = model.evaluate(input_fn=test_input_fn)
print(result)
# {'global_step': 2000, 'loss': 11.041612, 'accuracy': 0.97390002, 'average_loss': 0.087228738}

print(result['accuracy'])
# 0.97390002

# Predictions
# 训练好的模型也可以用来预测新数据。
# 请注意，每次我们对新数据进行预测时，都会重新创建TensorFlow图并重新加载检查点。 如果模型非常大，那么这可能会增加很大的开销。
# 目前还不清楚为什么Estimator是以这种方式设计的，可能是因为它总是使用最新的检查点，并且它也可以轻松分发以供多台计算机使用。
predictions = model.predict(input_fn=predict_input_fn)
'''
cls = [p['classes'] for p in predictions]
cls_pred = np.array(cls, dtype='int').squeeze()
'''
cls_pred = np.array(list(predictions))

print('pred',cls_pred,'|','real',some_images_cls)


