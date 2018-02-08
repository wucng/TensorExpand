# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

# These are the imports from the Keras API.
# Note the long format which can hopefully be shortened in the future to e.g.
# from tf.keras.models import Model.
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16 #,VGG19 ,ResNet50 ,InceptionV3
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

# 帮助函数用于加入目录和文件名列表
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

# 辅助功能绘制图像
# 函数用于在3x3网格中绘制最多9个图像，并在每个图像下面写入真实和预测的类。
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# 辅助函数用于打印混淆矩阵
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    print("Confusion matrix:")

    # Print the confusion matrix as text.
    print(cm)

    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))

# 帮助功能绘制示例错误
# 用于绘制已被错误分类的来自测试集的图像示例的功能。
def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_test)

    # Get the file-paths for images that were incorrectly classified.
    image_paths = np.array(image_paths_test)[incorrect]

    # Load the first 9 images.
    images = load_images(image_paths=image_paths[0:9])

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the 9 images we have loaded and their corresponding classes.
    # We have only loaded 9 images so there is no need to slice those again.
    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# 函数用于计算整个测试集的预测类别，并调用上述函数来绘制一些错误分类图像的示例。
def example_errors():
    # The Keras data-generator for the test-set must be reset
    # before processing. This is because the generator will loop
    # infinitely and keep an internal index into the dataset.
    # So it might start in the middle of the test-set if we do
    # not reset it first. This makes it impossible to match the
    # predicted classes with the input images.
    # If we reset the generator, then it always starts at the
    # beginning so we know exactly which input-images were used.
    generator_test.reset()

    # Predict the classes for all images in the test-set.
    y_pred = new_model.predict_generator(generator_test,
                                         steps=steps_test)

    # Convert the predicted classes from arrays to integers.
    cls_pred = np.argmax(y_pred, axis=1)

    # Plot examples of mis-classified images.
    plot_example_errors(cls_pred)

    # Print the confusion matrix.
    print_confusion_matrix(cls_pred)

# 辅助函数用于加载图像
# 数据集没有加载到内存中，而是具有训练集中图像的文件列表和测试集中图像的另一个文件列表。 这个辅助函数加载一些图像文件。
def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

# 帮助功能绘制训练历史
# 这绘制了在使用Keras API进行培训期间记录的分类准确度和损失值。
def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()

# Dataset: Knifey-Spoony
# Knifey-Spoony数据集在Tutorial＃09中介绍。 它是从视频文件生成的，通过采取单独的帧并将其转换为图像。
import knifey
knifey.maybe_download_and_extract()
knifey.copy_files()
train_dir = knifey.train_dir
test_dir = knifey.test_dir

# 预训练模型：VGG16
'''
下面使用Keras API创建预先训练的VGG16模型的一个实例。 
这会自动下载所需的文件，如果你还没有。 请注意，Keras与Tutorial＃08相比有多简单。
VGG16模型包含卷积部分和用于分类的完全连接（或密集）部分。 如果include_top = True，
则整个VGG16模型下载大约528 MB。 如果include_top = False，则只下载VGG16模型的卷积部分，仅为57 MB。
我们将尝试使用预先训练的模型来预测新数据集中某些图像的类别，因此我们必须下载完整模型，
但是如果您的网络连接速度较慢，则可以修改以下代码以使用 较小的预先训练的模型，不需要分类层。
'''
model = VGG16(include_top=True, weights='imagenet')

# 输入管道
# Keras API有自己的创建输入流水线的方法来使用文件来训练模型。
# 首先，我们需要知道由预先训练的VGG16模型输入的张量的形状。 在这种情况下，它是形状224 x 224 x 3的图像。

input_shape = model.layers[0].output_shape[1:3]
# input_shape # (224, 224)
'''
Keras使用所谓的数据发生器来将数据输入到神经网络中，这将使数据在永久的环境中循环。
我们有一个小的训练集，所以通过对图像进行各种转换来人为地增大其大小是有帮助的。 
我们使用一个内置的数据生成器，可以进行这些随机转换。 这也被称为增强数据集。
'''
datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')


'''
我们还需要一个用于测试集的数据生成器，但是这不应该对图像进行任何转换，
因为我们想要知道这些特定图像的精确分类准确性。 所以我们只是重新调整像素值，
使它们在0.0到1.0之间，因为这是VGG16模型所期望的。
'''
datagen_test = ImageDataGenerator(rescale=1./255)

# 数据生成器将返回批量的图像。 由于VGG16型号太大，批量不能太大，否则GPU内存不足。
batch_size = 20
'''
我们可以在训练过程中保存随机变换的图像，以检查它们是否过度失真，所以我们必须调整上面的数据生成器的参数。
'''
if True:
    save_to_dir = None
else:
    save_to_dir='augmented_images/'

'''
现在我们创建实际的数据生成器，它将从磁盘读取文件，调整图像的大小并返回一个随机批次。
将数据生成器的构建分成这两个步骤是有些尴尬的，但这可能是因为有不同类型的数据生成器
可用于不同的数据类型（图像，文本等）和源（内存 或磁盘）。
'''
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

'''
找到了属于3个类的4170个图像。
测试集的数据生成器不应该对图像进行转换和混洗。
'''
generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

'''
发现属于3个类的530个图像。
由于数据生成器将永久循环，因此我们需要指定在测试集评估和预测期间要执行的步骤数。 
由于我们的测试集包含530个图像，并且批处理大小设置为20，因此对测试集进行一次完整处理的步数为26.5。 
这就是为什么我们需要在上面的example_errors（）函数中重置数据生成器的计数器，所以它总是从测试集的开始处开始处理。
这是Keras API的另一个稍微尴尬的方面，或许可以改进。
'''
steps_test = generator_test.n / batch_size
# steps_test # 26.5

# 获取训练集和测试集中所有图像的文件路径。
image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

# 获取训练集和测试集中所有图像的类别编号
cls_train = generator_train.classes
cls_test = generator_test.classes

# 获取数据集的类名称。
class_names = list(generator_train.class_indices.keys())
# class_names # ['forky', 'knifey', 'spoony']

# 获取数据集的类别总数。
num_classes = generator_train.num_class
# num_classes # 3

# 绘制一些图片，看看数据是否正确
# Load the first images from the train-set.
images = load_images(image_paths=image_paths_train[0:9])

# Get the true classes for those images.
cls_true = cls_train[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)

'''
Knifey-Spoony 数据集相当不平衡，因为它的叉子图像少，刀片图像多，还有更多的匙子图像。 
这可能会导致在训练过程中出现问题，因为神经网络将显示更多勺子比叉子的例子，所以它可能会更好地识别汤匙。
在这里，我们使用scikit-learn来计算将适当平衡数据集的权重。 
在训练期间将这些权重应用于批次中的每个图像的梯度，以便缩放对批次的总体梯度的影响。
'''
from sklearn.utils.class_weight import compute_class_weight
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

# 请注意，forky-class 权重大约是1.398，而 spoony-class的权重只有0.707。
# 这是因为这些图像的图像较少，所以应该放大这些图像forky-class 的梯度，而对于这些图像spoony-images则应该降低梯度。
# class_weight
# array([ 1.39839034,  1.14876033,  0.70701933])
# class_names
# ['forky', 'knifey', 'spoony']

# 示例预测
# 这里我们将展示一些使用预先训练的VGG16模型进行预测的例子。
# 我们需要一个辅助函数来加载和调整图像大小，以便将其输入到VGG16模型中，并进行实际预测并显示结果。
def predict(image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    plt.imshow(img_resized)
    plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)

    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred)[0]

    # Print the predictions.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))

# 然后，我们可以在被归类为金刚鹦鹉（鹦鹉物种）的鹦鹉的照片上使用VGG16模型，得分相当高，为79％。
predict(image_path='images/parrot_cropped1.jpg')

# 然后，我们可以使用VGG16模型来预测新训练集中其中一个图像的类别。 VGG16模型对这个图像非常困惑，不能做出好的分类。
predict(image_path=image_paths_train[0])
# 我们可以在我们的新训练集中尝试另一个图像，而VGG16模型仍然困惑。
predict(image_path=image_paths_train[1])

# 我们也可以从我们的新测试集中尝试一个图像，而VGG16模型又是非常混乱的。
predict(image_path=image_paths_test[0])

# 迁移学习
'''
预先训练的VGG16模型无法分类来自Knifey-Spoony数据集的图像。 
原因可能是VGG16模型是在所谓的ImageNet数据集上进行训练的，可能没有包含许多餐具图像。
卷积神经网络的较低层可以识别图像中的许多不同形状或特征。 它是将这些特征组合成整个图像分类的最后几个完全连通的图层。
因此，我们可以尝试将VGG16模型的最后一个卷积层的输出重新路由到一个新的完全连接的神经网络，我们创建这个网络是为了在Knifey-Spoony数据集上进行分类。

首先我们打印VGG16模型的摘要，以便我们可以看到它的图层的名称和类型，以及层之间流动张量的形状。 
这是我们在本教程中使用VGG16模型的主要原因之一，因为Inception v3模型有很多图层，打印出来会令人困惑。
'''
# model.summary()
'''
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
'''

# 我们可以看到最后一个卷积层被称为“block5_pool”，所以我们使用Keras来获得对该层的引用。
transfer_layer = model.get_layer('block5_pool')
'''
我们将这个层称为传输层，因为它的输出将被重新路由到我们的新的完全连接的神经网络，这将对Knifey-Spoony数据集进行分类。
转印层的输出具有以下形状：
'''
# transfer_layer.output
# <tf.Tensor 'block5_pool/MaxPool:0' shape=(?, 7, 7, 512) dtype=float32>

'''
使用Keras API创建新模型非常简单。 首先，我们将VGG16模型的一部分从输入层迁移到传输层的输出。 
我们可以称之为卷积模型，因为它由VGG16模型的所有卷积层组成。
'''
conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)

# 然后，我们可以使用Keras在此之上建立一个新的模型。
# Start a new Keras Sequential model.
new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add the final layer for the actual classification.
new_model.add(Dense(num_classes, activation='softmax'))

'''
我们使用Adam优化器的学习率相当低。 学习率也许可能更大。 
但是如果你尝试和训练更多层的原始VGG16模型，那么学习率应该是相当低的，
否则VGG16模型的预训练权重将会失真，并且将无法学习。
'''
optimizer = Adam(lr=1e-5)

# 我们在Knifey-Spoony数据集中有3个类，所以Keras需要使用这个loss函数。
loss = 'categorical_crossentropy'

# 我们感兴趣的唯一性能指标是分类精度。
metrics = ['categorical_accuracy']

# 辅助功能用于打印VGG16模型中的图层是否应该被训练。
def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

# 默认情况下，VGG16模型的所有层都是可训练的。
print_layer_trainable()
'''
True:	input_1
True:	block1_conv1
True:	block1_conv2
True:	block1_pool
True:	block2_conv1
True:	block2_conv2
True:	block2_pool
True:	block3_conv1
True:	block3_conv2
True:	block3_conv3
True:	block3_pool
True:	block4_conv1
True:	block4_conv2
True:	block4_conv3
True:	block4_pool
True:	block5_conv1
True:	block5_conv2
True:	block5_conv3
True:	block5_pool
'''

# 在迁移学习中，我们最初只想重复使用预先训练的VGG16模型，所以我们将禁用所有层的训练。
conv_model.trainable = False
for layer in conv_model.layers:
    layer.trainable = False

print_layer_trainable()
'''
False:	input_1
False:	block1_conv1
False:	block1_conv2
False:	block1_pool
False:	block2_conv1
False:	block2_conv2
False:	block2_pool
False:	block3_conv1
False:	block3_conv2
False:	block3_conv3
False:	block3_pool
False:	block4_conv1
False:	block4_conv2
False:	block4_conv3
False:	block4_pool
False:	block5_conv1
False:	block5_conv2
False:	block5_conv3
False:	block5_pool
'''
# 一旦我们改变了模型的图层是否可以训练，我们需要编译模型以使更改生效。
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

'''
一个epoch通常意味着一个完整的训练集处理。 但是我们上面创建的数据生成器，将会产生批量的永久性训练数据。 
所以我们需要定义每个epoch 我们要运行的步数，这个数乘以上面定义的批量大小。 
在这种情况下，我们每个epoch  有100步，批量为20，所以epoch由训练集的2000个随机图像组成。 我们跑20个这样的epochs。
选择这些特定数字的原因是因为它们似乎足以用这个特定的模型和数据集进行训练，并且没有花费太多时间，并导致20个数据点（每个“epoch”一个） 可以事后绘制。
'''
epochs = 20
steps_per_epoch = 100
# 训练新模型只是Keras API中的一个函数调用。 GTX 1070 GPU需要大约6-7分钟的时间。
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

'''
Keras在每个epoch结束时记录性能指标，以便稍后进行绘制。 
这表明训练集的损失值在训练期间通常减少，但是测试集的损失值更不稳定。 
类似地，训练集的分类准确性通常有所提高，而测试集的分类准确度则更为不稳定。
'''
plot_training_history(history)

# 在训练之后，我们还可以使用Keras API中的单个函数调用来评估新模型在测试集上的性能。
result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))

'''
我们可以从测试集中绘制一些错误分类图像的例子。 这些图像中的一些对于人类来说也是难以分类的。
混淆矩阵表明，新模型特别是在分类问题上存在问题。
'''
# example_errors()

# 微调
'''
在转移学习中，在训练新分类器期间原始的预训练模型被锁定或冻结。 
这确保了原始VGG16型号的重量不会改变。 这样做的一个好处是，
新分类器的训练不会通过VGG16模型传播大的梯度，这可能使其权重变形或导致过度拟合到新的数据集。
但是一旦新的分类器被训练完毕，我们可以尝试轻轻微调VGG16模型中的一些更深的层次。 我们称之为微调。
Keras在原始VGG16模型的每一层中是否使用可训练的布尔值，或者是否被我们称之为conv_layer的“元层”中的
可训练布尔值所覆盖，这有点不清楚。 因此，我们将启用conv_layer和原始VGG16模型中的所有相关图层的可训练布尔值。
'''

conv_model.trainable = True
# 我们想要训练最后两个卷积层，其名称包含“block5”或“block4”。
for layer in conv_model.layers:
    # Boolean whether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)

    # Set the layer's bool.
    layer.trainable = trainable

# 我们可以检查这是否更新了相关图层的可训练布尔值。
print_layer_trainable()
'''
False:	input_1
False:	block1_conv1
False:	block1_conv2
False:	block1_pool
False:	block2_conv1
False:	block2_conv2
False:	block2_pool
False:	block3_conv1
False:	block3_conv2
False:	block3_conv3
False:	block3_pool
True:	block4_conv1
True:	block4_conv2
True:	block4_conv3
True:	block4_pool
True:	block5_conv1
True:	block5_conv2
True:	block5_conv3
True:	block5_pool
'''

# 我们将使用较低的学习率进行微调，因此原始VGG16模型的权重只会变得缓慢。
optimizer_fine = Adam(lr=1e-7)
'''
因为我们已经定义了一个新的优化器，并且已经改变了模型中许多层次的可训练布尔值，
所以我们需要重新编译模型，以便在继续训练之前变更生效。
'''
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

# 然后可以继续进行训练，以便与新的分类器一起对VGG16模型进行微调。
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

'''
然后，我们可以绘制训练中的损失值和分类精度。 根据数据集，原始模型，
新分类器以及诸如学习速率的超参数，这可以提高训练集和测试集上的分类精度，
或者可以改善训练集但是恶化 它适合在过度配合的情况下进行测试。 这可能需要一些实验参数才能正确使用。
'''

plot_training_history(history)

result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))

'''
我们可以再次绘制一些错误分类的图像的例子，从混淆矩阵中我们也可以看出，这个模型仍然存在分类正确的问题。
部分原因可能是训练集只包含994个叉子图像，而其中包含1210个刀片图像和1966个勺子图像。 
尽管我们已经对类别进行加权以弥补这种不平衡，并且我们还通过在训练期间通过以不同的方式随机地转换图像来增加训练集，
但是模型学习识别分叉可能是不够的。
'''
example_errors()

# 结论
'''
本教程展示了如何使用Keras API for TensorFlow在新数据集上对预先训练的VGG16模型进行迁移学习和微调。 
使用Keras API而非直接在TensorFlow中实现这一点要容易得多。
微调是否提高了分类精度，而不仅仅是使用迁移学习取决于预先训练的模型，
选择的转换层，数据集以及如何训练新模型。 您可能会从微调中体验到改进的性能，或者如果精调模型过度训练数据，则性能可能会变差。
'''

# 演习
'''
这些是一些锻炼的建议，可能有助于提高您的技能TensorFlow。 获得TensorFlow的实践经验对于学习如何正确使用它非常重要。
在进行任何更改之前，您可能需要备份此Notebook和其他文件。

尝试使用VGG16模型中的其他图层作为传输层。它如何影响培训和分类的准确性？
更改我们添加的新分类图层。您可以通过增加还是减少完全连接/密集层中的节点数来提高分类精度？
如果您删除新分类器中的Dropout层，会发生什么情况？
改变迁移学习和微调的学习率。
尝试在整个VGG16模型上进行微调，而不仅仅是最后几个图层。它如何影响训练和测试集的分类准确性？为什么？
尝试从一开始就进行微调，以便从零开始训练新的分类层以及VGG16模型的所有卷积层。您可能需要降低优化器的学习率。
从测试集中添加一些图像到训练集。这会提高性能吗？
尝试从训练集中删除一些差的图像，这样所有的图像都具有相同数量的图像。这是否改善了混淆矩阵中的数字？
使用另一个数据集。
使用Keras提供的另一个预先训练的模型。
向朋友解释程序是如何工作的。
'''
