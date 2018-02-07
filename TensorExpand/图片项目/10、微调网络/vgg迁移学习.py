# -*- coding: utf8 -*-

'''
在ImageNet上。 请注意，使用TensorFlow时，
为了获得最佳性能，您应该设置
在你的Keras配置中的image_data_format =“channels_last”
在~/.keras/keras.json

https://github.com/fchollet/deep-learning-models/releases/  存放各种训练好的模型

数据说明： 
数据下载：https://download.pytorch.org/tutorial/hymenoptera_data.zip
目录结构：
ls hymenoptera_data  
==>train  val (训练与验证目录)

ls hymenoptera_data/train
==>ants  bees (2种类 每一类别放在一个文件夹下)

ls hymenoptera_data/train/ants
==> *.jpg (存放的图片)
'''

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
from tensorflow.python.keras.applications.vgg16 import decode_predictions ,preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

# 帮助函数用于加入目录和文件名列表
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]
'''
下面使用Keras API创建预先训练的VGG16模型的一个实例。 
这会自动下载所需的文件，如果你还没有。 请注意，Keras与Tutorial＃08相比有多简单。
VGG16模型包含卷积部分和用于分类的完全连接（或密集）部分。 如果include_top = True，
则整个VGG16模型下载大约528 MB。 如果include_top = False，则只下载VGG16模型的卷积部分，仅为57 MB。
我们将尝试使用预先训练的模型来预测新数据集中某些图像的类别，因此我们必须下载完整模型，
但是如果您的网络连接速度较慢，则可以修改以下代码以使用 较小的预先训练的模型，不需要分类层。
'''
# 迁移学习只用到卷积层部分，include_top=False
model = VGG16(include_top=False, weights='imagenet') # 模型下载到 ~/.keras/models
# 手动下载模型 并存储在 ~/.keras/models （目录下）
'''
cd ~
mkdir -p  .keras/models
在 https://github.com/fchollet/deep-learning-models/releases/ 下载
vgg16_weights_tf_dim_ordering_tf_kernels.h5
'''

# Keras API有自己的创建输入流水线的方法来使用文件来训练模型。
# 首先，我们需要知道由预先训练的VGG16模型输入的张量的形状。 在这种情况下，它是形状224 x 224 x 3的图像。

input_shape = model.layers[0].output_shape[1:3]

# 文件路径
train_dir = '../../hymenoptera_data/train'
test_dir = '../../hymenoptera_data/val'


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
# print(num_classes)

# 数据集相当不平衡,在这里，我们使用scikit-learn来计算将适当平衡数据集的权重。
# 在训练期间将这些权重应用于批次中的每个图像的梯度，以便缩放对批次的总体梯度的影响
from sklearn.utils.class_weight import compute_class_weight
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)
# print(class_weight)

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

# 我们将这个层称为传输层，因为它的输出将被重新路由到我们的新的完全连接的神经网络，这将对Knifey-Spoony数据集进行分类
'''
使用Keras API创建新模型非常简单。 首先，我们将VGG16模型的一部分从输入层迁移到传输层的输出。 
我们可以称之为卷积模型，因为它由VGG16模型的所有卷积层组成。
'''
conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output) # 将VGG16 input_1到block5_pool这部分的网络提取出来

# 然后，我们可以使用Keras在此之上建立一个新的模型。
new_model = Sequential()
new_model.add(conv_model) # [None, 7, 7, 512]
new_model.add(Flatten()) # [n,7*7*512]
new_model.add(Dense(1024, activation='relu')) # [n,1024]
new_model.add(Dropout(0.5))
new_model.add(Dense(num_classes, activation='softmax')) # [n,num_classes]

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

# 在迁移学习中，我们最初只想重复使用预先训练的VGG16模型，所以我们将禁用所有层的训练(不更新)。
conv_model.trainable = False
for layer in conv_model.layers:
    layer.trainable = False

# 一旦我们改变了模型的图层是否可以训练，我们需要编译模型以使更改生效。
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

epochs = 20
steps_per_epoch = 100

# 训练新模型只是Keras API中的一个函数调用。 GTX 1070 GPU需要大约6-7分钟的时间。
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

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


'''
Keras在每个epoch结束时记录性能指标，以便稍后进行绘制。 
这表明训练集的损失值在训练期间通常减少，但是测试集的损失值更不稳定。 
类似地，训练集的分类准确性通常有所提高，而测试集的分类准确度则更为不稳定。
'''
plot_training_history(history)

# 在训练之后，我们还可以使用Keras API中的单个函数调用来评估新模型在测试集上的性能。
result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))

