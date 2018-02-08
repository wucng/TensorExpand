# -*- coding: utf8 -*-

'''
https://github.com/fchollet/deep-learning-models/releases/  存放各种训练好的模型
'''

import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

# These are the imports from the Keras API.
# Note the long format which can hopefully be shortened in the future to e.g.
# from tf.keras.models import Model.
# from tensorflow.python.keras.models import Model, Sequential
# from tensorflow.python.keras.layers import Dense, Flatten, Dropout

# from tensorflow.python.keras.applications import InceptionV3#VGG16 #,VGG19 ,ResNet50 ,InceptionV3
# from tensorflow.python.keras.applications.inception_v3 import decode_predictions

# from tensorflow.python.keras.applications.vgg16 import decode_predictions # preprocess_input,

from tensorflow.python.keras._impl.keras.applications.inception_v3 import decode_predictions
from tensorflow.python.keras._impl.keras.applications.inception_v3 import InceptionV3

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.optimizers import Adam, RMSprop


'''
下面使用Keras API创建预先训练的VGG16模型的一个实例。 
这会自动下载所需的文件，如果你还没有。 请注意，Keras与Tutorial＃08相比有多简单。
VGG16模型包含卷积部分和用于分类的完全连接（或密集）部分。 如果include_top = True，
则整个VGG16模型下载大约528 MB。 如果include_top = False，则只下载VGG16模型的卷积部分，仅为57 MB。
我们将尝试使用预先训练的模型来预测新数据集中某些图像的类别，因此我们必须下载完整模型，
但是如果您的网络连接速度较慢，则可以修改以下代码以使用 较小的预先训练的模型，不需要分类层。
'''
model = InceptionV3(include_top=True, weights='imagenet') # 模型下载到 ~/.keras/models
# 手动下载模型 并存储在 ~/.keras/models （目录下）
'''
cd ~
mkdir -p  .keras/models
在 https://github.com/fchollet/deep-learning-models/releases/ 下载
inception_v3_weights_tf_dim_ordering_tf_kernels.h5

下载 https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json 复制到 ~/.keras/models

'''

# Keras API有自己的创建输入流水线的方法来使用文件来训练模型。
# 首先，我们需要知道由预先训练的VGG16模型输入的张量的形状。 在这种情况下，它是形状224 x 224 x 3的图像。

# input_shape = model.layers[0].output_shape[1:3]
input_shape=(299,299) # InceptionV3 输入shape [299,299,3]

# 这里我们将展示一些使用预先训练的VGG16模型进行预测的例子。
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

predict(image_path='images/willy_wonka_old.jpg')
