# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist


# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
'''
我们将使用由IMDB电影评论组成的50000个数据集。 
Keras有一个内置的功能来下载一个类似的数据集（但显然是一半的大小）。 
然而，Keras的版本已经将数据集中的文本转换为整数标记，这是使用自然语言工作的关键部分，
本教程中还将演示这些语言，因此我们下载了实际的文本数据。
注意：数据集为84 MB，将自动下载。
'''
import imdb
# 如果要将文件保存在其他目录中，请更改此项。
# imdb.data_dir = "data/IMDB/"

# 自动下载并提取文件。
imdb.maybe_download_and_extract()
x_train_text, y_train = imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)

# 将下面的一些用途组合成一个数据集。
data_text = x_train_text + x_test_text
# 从训练集打印一个例子，看看数据看起来是否正确。
print(x_train_text[1])




