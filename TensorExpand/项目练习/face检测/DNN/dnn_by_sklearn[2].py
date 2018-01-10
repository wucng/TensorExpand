#! /usr/bin/python
# -*- coding: utf8 -*-

'''model
LinearRegression
KNeighborsClassifier
SVC
DecisionTreeClassifier
MLPClassifier # DNN
'''

from sklearn import datasets,preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

import pandas as pd
# import tensorflow as tf
import numpy as np
import glob

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 64
display_step = 10
EPOCH=3

# Network Parameters
img_h=19
img_w=19
img_c=4
n_input = img_h*img_w*img_c
n_classes = 2 #
dropout = 0.75 #


# 加载数据
filepaths=glob.glob('./data_*.pkl')
for i,filepath in enumerate(filepaths):
    if i==0:
        data=pd.read_pickle(filepath)
    else:
        data=np.vstack((data,pd.read_pickle(filepath)))
np.random.shuffle(data)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# train_loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
# xx=np.reshape(data[:,:-1],[-1,img_h,img_w,img_c]).transpose([0,3,1,2]).astype(np.float32) # [-1,4,19,19]

X=data[:,:-1].astype(np.float32) # [-1,19*19*4]
y=data[:,-1].astype(np.uint8) # uint8、int16、int32、int64 , 且是 非one_hot 标签

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

print(X_train.shape,y_train.shape)

# model = LinearRegression(normalize=True)
# model=KNeighborsClassifier(n_neighbors=9)
# model=SVC()
# model=DecisionTreeClassifier()
model=MLPClassifier(256,activation='relu')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
