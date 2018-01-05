#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

iris=load_iris()
iris_X=iris.data
iris_y=iris.target
X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3) # 30%做测试

print(X_train.shape,y_train.shape) # (105, 4) (105,)

y_train = np_utils.to_categorical(y_train, num_classes=3) # 转成one_hot编码
y_test = np_utils.to_categorical(y_test, num_classes=3)

'''
model = Sequential([
    Dense(32, input_dim=4), # 输出节点32 输入维度 4
    Activation('relu'),
    Dense(3), # 输出10 个节点
    Activation('softmax'),
])
'''
model = Sequential([
    Dense(3, input_dim=4), # 输出节点32 输入维度 4
    Activation('softmax')
])

rmsprop = RMSprop(lr=0.5, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
