#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
使用sklearn进行数据预处理
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA
from sklearn.utils import shuffle
import pandas
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

# 数据预处理
class Data_processing():

    def __init__(self,data_path,save_path=None):
        '''
        :param data_path: 数据路径
        :param save_path: 保存路径
        '''
        self.data_path=data_path
        self.save_path=save_path

    def Text_conversion(self):
        '''
        将数据中的文本转成数字
        :return: 
        '''
        data = pandas.read_csv(self.data_path)
        to_replaced = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        replaced_value = [0, 1, 2]

        data = data.replace(to_replace=to_replaced[0], value=replaced_value[0]). \
            replace(to_replace=to_replaced[1], value=replaced_value[1]). \
            replace(to_replace=to_replaced[2], value=replaced_value[2])
        return data.values[:,:-1],data.values[:,-1]  # X,y

    def StandardScaler(self,X_data):
        """Standardization, or mean removal and variance scaling"""
        # 均值为0，方差为1
        scaler =preprocessing.StandardScaler().fit(X_data)
        X_data=scaler.transform(X_data)
        return [scaler,X_data]

    def MinMaxScaler(self,X_data):
        '''
        将特征缩放到给定的最小值和最大值之间，通常在0和1之间，也可以自行设置
        显示公式为：
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
        :param X_data: 
        :return: 
        '''
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_data)
        X_train_minmax = min_max_scaler.fit_transform(X_data)
        return [min_max_scaler,X_train_minmax]

    def MaxAbsScaler(self,X_data):
        '''
        每个特征的最大绝对值被缩放到单位大小
        :param X_data: 
        :return: 
        '''
        max_abs_scaler=preprocessing.MaxAbsScaler().fit(X_data)
        X_train_maxabs=max_abs_scaler.fit_transform(X_data)
        return [max_abs_scaler,X_train_maxabs]

    def QuantileTransformer(self,X_data,output_distribution='normal'):
        '''
        Non-linear transformation
        将每个功能放在相同的范围或分布中。 然而，通过执行秩变换，它平滑了异常分布，
        并且比缩放方法更少受离群值的影响。 然而，它确实扭曲了功能内和跨功能的相关性和距离。
        :param X_data: 
        :return: 
        '''
        if output_distribution: # 对应正态分布
            quantile_transformer = preprocessing.QuantileTransformer(output_distribution=output_distribution,
                                                                     random_state=0).fit(X_data)
            X_train_trans = quantile_transformer.fit_transform(X_data)
        else: # 对应均匀分布
            quantile_transformer = preprocessing.QuantileTransformer(random_state=0).fit(X_data)
            X_train_trans = quantile_transformer.fit_transform(X_data)
        return [quantile_transformer,X_train_trans]

    def Normalization(self,X_data,norm='l1'):
        '''
        将样本归一化为单位范数。
        可以在使用l1或l2规范的单个阵列数据集上执行此操作
        :param X_data: 
        :return: 
        '''
        normalizer = preprocessing.Normalizer(norm=norm).fit(X_data)
        norm_X_data=normalizer.transform(X_data)
        return [normalizer,norm_X_data]

    def Binarization(self,X_data):
        '''
        根据阈值对数据进行二值化（将特征值设置为0或1）
        :param X_data: 
        :return: 
        '''
        binarizer = preprocessing.Binarizer(threshold=0.0).fit(X_data)
        bina_X_data=binarizer.transform(X_data)
        return [binarizer,bina_X_data]

    def PolynomialFeatures(self,X_data,degree=2,interaction_only=True):
        '''
        生成多项式和交互特征,特征列会增加
        如果输入样本是二维的，并且形式为[a，b]，则2级多项式特征为[1，a，b，a ^ 2，ab，b ^ 2]。
        只需要功能之间的交互项，并且可以通过设置来获得interaction_only=True  [1,a,b,ab]
        :param X_data: 
        :return: 
        '''
        poly = preprocessing.PolynomialFeatures(degree=degree,interaction_only=interaction_only).fit(X_data) # 2级多项式
        poly_X_data=poly.fit_transform(X_data)
        return [poly,poly_X_data]

    def PCA(self,X_data,n_components=2,svd_solver='randomized',whiten=True): # svd_solver='full'
        pca = PCA(n_components=n_components,svd_solver=svd_solver,
          whiten=whiten).fit(X_data)
        X_data=pca.transform(X_data)
        return [pca,X_data]

    def IncrementalPCA(self,X_data,n_components=2,batch_size=10):
        '''
        IncrementalPCA对象使用不同的处理形式，并允许部分计算几乎完全匹配PCA的结果，同时以迷你形式处理数据
        :param X_data: 
        :return: 
        '''
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size).fit(X_data)
        X_ipca = ipca.fit_transform(X_data)
        return [ipca,X_ipca]

    def KernelPCA(self,X_data):
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        X_kpca = kpca.fit_transform(X_data)
        # X_back = kpca.inverse_transform(X_kpca)
        return [kpca,X_kpca]

    # BN 层
    def batch_norm_layer(self,inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                       lambda: batch_norm(inputT, is_training=True,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                       lambda: batch_norm(inputT, is_training=False,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                          scope=scope))  # , reuse = True))


# 数据打乱
class Data_random_shuffle():
    def __init__(self,data):
        self.data=data

    def numpy_shuffle(self):
        # 直接打乱
        np.random.seed(100) # 设置随机因子
        np.random.shuffle(self.data)
        return self.data

    def numpy_shuffle_index(self):
        # 先打乱数组的索引
        index=np.arange(0,len(self.data))
        np.random.seed(100) # 设置随机因子
        np.random.shuffle(index)
        return self.data[index]

    def sklearn_shuffle(self):
        return shuffle(self.data)

    def tf_shuffle(self,batch_size):
        # #将数据加入tf队列
        data = tf.train.input_producer(self.data, shuffle=True)
        # #每次出队一个
        data2 = tf.reshape(data.dequeue(),[1,self.data.shape[1]])
        for _ in range(batch_size-1):
            data2=tf.concat([data2,tf.reshape(data.dequeue(),[1,self.data.shape[1]])],0)
        return data2

    def tf_shuffle2(self,batch_size):
        min_after_dequeue = 1000
        capacity = min_after_dequeue + batch_size

        data = tf.train.shuffle_batch(
            [self.data.reshape([1,3])], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return data


if __name__=="__main__":
    # d=Data_processing('iris.data')
    # X,y=d.Text_conversion()
    #
    # # print(X.shape,y.shape) # (149, 4) (149,)
    #
    # # 选取75%做训练，25%做测试
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.25, random_state=42) # 或 y[:,np.newaxis]
    #
    # # 只对X做归一化
    # scaler, X_train = d.StandardScaler(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.MinMaxScaler(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.MaxAbsScaler(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.QuantileTransformer(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.Normalization(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.Binarization(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.PolynomialFeatures(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.PCA(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.IncrementalPCA(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape, X_test.shape)
    #
    # scaler, X_train = d.KernelPCA(X_train)
    # X_test = scaler.transform(X_test)
    # # print(np.mean(X_train, 0), np.mean(X_test, 0))
    # # print(np.var(X_train, 0), np.var(X_test, 0))
    # print(X_train.shape,X_test.shape)

    #'''tf数据打乱
    data = np.array([[3, 4, 1], [2, 4, 0], [1, 3, 1]])
    d=Data_random_shuffle(data)
    data=d.tf_shuffle(10)
    # print(data.get_shape())
    sess=tf.InteractiveSession()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            tf.global_variables_initializer().run()
            print(sess.run(data))
            exit(0)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    #'''












