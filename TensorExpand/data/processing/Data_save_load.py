#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
数据存储与加载
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import csv
import pickle
import json
import h5py
import numpy as np
import gzip
import dask.array as da
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession

# 数据存储与加载
class Data_save_load():
    def __init__(self,file_path,data=None):
        self.file_path=file_path
        self.data=data
        self.sc = SparkContext(conf=SparkConf().setAppName("The first example"))

    def Save_numpy_array(self):
        np.save(file=self.file_path,arr=self.data)
    def Load_numpy_array(self):
        return np.load(self.file_path)

    def Save_csv(self):
        pandas.DataFrame(self.data).to_csv(self.file_path)

    def Load_csv(self,header=False):
        # data = pandas.read_csv(self.file_path)
        data=pandas.read_csv(self.file_path, header=header) # header=Fasle 跳过标题行
        return data

    def Save_csv2(self,is_gzip=True):

        if is_gzip: # 压缩文件
            with gzip.open(self.file_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(np.zeros([1, self.data.shape[1]]))  # 随便写一行做标题
                writer.writerows(self.data)  # 写入多行
        else:
            with open(self.file_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(np.zeros([1, self.data.shape[1]]))  # 随便写一行做标题
                writer.writerows(self.data)  # 写入多行

    def Load_csv2(self,header=False,is_gzip=True):
        if is_gzip:
            with gzip.open(self.file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                if not header: next(reader)  # 这样做可以跳过第一行，通常第一行为头标题
                result = []
                [result.append(line) for line in reader if line]
        else:
            with open(self.file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                if not header:next(reader)  # 这样做可以跳过第一行，通常第一行为头标题
                result = []
                [result.append(line) for line in reader if line]
        data = [list(map(lambda x: int(x), line)) for line in result]  # str-->int
        return np.array(data)

    def Save_pickle(self):
        pandas.DataFrame(self.data).to_pickle(self.file_path)
    def Load_pickle(self):
        return pandas.read_pickle(self.file_path)

    def Save_pickle2(self,is_gzip=True):
        if is_gzip:
            with gzip.open(self.file_path, 'wb') as file:
                pickle.dump(data, file)
        else:
            with open(self.file_path, 'wb') as file:
                pickle.dump(data, file)

    def Load_pickle2(self,is_gzip=True):
        if is_gzip:
            with gzip.open(self.file_path,'rb') as file:
                return pickle.load(file)
        else:
            with open(self.file_path,'rb') as file:
                return pickle.load(file)

    def Save_json(self):
        pandas.DataFrame(self.data).to_json(self.file_path)

    def Load_json(self):
        return pandas.read_json(self.file_path)

    def Save_json2(self):
        result = []
        [result.append(str(y)) for x in self.data for y in x]
        with open(self.file_path, 'w') as file:
            json.dump(result, file)

    def Load_json2(self):
        with open(self.file_path, 'r') as file:
            text = json.load(file)
            result = []
            [result.append(int(x)) for x in text]
            return np.array(result)

    def Save_hdf(self):
        pandas.DataFrame(self.data).to_hdf(self.file_path,self.file_path.split('.')[-1])

    def Load_hdf(self):
        return pandas.read_hdf(self.file_path)

    def Save_hdf5(self):
        with h5py.File(self.file_path,'w') as h5f:
            h5f.create_dataset('data', data=self.data[:,:-1])
            h5f.create_dataset('label',data=self.data[:,-1])

    def Load_hdf5(self):
        with h5py.File(self.file_path, 'r') as h5f:
            # data.keys()  # #可以查看所有的主键
            data=h5f['data'][:] # 取出主键为data的所有的键值
            label=h5f['label'][:]
            return [data,label]

    def numpy_to_dask(self):
        # Create DASK array using numpy arrays
        # (Note that it can work with HDF5 Dataset too)
        data=da.from_array(self.data,chunks=(1000, 1000))
        return data

    def Save_html(self):
        pandas.DataFrame(self.data).to_html(self.file_path)
    def Load_html(self):
        data=pandas.read_html(self.file_path)
        return pandas.DataFrame(data.pop(0)).values[:, 1:]

    def Save_pickle_with_spark(self):
        self.sc.parallelize(self.data).saveAsPickleFile(self.file_path)
    def Load_pickle_with_spark(self): # 只能加载使用spark saveAsPickleFile保存的pickle文件
        textFiles = self.sc.pickleFile(self.file_path)
        data = textFiles.collect()
        return np.array(data, np.float16)

    def Save_csv_with_spark(self):
        self.sc.parallelize(self.data).saveAsTextFile(self.file_path)
    def Load_csv_with_spark(self):
        textFiles = self.sc.textFile(self.file_path)
        data = textFiles.collect()
        result = []
        [result.append(int(y)) for x in data for y in x.strip('[]').split(' ')]
        return np.array(result,np.float16)

if __name__=="__main__":
    data = np.array([[3, 4, 1], [2, 4, 0], [1, 3, 1]])
    # d=Data_save_load(r"123.pkl.gz",data)
    # d.Save_pickle2()
    # print(d.Load_pickle2())

    # d=Data_save_load('123.json.gz',data)
    # d.Save_json2()
    # print(d.Load_json())

    # d=Data_save_load('123.h5',data)
    # d.Save_hdf5()
    # print(d.numpy_to_dask())

    # d = Data_save_load('123.h5', data)
    # d.Save_hdf()
    # print(d.Load_hdf())

    # d=Data_save_load('123.html',data)
    # d.Save_html()
    # print(d.Load_html())

    d = Test('123.pkl', data)
    # d.Save_pickle_with_spark()
    # print(d.Load_pickle_with_spark())

    # d.Save_csv_with_spark()
    data2 = d.Load_csv_with_spark()
    print(data2)
    print(type(data2))


