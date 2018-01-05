#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
从网上下载数据
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import urllib
from urllib import request
import os

class Data_downing():
    def __init__(self,url,file_path):
        self.url=url
        self.file_path=file_path

    def downing(self):
        print("downloading...")
        # 第一个参数网络地址；第二个参数 本地保存位置
        request.urlretrieve(self.url, self.file_path)
        print("finished!")


if __name__=="__main__":
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    file_path="./iris.data"
    if not os.path.exists(file_path):
        Data_downing(url,file_path).downing()
    else:
        print("数据已经存在！")
        exit(0)
