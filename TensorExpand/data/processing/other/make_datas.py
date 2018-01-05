#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
生成训练集与生成测试集
"""

import tool_set
import datetime
# dir_name='/home/ubuntu_wu/桌面/image_60/image3/image1/'
# dir_name='/home/ubuntu_wu/桌面/image_60/image5/'
# dir_name='/home/ubuntu_wu/桌面/image_60/image6/'



starttime = datetime.datetime.now()
# 生成训练集
# dir_name='F:/PL/train_data/'
dir_name='./outfile/'
tool_set.create_pickle_train2(dir_name,10,3)
# 生成测试集
# dir_name='F:/PL/test_data/'
# dir_name='F:/PL/2/test_data/'
# dir_name='F:/PL/20170321_022817_0e0d_visual/test_data/'
# tool_set.create_pickle_test(dir_name,10,3)

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

