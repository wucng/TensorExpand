#! /usr/bin/python
# -*- coding: utf8 -*-

'''
直接从.zip文件中提取数据和标签，用于训练
'''

import pandas as pd
pd.set_option('display.max_columns', 500)
import zipfile
import numpy as np
# from sklearn.preprocessing import scale

with zipfile.ZipFile('D:/MultiSpectralImages.zip', 'r') as z:
    f = z.open('Labels.csv')
    data = pd.read_csv(f)
    data_label=data.iloc[:,0:2].values # 只有前两列需要

    label=set(data_label[:,0]) # 去除重复的
    # print(label)
    label_dict={k:v for v,k in enumerate(label)} # 文本标签与数值对应起来
    # print(label_dict)
    data=None
    f.close()

# 数据与标签对应起来
with zipfile.ZipFile('D:/MultiSpectralImages.zip', 'r') as z:
    # for i in z.namelist():
    m=0
    for i,item in enumerate(data_label[:,1]):
        try:
            f = z.open(str(item))
        except:
            continue
        data = pd.read_csv(f, index_col=0)
        data=data.iloc[:, 1:-1].values/255. # shape [12250,10] -->每张图片[350,350,10]
        if m==0:
            data2=np.hstack((data.flatten(),label_dict[data_label[i,0]]))[np.newaxis,:]
        else:
            data2=np.vstack((data2,np.hstack((data.flatten(),label_dict[data_label[i,0]]))[np.newaxis,:]))

        m += 1

        if m%100==0: # 每100个保存一次
            print(data2.shape) # [none,122501]
            pd.to_pickle(data2, 'data_'+str(i)+'.pkl')  # 保存为pickle文件
            data2 = None
            m=0
            print('数据已保存！')

    data=None
    f.close()

    print(data2.shape)
    pd.to_pickle(data2, 'data_0.pkl')  # 保存为pickle文件
    data2 = None
    print('数据保存完成！')
