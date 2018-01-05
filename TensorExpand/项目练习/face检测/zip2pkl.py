#! /usr/bin/python
# -*- coding: utf8 -*-
import zipfile
import numpy as np
from PIL import Image
import pandas as pd

# 解析zip文件中的数据与标签
with zipfile.ZipFile('D:/faces.zip', 'r') as z:
    filenames=z.namelist()
    m=0
    for i,filename in enumerate(filenames):
        if not '.png' in filename: continue
        
        f = z.open(filename) # 得到文件路径
        if str(f).split('/')[-3]=='train':          
            img=Image.open(f)
            img=np.array(img,np.uint8).flatten()/255.

            if str(f).split('/')[-2]=='face':
                label=1
            else:
                label=0
            if m==0:
                data=np.hstack((img,label))[np.newaxis,:]
            else:
                data=np.vstack((data,np.hstack((img,label))[np.newaxis,:]))

            m += 1

            if m%10000==0: # 每10000个保存一次
                print(data.shape) # [none,19*19*(1+4)]
                pd.to_pickle(data, 'data_'+str(i)+'.pkl')  # 保存为pickle文件
                data = None
                m=0
                print('数据已保存！')
                # exit(-1)

    f.close()
    print(data.shape)
    pd.to_pickle(data, 'data_0.pkl')  # 保存为pickle文件
    data = None
    print('数据保存完成！')
