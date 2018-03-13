# -*- coding:utf8 -*-

import pickle

pkl_path='./data/data_200.pkl'

with open(pkl_path,'rb') as fp:
    data=pickle.load(fp)

print(data[0][0])
exit(-1)
pass
