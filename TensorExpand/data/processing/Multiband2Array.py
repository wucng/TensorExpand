#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
import gdal
import sys

'''
# 多波段图像(遥感图像)提取每个波段信息转换成数组（波段数>=4 或者 波段数<=3）
# 一般的方法如：opencv，PIL，skimage 最多只能读取3个波段
# path 图像的路径
# return： 图像数组
'''
def Multiband2Array(path):

    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s'% path)
        sys.exit(1)

    xcount=src_ds.RasterXSize # 宽度
    ycount=src_ds.RasterYSize # 高度
    ibands=src_ds.RasterCount # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band) # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float32)
        if band==1:
            data=dataraster.reshape((ycount,xcount,1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data=np.append(data,dataraster.reshape((ycount,xcount,1)),axis=2)

    return data

def Multiband2Array(path,channels):

    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s'% path)
        sys.exit(1)

    xcount=src_ds.RasterXSize # 宽度
    ycount=src_ds.RasterYSize # 高度
    ibands=src_ds.RasterCount # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    # if ibands==4:ibands=3
    ibands=min(channels,ibands)
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band) # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float16)
        if ibands==1:return dataraster.reshape((ycount,xcount))
        if band==1:
            data=dataraster.reshape((ycount,xcount,1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data=np.append(data,dataraster.reshape((ycount,xcount,1)),axis=2)

    return data

if __name__=="__main__":
    cwd = os.getcwd()  # 返回脚本文件所在的工作目录
    path = os.path.join(cwd, '10.tif')
    path=r"F:\PL\2_1.tif"
    img=Multiband2Array(path)
    print(img.shape)
    # img = img.resize((28, 28,3))

    img_raw = img.tobytes()  # 将图片转化为原生bytes

    pass
