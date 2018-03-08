# -*- coding: utf-8 -*-

from osgeo import gdal, ogr,osr
from osgeo.gdalconst import *
import numpy as np
import os
from os import path
import cv2
import matplotlib.pyplot as plt

gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")

gdal.AllRegister() #注册驱动
ogr.RegisterAll()


img_path="E:/test/04.tif"
# img_path=input('输入影像路径:')
srcDS=gdal.Open(img_path,GA_ReadOnly)# 只读方式打开影像
geoTrans = srcDS.GetGeoTransform() # 获取地理参考6参数
srcPro=srcDS.GetProjection() # 获取坐标引用

srcXSize=srcDS.RasterXSize # 宽度
srcYSize=srcDS.RasterYSize # 高度
nbands=srcDS.RasterCount # 波段数

vector_fn_sample="E:/test/04.shp"

raster_fn_sample ="E:/test/04_mask.tif" # 存放掩膜影像

if os.path.exists(raster_fn_sample):
    gdal.GetDriverByName('GTiff').Delete(raster_fn_sample)# 删除掉样本提取掩膜

source_ds = ogr.Open(vector_fn_sample) # 打开矢量文件
source_layer = source_ds.GetLayer() # 获取图层  （包含图层中所有特征  所有几何体）
mark_ds = gdal.GetDriverByName('GTiff').Create(raster_fn_sample, srcXSize, srcYSize, 1, gdal.GDT_Byte)  # 1表示1个波段，按原始影像大小生成 掩膜
mark_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
mark_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
band = mark_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
band.SetNoDataValue(0)  # 将这个波段的值全设置为0

# Rasterize 矢量栅格化
gdal.RasterizeLayer(mark_ds, [1], source_layer, burn_values=[1])  # 几何体内的值全设置为1
mark_ds.FlushCache()  # 将数据写入文件


img=srcDS.GetRasterBand(1).ReadAsArray(0, 0, srcXSize, srcYSize, srcXSize, srcYSize)
mask = mark_ds.GetRasterBand(1).ReadAsArray(0, 0, srcXSize, srcYSize, srcXSize, srcYSize)

plt.subplot(121)
plt.imshow(img,'gray')

plt.subplot(122)
plt.imshow(mask,'gray')

plt.show()

srcDS=None
mark_ds=None
source_ds=None