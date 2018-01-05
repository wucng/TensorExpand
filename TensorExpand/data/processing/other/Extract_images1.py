# -*- coding: utf-8 -*-
'''
掩膜按照原始影像大小进行生成
制作10x10的样本，每次读取10行像素（原始影像与掩膜影像），取10x10的模版，循环这个模版
如果有像素点落在掩膜内，则保存为1类数据（输出为图像），掩膜外保存为另一类数据
一个提取影像的shp文件，无裁剪shp
'''

from osgeo import gdal, ogr
from osgeo.gdalconst import *
import numpy as np
import os
from os import path
import gdalnumeric

# 为了支持中文路径，请添加下面这句代码
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")

gdal.AllRegister() #注册驱动
ogr.RegisterAll()

module_path = path.dirname(__file__) # 返回脚本文件所在的工作目录

srcImagePath=path.join(module_path,'11.tif')
srcDS=gdal.Open(srcImagePath,GA_ReadOnly)# 只读方式打开原始影像

isize=10 # 制作10x10的样本

geoTrans = srcDS.GetGeoTransform() # 获取地理参考6参数
srcPro=srcDS.GetProjection() # 获取坐标引用
srcXSize=srcDS.RasterXSize # 宽度
srcYSize=srcDS.RasterYSize # 高度
nbands=srcDS.RasterCount # 波段数

# 原始影像的左上角坐标 (geoTrans[0],geoTrans[3])
# 原始影像的左上角坐标（地理坐标）
srcX=geoTrans[0]+srcXSize*geoTrans[1]+srcYSize*geoTrans[2]
srcY=geoTrans[3]+srcXSize*geoTrans[4]+srcYSize*geoTrans[5]

# print(ds.GetDriver().ShortName)

# Filename of input OGR file
vector_fn = path.join(module_path,'11.shp')

# Filename of the raster Tiff that will be created
raster_fn =path.join(module_path,'test_mask.tiff')  # 存放掩膜影像

# Open the data source and read in the extent
source_ds = ogr.Open(vector_fn) # 打开矢量文件
source_layer = source_ds.GetLayer() # 获取图层
# x_min, x_max, y_min, y_max = source_layer.GetExtent() # 矢量文件的四至范围 左上角(x_min，y_max) 右下角（x_max，y_min）


target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, srcXSize, srcYSize, 1, gdal.GDT_Byte)# 1表示1个波段，按原始影像大小生成 掩膜
# target_ds.SetGeoTransform((geoTrans2[0], geoTrans2[1], geoTrans2[2], geoTrans2[3], geoTrans2[4], geoTrans2[5])) # 设置掩膜的地理参考
target_ds.SetGeoTransform(geoTrans) # 设置掩膜的地理参考
target_ds.SetProjection(srcPro) # 设置掩膜坐标引用
band = target_ds.GetRasterBand(1)#获取第一个波段(影像只有一个波段)
band.SetNoDataValue(0)# 将这个波段的值全设置为0

# Rasterize 矢量栅格化
gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[255])# 几何体内的值全设置为255

target_ds.FlushCache() # 将数据写入文件

'''
以上完成掩膜
'''
'''
# 获取原始影像数据类型
eDT=srcDS.GetRasterBand(1).DataType

if eDT==1:
    eDT=gdal.GDT_Byte
    data_type=np.uint8
elif eDT==2:
    eDT=gdal.GDT_UInt16
    data_type = np.uint16
elif eDT==3:
    eDT=gdal.GDT_Int16
    data_type = np.int16
elif eDT==4:
    eDT=gdal.GDT_UInt32
    data_type = np.uint32
elif eDT==5:
    eDT=gdal.GDT_Int32
    data_type = np.int32
'''

# 转成8bit
eDT=gdal.GDT_Byte
data_type=np.uint8

markDS=gdal.Open(raster_fn,GA_ReadOnly)# 只读方式打开掩膜影像

outfile='outfile'
# outDriver = ogr.GetDriverByName("ESRI Shapefile")
if not os.path.isdir(path.join(module_path, outfile)):
    os.mkdir(path.join(module_path, outfile))  # 创建文件夹

clip_image_0 =path.join(module_path,outfile,'0')  # 存放样本
clip_image_1 =path.join(module_path,outfile,'1')


# 创建2个文件夹
if not os.path.isdir(clip_image_0):os.mkdir(clip_image_0)
if not os.path.isdir(clip_image_1):os.mkdir(clip_image_1)

# Create a new geomatrix for the image
geoTrans2 = list(geoTrans)

srcBuf=np.zeros([isize,srcXSize,nbands],data_type) # 用于存储所有波段缓存
m=0 # 文件名
# 按波段每次读取isize行像素(按照掩膜范围内的行读取)
flagY = True
for i in range(0,srcYSize,isize//2):# 行变化 range(0,srcYSize,isize) 如果步长取isize 由于存在黑边像素，会导致丢失一些图像
    if not flagY: break
    if m>100:break # 查看100张情况
    if i+isize>srcYSize-1:
        i=srcYSize-1-isize
        flagY=False
    markBuf = markDS.GetRasterBand(1).ReadAsArray(0,i,srcXSize,isize,srcXSize,isize)# 读取isize行
    for b in range(1,nbands+1):
        srcBuf[:,:,b-1]=srcDS.GetRasterBand(b).ReadAsArray(0,i,srcXSize,isize,srcXSize,isize).astype(data_type)# 读取isize行  掩膜影像与原始影像大小和地理参考一样
    # srcBuf = srcDS.GetRasterBand(1).ReadAsArray(0, i, srcXSize, isize, srcXSize, isize)

    # dstBuf = np.zeros((isize,isize),data_type) # 存放isize x isize像素
    flagX = True
    for k in range(0,srcXSize,isize//2):# 如果步长取isize 由于存在黑边像素，会导致丢失一些图像
        if not flagX: break

        if k+isize>srcXSize-1:
            k=srcXSize-1-isize
            flagX=False
        # dstBuf=srcBuf[:,k:k+isize,0] # 取出原始像素的 isize x isize 的矩阵
        markBuf2=markBuf[:,k:k+isize]

        # 去除点含有黑边像素的图像
        flag0 = True
        for ii in range(isize):
            if flag0==False:break
            for jj in range(isize):
                if flag0 == False: break
                if not (srcBuf[:, k:k + isize, 0])[ii, jj]:
                    flag0=False

        flagZ = True
        if flag0 == True:
            geoTrans2[0] = geoTrans[0]+k*geoTrans[1]  # 重新设置左上角地理坐标
            geoTrans2[3] = geoTrans[3]+i*geoTrans[5]

            for ii in range(isize):
                if flagZ == False: break
                for jj in range(isize):
                    if flagZ == False: break
                    if markBuf2[ii,jj]:
                        clip_image = path.join(clip_image_1,str(m)+'.tiff')  # 存放裁剪影像
                        # target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, x_res, y_res,nbands, gdal.GDT_Byte)# 创建裁剪影像数据集
                        target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, isize, isize, nbands,
                                                                         eDT)  # 创建裁剪影像数据集
                        target_ds.SetGeoTransform(geoTrans2)  # 设置裁剪影像的地理参考
                        target_ds.SetProjection(srcPro)  # 设置裁剪影像坐标引用
                        for b in range(1, nbands + 1):
                            target_ds.GetRasterBand(b).WriteArray(srcBuf[:,k:k+isize,b-1], 0, 0)
                        target_ds.FlushCache()  # 将数据写入文件
                        flagZ=False

            if flagZ:
                clip_image = path.join(clip_image_0, str(m)+'.tiff')  # 存放裁剪影像
                # target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, x_res, y_res,nbands, gdal.GDT_Byte)# 创建裁剪影像数据集
                target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, isize, isize, nbands,
                                                                 eDT)  # 创建裁剪影像数据集
                target_ds.SetGeoTransform(geoTrans2)  # 设置裁剪影像的地理参考
                target_ds.SetProjection(srcPro)  # 设置裁剪影像坐标引用
                for b in range(1, nbands + 1):
                    target_ds.GetRasterBand(b).WriteArray(srcBuf[:, k:k + isize, b - 1], 0, 0)
                # target_ds.GetRasterBand(b).WriteArray(dstBuf, 0, 0)
                target_ds.FlushCache()  # 将数据写入文件
            m=m+1 # 记录图像名


