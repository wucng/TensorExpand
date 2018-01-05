# -*- coding: utf-8 -*-
'''
融合影像裁剪+按shp提起样本
2个shp文件，第一个shp确定像素提取范围(裁剪)，第二个shp确定样本提取，
方法思路：
1、2个shp文件都按照原图大小生成掩膜图像
2、先判断像素点是否落在第一个shp里，（不在 跳过）如果在（在像素提取范围内），如果满足再进行第3步判断，提取样本
3、再判断是否落在第二个shp里，提取样本

补充：根据裁剪shp文件先大致确定一个像素提取范围
'''

from osgeo import gdal, ogr,osr
from osgeo.gdalconst import *
import numpy as np
import os
from os import path

# 为了支持中文路径，请添加下面这句代码
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")

gdal.AllRegister() #注册驱动
ogr.RegisterAll()

img_path="11.tif"
# img_path=input('输入影像路径:')
srcDS=gdal.Open(img_path,GA_ReadOnly)# 只读方式打开影像
geoTrans = srcDS.GetGeoTransform() # 获取地理参考6参数
srcPro=srcDS.GetProjection() # 获取坐标引用

# prjRef
# prjRef = osr.SpatialReference(srcPro) # (栅格坐标参考)转换成矢量坐标参考

srcXSize=srcDS.RasterXSize # 宽度
srcYSize=srcDS.RasterYSize # 高度
nbands=srcDS.RasterCount # 波段数

module_path = path.dirname(__file__) # 返回脚本文件所在的工作目录

# 生成掩膜影像(确定样本提取)
# vector_fn_sample=input('输入掩膜shp路径:')
vector_fn_sample="11.shp"

raster_fn_sample ="sample_mask.tiff"  # 存放掩膜影像

if os.path.exists(raster_fn_sample):
    gdal.GetDriverByName('GTiff').Delete(raster_fn_sample)# 删除掉样本提取掩膜

source_ds_1 = ogr.Open(vector_fn_sample) # 打开矢量文件
source_layer_1 = source_ds_1.GetLayer() # 获取图层  （包含图层中所有特征  所有几何体）
mark_ds = gdal.GetDriverByName('GTiff').Create(raster_fn_sample, srcXSize, srcYSize, 1, gdal.GDT_Byte)  # 1表示1个波段，按原始影像大小生成 掩膜
mark_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
mark_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
band = mark_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
band.SetNoDataValue(0)  # 将这个波段的值全设置为0
# Rasterize 矢量栅格化
gdal.RasterizeLayer(mark_ds, [1], source_layer_1, burn_values=[255])  # 几何体内的值全设置为255
mark_ds.FlushCache()  # 将数据写入文件
source_ds_1=None

markDS_sample=gdal.Open(raster_fn_sample,GA_ReadOnly)# 只读方式打开掩膜影像

# 生成掩膜影像（确定像素提取范围）
# vector_fn_clip=input('输入裁剪shp路径:')
vector_fn_clip="clip/clip_11.shp"
raster_fn_clip ="clip_mask.tiff"  # 存放掩膜影像

if os.path.exists(raster_fn_clip):
    gdal.GetDriverByName('GTiff').Delete(raster_fn_clip)# 删除掉裁剪范围掩膜

# Open the data source and read in the extent
source_ds = ogr.Open(vector_fn_clip) # 打开矢量文件
source_layer = source_ds.GetLayer() # 获取图层  （包含图层中所有特征  所有几何体）
# x_min, x_max, y_min, y_max = source_layer.GetExtent() # 矢量文件的四至范围 左上角(x_min，y_max) 右下角（x_max，y_min）
# featureCount = source_layer.GetFeatureCount() # 获取图层中特征数
x_min, x_max, y_min, y_max = source_layer.GetExtent() # 获取几何体四至范围，确定裁剪范围(提取像素范围)

# 先大致确定一个像素提取范围
# 计算出落在图像内的像素行 变化范围，根据裁剪影像的范围反算出对应原始影像的行变化范围
y1 = int((y_max - geoTrans[3]) / geoTrans[5])
y2 = int((y_min - geoTrans[3]) / geoTrans[5])+1

# 计算出落在图像内的像素列 变化范围，根据裁剪影像的范围反算出对应原始影像的行变化范围
x1 = int((x_min - geoTrans[0]) / geoTrans[1])
x2 = int((x_max - geoTrans[0]) / geoTrans[1])+1

mark_ds = gdal.GetDriverByName('GTiff').Create(raster_fn_clip, srcXSize, srcYSize, 1, gdal.GDT_Byte)  # 1表示1个波段，按原始影像大小生成 掩膜
mark_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
mark_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
band = mark_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
band.SetNoDataValue(0)  # 将这个波段的值全设置为0
# Rasterize 矢量栅格化
gdal.RasterizeLayer(mark_ds, [1], source_layer, burn_values=[255])  # 几何体内的值全设置为255
mark_ds.FlushCache()  # 将数据写入文件
source_ds=None

markDS_clip=gdal.Open(raster_fn_clip,GA_ReadOnly)# 只读方式打开掩膜影像


outfile='outfile'
# outDriver = ogr.GetDriverByName("ESRI Shapefile")
if not os.path.isdir(path.join(module_path, outfile)):
    os.mkdir(path.join(module_path, outfile))  # 创建文件夹

clip_image_0 =path.join(module_path,outfile,'0')  # 存放样本
clip_image_1 =path.join(module_path,outfile,'1')

# 创建2个文件夹
if not os.path.isdir(clip_image_0):os.mkdir(clip_image_0)
if not os.path.isdir(clip_image_1):os.mkdir(clip_image_1)

m=0 # 文件名
isize=10 # 样本大小 4x4

eDT=gdal.GDT_Byte
data_type=np.uint8

# Create a new geomatrix for the image
geoTrans2 = list(geoTrans)

srcBuf=np.zeros([isize,srcXSize,nbands],data_type) # 用于存储所有波段缓存

'''
for feature in source_layer:# 循环图层中每个特征,确定裁剪范围(提起像素范围)
    # geom = feature.GetGeometryRef() # 获取特征对应的几何体
    # print geom.Centroid().ExportToWkt()
    
    # Remove output shapefile if it already exists
    if os.path.exists(path.join(module_path, outfile,"1.shp")):
        outDriver.DeleteDataSource(path.join(module_path, outfile,"1.shp"))

    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(path.join(module_path, outfile,"1.shp"))
    outLayer = outDataSource.CreateLayer("layers", prjRef,geom_type=ogr.wkbPolygon)
    outLayer.CreateFeature(feature)
    x_min, x_max, y_min, y_max = outLayer.GetExtent() # 获取几何体四至范围，确定裁剪范围(提取像素范围)

    # 计算出落在图像内的像素行 变化范围，根据裁剪影像的范围反算出对应原始影像的行变化范围
    y1 = int((y_max - geoTrans[3]) / geoTrans[5])
    y2 = int((y_min - geoTrans[3]) / geoTrans[5])

    # 计算出落在图像内的像素列 变化范围，根据裁剪影像的范围反算出对应原始影像的行变化范围
    x1 = int((x_min - geoTrans[0]) / geoTrans[1])
    x2 = int((x_max - geoTrans[0]) / geoTrans[1])

    outDataSource=None
    if os.path.exists(path.join(module_path, outfile, "1.shp")):
        outDriver.DeleteDataSource(path.join(module_path, outfile, "1.shp"))  # 删除掉shp文件

    #几何体确定的范围，对应到原始影像的行列坐标范围是左上角（x1,y1）,右下角（x2,y2）,只要对这个范围的像素进行读取
    '''
# 按波段每次读取一行像素,一行行处理(按照掩膜范围内的行读取)
flagY = True
for i in range(y1, y2 + 1,isize//2):  # 行变化，按裁剪影像的范围设置
    print("样本提取：%.2f%%" % ((i - y1) / (y2 + 1 - y1) * 100), end="\r") #加上进度信息
    # print("正在裁剪:{0}%".format((i-y1)/(y2+1-y1)*100), end="\r") #进度条信息
    if not flagY: break
    if i + isize > y2:
        i = y2 - isize
        flagY = False
    markBuf_clip = markDS_clip.GetRasterBand(1).ReadAsArray(0, i, srcXSize, isize, srcXSize, isize)  # 读取isize行
    markBuf_sample = markDS_sample.GetRasterBand(1).ReadAsArray(0, i, srcXSize, isize, srcXSize, isize)  # 读取isize行

    for b in range(1, nbands + 1):
        srcBuf[:, :, b - 1] = srcDS.GetRasterBand(b).ReadAsArray(0, i, srcXSize, isize, srcXSize, isize).astype(data_type)  # 读取isize行  掩膜影像与原始影像大小和地理参考一样

    flagX = True
    for k in range(x1, x2+1, isize // 2):  # 如果步长取isize 由于存在黑边像素，会导致丢失一些图像
        if not flagX:
            break
        if k + isize > x2 :
            k = x2 - isize
            flagX = False

        markBuf2_clip = markBuf_clip[:, k:k + isize]
        markBuf2_sample = markBuf_sample[:, k:k + isize]
        # 去除点含有黑边像素的图像
        flag0 = True
        for ii in range(isize):
            if flag0 == False: break
            for jj in range(isize):
                if flag0 == False: break
                if not (srcBuf[:, k:k + isize, 0])[ii, jj]:
                    flag0 = False

        flagZ = True
        flag_clip=True
        if flag0 == True:
            # 判断是否落在裁剪范围内
            for ii in range(isize):
                if flag_clip == False: break
                for jj in range(isize):
                    if flag_clip == False: break
                    if markBuf2_clip[ii,jj]: # 落在裁剪掩膜范围内(落在像素提取范围内)
                        flag_clip = False
                        break

            if flag_clip == False: # 落在裁剪掩膜范围内(落在像素提取范围内)

                geoTrans2[0] = geoTrans[0] + k * geoTrans[1]  # 重新设置左上角地理坐标
                geoTrans2[3] = geoTrans[3] + i * geoTrans[5]

                for ii in range(isize):
                    if flagZ == False: break
                    for jj in range(isize):
                        if flagZ == False: break
                        if markBuf2_sample[ii, jj]: # 落在样本提取掩膜范围内
                            clip_image = path.join(clip_image_1, str(m) + '.tiff')  # 存放裁剪影像
                            # target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, x_res, y_res,nbands, gdal.GDT_Byte)# 创建裁剪影像数据集
                            target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, isize, isize, nbands,
                                                                             eDT)  # 创建裁剪影像数据集
                            target_ds.SetGeoTransform(geoTrans2)  # 设置裁剪影像的地理参考
                            target_ds.SetProjection(srcPro)  # 设置裁剪影像坐标引用
                            for b in range(1, nbands + 1):
                                target_ds.GetRasterBand(b).WriteArray(srcBuf[:, k:k + isize, b - 1], 0, 0)
                            target_ds.FlushCache()  # 将数据写入文件
                            flagZ = False

                if flagZ: # 没有落在掩膜范围内
                    clip_image = path.join(clip_image_0, str(m) + '.tiff')  # 存放裁剪影像
                    # target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, x_res, y_res,nbands, gdal.GDT_Byte)# 创建裁剪影像数据集
                    target_ds = gdal.GetDriverByName('GTiff').Create(clip_image, isize, isize, nbands,
                                                                     eDT)  # 创建裁剪影像数据集
                    target_ds.SetGeoTransform(geoTrans2)  # 设置裁剪影像的地理参考
                    target_ds.SetProjection(srcPro)  # 设置裁剪影像坐标引用
                    for b in range(1, nbands + 1):
                        target_ds.GetRasterBand(b).WriteArray(srcBuf[:, k:k + isize, b - 1], 0, 0)
                    # target_ds.GetRasterBand(b).WriteArray(dstBuf, 0, 0)
                    target_ds.FlushCache()  # 将数据写入文件
                m = m + 1  # 记录图像名

# feature=None
# outDataSource = None

print("样本提取：100.00%")
# close DataSource
if os.path.exists(raster_fn_clip):
    gdal.GetDriverByName('GTiff').Delete(raster_fn_clip)# 删除掉裁剪范围掩膜
if os.path.exists(raster_fn_sample):
    gdal.GetDriverByName('GTiff').Delete(raster_fn_sample)# 删除掉样本提取掩膜
# if os.path.exists(path.join(module_path, outfile,"1.shp")):
#     outDriver.DeleteDataSource(path.join(module_path, outfile,"1.shp"))# 删除掉shp文件
source_ds = None
srcDS=None
markDS_clip=None
markDS_sample=None
exit()

