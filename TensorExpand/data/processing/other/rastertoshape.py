#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
栅格影像 转成 对应的 shp文件
栅格矢量化
"""

import sys
from osgeo import ogr, osr, gdal
from osgeo import gdalconst

class RasterToShape(object):
    def __init__(self):
        pass

    def rasterToShape(self, raster, shape):
        src_tif = raster # 栅格影像路径
        out_shp = shape # 输出的shp文件路径

        src_raster = gdal.Open(src_tif, gdalconst.GA_ReadOnly)# 打开栅格影像
        if not src_raster:
            print("can not open {}".format(src_tif), file = sys.stderr)
            return False
        ogr_drv = ogr.GetDriverByName("ESRI Shapefile") # 获取矢量驱动
        if not ogr_drv:
            print('No "ESRI Shapefile" driver', file = sys.stderr)
            return False
            
        out_data_source = ogr_drv.CreateDataSource(out_shp)# 创建shp 数据集
        spatialRef = osr.SpatialReference(src_raster.GetProjectionRef())# 设置空间参考
        # create layer
        out_layer = out_data_source.CreateLayer("raster", spatialRef, ogr.wkbLineString) # 创建图层(图层要素 环形线)
        # add field
        out_layer_field = ogr.FieldDefn("DN") #
        out_layer.CreateField(out_layer_field) #图层中添加文件
        src_band = src_raster.GetRasterBand(1) #获取栅格图像第一个波段
        gdal.Polygonize(src_band, src_band, out_layer, 0)# 进行矢量化

        src_band = None
        src_raster = None
        ogr_drv = None
        out_layer_field = None
        out_data_source = None
        return True

# argv[0] <out shapefile> <raster file>
if __name__ == "__main__":
    #'''
    if len(sys.argv) != 3:
        print("{} <out shapefile> <raster file>".format(sys.argv[0]), file = sys.stderr)
        sys.exit(1)
    rs = RasterToShape()
    r = rs.rasterToShape(sys.argv[1], sys.argv[2])
    if not r:
        sys.exit(1)
    '''
    rs = RasterToShape()
    rs.rasterToShape(r"F:\PL\2\34561.tiftest_mask.tiff.tiff",r'F:\PL\2\shp\3456123.shp')
    '''

