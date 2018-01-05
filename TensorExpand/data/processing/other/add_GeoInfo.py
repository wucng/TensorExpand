#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
opencv去除噪声后，丢失坐标信息，使用该脚本重新加上坐标信息
"""

import gdal, ogr
import os
import sys


def add_GeoInfo(argv):
    srcDS = gdal.Open(argv[2])
    geoTrans = srcDS.GetGeoTransform()
    srcPro = srcDS.GetProjection()

    target_ds = gdal.Open(argv[1],gdal.GA_Update)
    target_ds.SetGeoTransform(geoTrans)
    target_ds.SetProjection(srcPro)

    target_ds.FlushCache()

    target_ds = None
    
    
def add_GeoInfo_batch():
    srcDS = gdal.Open('E:/ChinaRS/data/gft1_test/test.tif')
    geoTrans = srcDS.GetGeoTransform()
    srcPro = srcDS.GetProjection()

    for fn in os.listdir("binary_image/"):
        fn = os.path.join("E:/ChinaRS/data/gft1_test/binary_image/" ,fn)
        target_ds = gdal.Open(fn,gdal.GA_Update)
        target_ds.SetGeoTransform(geoTrans)
        target_ds.SetProjection(srcPro)

        target_ds.FlushCache()

        target_ds = None

        
        
def add_GeoInfo_dir(img_dir):
    """for directory including tif and mask"""
    lst = os.listdir(img_dir)
    if len(lst) % 2 != 0:
        print("Please check the numbers of files!")
        sys.exit(1)
    temp_lst = []
    for i in range(len(lst)):
        temp_lst.append(os.path.join(img_dir, lst[i]))
    for i in range(len(temp_lst)//2):
        argv = ["", temp_lst[i*2+1], temp_lst[i*2]]
        add_GeoInfo(argv)
        # add_GeoInfo("", temp_lst[i*2+1], temp_lst[i*2])

    
if __name__ == "__main__":
    add_GeoInfo(sys.argv)
