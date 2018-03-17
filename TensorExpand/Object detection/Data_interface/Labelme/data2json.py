# -*- coding:utf-8 -*-

'''
仿照labelme的json文件写入自己的数据
'''
import cv2
import json

# json_file = './1.json'

# data = json.load(open(json_file))

# 参考labelme的json格式重新生成json文件，
# 便可以使用labelme的接口解析数据

def dict_json(imageData,shapes,imagePath,fillColor=None,lineColor=None):
    '''

    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"imageData":imageData,"shapes":shapes,"fillColor":fillColor,
            'imagePath':imagePath,'lineColor':lineColor}

def dict_shapes(points,label,fill_color=None,line_color=None):
    return {'points':points,'fill_color':fill_color,'label':label,'line_color':line_color}

# 注以下都是虚拟数据，仅为了说明问题
imageData="image data"
shapes=[]
# 第一个对象
points=[[10,10],[120,10],[120,120],[10,120]] # 数据模拟
# fill_color=null
label='cat_1'
# line_color=null
shapes.append(dict_shapes(points,label))

# 第二个对象
points=[[150,200],[200,200],[200,250],[150,250]] # 数据模拟
label='cat_2'
shapes.append(dict_shapes(points,label))

fillColor=[255,0,0,128]

imagePath='E:/practice/1.jpg'

lineColor=[0,255,0,128]

data=dict_json(imageData,shapes,imagePath,fillColor,lineColor)

# 写入json文件
json_file = 'E:/practice/2.json'
json.dump(data,open(json_file,'w'))