# -*- coding:utf-8 -*-

'''
解析自己生成的json文件

已知边界点画边界线和mask
'''
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

json_file = './2.json'
data = json.load(open(json_file))

imagePath=data['imagePath']
shapes=data['shapes']
img=cv2.imread(imagePath,0) # 通过路径打开图片
# 解析shapes里的内容
# 第一个对象
object1_polygon=shapes[0]['points'] # 第一个对象的边界点（多边形的边界点）
object1_label=shapes[0]['label'] # 第一个对象的label

# 第二个对象
object2_polygon=shapes[1]['points']
object2_label=shapes[1]['label']

# 使用opencv在原图上画出区域
# cv2.drawMarker(img,(100,100),255)
# cv2.drawKeypoints(img,object1_polygon)
img=np.zeros_like(img)
cv2.polylines(img,[np.asarray(object1_polygon)],True,1,lineType=cv2.LINE_AA) # 画边界线
cv2.fillPoly(img,[np.asarray(object1_polygon)],1) # 画多边形
cv2.putText(img,object1_label,tuple(object1_polygon[0]),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,255)

cv2.polylines(img,[np.asarray(object2_polygon)],True,2) # 将多边形内像素值设为2
cv2.fillPoly(img,[np.asarray(object2_polygon)],2)
cv2.putText(img,object2_label,tuple(object2_polygon[0]),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,255)
# cv2.fillConvexPoly()

# cv2.imshow('123',img)
# cv2.waitKey(0)

plt.imshow(img,'gray')
plt.show()