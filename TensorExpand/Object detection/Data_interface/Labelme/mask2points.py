# -*- coding:utf-8 -*-

'''
从现有的mask提取出对象的边界点和边界线
'''
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
# from labelme import utils

# json_file = './1.json'
# data = json.load(open(json_file))
# img = utils.img_b64_to_array(data['imageData'])  # 解析原图片数据
# lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

# 虚假数据演示mask
img=np.zeros([200,200],np.uint8)
# pts=[[10,10],[120,10],[120,120],[10,120]]

pts=[[50,90],[80,4],[100,80],[160,30],[170,40],[130,180],[100,150],[16,170],[10,65]]

cv2.fillPoly(img,[np.asarray(pts)],1) # img 变成0、1二值图
# cv2.imshow("img", img)
# cv2.waitKey(0)
# mask=[]
# mask.append((lbl==1).astype(np.uint8)) # 解析第一个对象的mask

# 现在从mask中解析出边界点和边界线
# '''
# 方法一 使用opencv 获取边界
img2=np.zeros_like(img,np.uint8)
# 需要注意的是cv2.findContours()函数接受的参数为二值图
contours=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # 找到轮廓线
cv2.drawContours(img2,contours[1],-1,255,3) # contours[1] 返回的就是边界点

# cv2.imshow("img", img)
# cv2.waitKey(0)
'''

# 方法二 使用像素错位相减,提取边界线，边界点还的借助opencv
img2=np.zeros_like(img,np.uint8)
img5=np.zeros_like(img,np.uint8)
img3=abs(img[:,:-1].astype(np.int16)-img[:,1:].astype(np.int16)) # 因为相减后会出现负数，unit8不支持负数
img4=abs(img[:-1,:].astype(np.int16)-img[1:,:].astype(np.int16)) # 因为相减后会出现负数，unit8不支持负数

img2[:,:-1]=img3 # 相减后会少一列，必须补上
img5[:-1,:]=img4
img2+=img5  #

# dst = cv2.cornerHarris(img2.astype(np.float32),2,3,0.04)
contours=cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # 找到轮廓线
img2=np.zeros_like(img,np.uint8)
cv2.drawContours(img2,contours[1],-1,255,3) # contours[1] 返回的就是边界点

# '''

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(img2)

plt.show()