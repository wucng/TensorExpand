参考：
1、http://ai.baidu.com/docs#/Face-Python-SDK/top

2、http://blog.csdn.net/u012236875/article/details/74695677

----------
使用百度AI的人脸识别库，做出的调用示例，其中filePath是图片的路径，可以自行传入一张图片，进行识别。
下载baidu-aip这个库，可以直接使用pip下载：<font color=#d000 size=5>pip install baidu-aip</font> 或者进入https://ai.baidu.com/sdk 下载对应的SDK

#代码
```python
# -*- coding: UTF-8 -*-  

from aip import AipFace
import cv2
import matplotlib.pyplot as plt

# 定义常量
APP_ID = '9851066'
API_KEY = 'LUGBatgyRGoerR9FZbV4SQYk'
SECRET_KEY = 'fB2MNz1c2UHLTximFlC4laXPg7CVfyjV'

# 初始化AipFace对象  
aipFace = AipFace(APP_ID, API_KEY, SECRET_KEY)

# 读取图片  
filePath = "messi_2.jpg"


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

        # 定义参数变量


options = {
    'max_face_num': 1, # 图像数量
    'face_fields': "age,beauty,expression,faceshape",
}
# 调用人脸属性检测接口  
result = aipFace.detect(get_file_content(filePath), options)

# print(result)
# print(type(result))

# 解析位置信息
location=result['result'][0]['location']
left_top=(location['left'],location['top'])
right_bottom=(left_top[0]+location['width'],left_top[1]+location['height'])

img=cv2.imread(filePath)
cv2.rectangle(img,left_top,right_bottom,(0,0,255),2)

cv2.imshow('img',img)
cv2.waitKey(0)
# plt.imshow(img,"gray")
# plt.show()
```
<font color=#d000 size=5>注：如果一张图上有多个人脸，只会识别一个人脸

#附加：
关于APP_ID、API_KEY、SECRET_KEY的获取
进入https://console.bce.baidu.com/ai/  ，在该网站创建对应的应用
 
 如：依次进入   产品服务 / 图像识别 - 应用列表 / 创建应用

如：人脸识别，创建应用时选择人脸识别，需要什么就创建对应的应用，创建完成后就能获取到APP_ID、API_KEY、SECRET_KEY

![这里写图片描述](http://img.blog.csdn.net/20171117105736127?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
