参考：
1、http://ai.baidu.com/docs#/ImageClassify-Python-SDK/top

2、http://blog.csdn.net/wc781708249/article/details/78558860

----------
#代码

```
# -*- coding: UTF-8 -*-  

from aip import AipImageClassify
# import cv2
# import matplotlib.pyplot as plt

# 定义常量  
APP_ID = '10376628'
API_KEY = 'nf66Tjv0NnbkzcodwqMzUuTp'
SECRET_KEY = 'fYv9H15VrwGnGerhGyxgOo4cOgAw2ypy'

# 初始化图像分类器  
imgClass=AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

# 读取图片  
filePath = "1.jpg"

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 定义参数变量
options = {}
options["top_num"] = 5 # 输出前5个可能性预测

# 调用动物分类器
result=imgClass.animalDetect(get_file_content(filePath),options)

# 调用汽车分类器
result=imgClass.carDetect(get_file_content(filePath),options)

print(result)
```
#输出结果：

```
{'log_id': 3322611835559934374, 'result': [{'score': '0.2834', 'name': '异国短毛猫'}, {'score': '0.159157', 'name': '美国硬毛猫'}, {'score': '0.109393', 'name': '美国短毛猫'}, {'score': '0.0743276', 'name': '欧洲短毛猫'}, {'score': '0.0557966', 'name': '马恩岛猫'}]}
```
#附加
 APP_ID、API_KEY、SECRET_KEY的获取
 参考：http://blog.csdn.net/wc781708249/article/details/78558860

