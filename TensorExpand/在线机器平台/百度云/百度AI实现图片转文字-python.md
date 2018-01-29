参考：
1、http://ai.baidu.com/docs#/OCR-Python-SDK/top
2、http://blog.csdn.net/wc781708249/article/details/78558860


----------
1、安装百度AI ： pip install baidu-aip
2、到https://console.bce.baidu.com/ai/创建文字识别应用，获取APP_ID、API_KEY、SECRET_KEY


----------
#代码

```
# -*- coding: UTF-8 -*-  

from aip import AipOcr


# 定义常量  
APP_ID = '10379743'
API_KEY = 'QGGvDG2yYiVFvujo6rlX4SvD'
SECRET_KEY = 'PcEAUvFO0z0TyiCdhwrbG97iVBdyb3Pk'

# 初始化文字识别分类器
aipOcr=AipOcr(APP_ID, API_KEY, SECRET_KEY)

# 读取图片  
filePath = "wenzi.png"

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 定义参数变量
options = {
    'detect_direction': 'true',
    'language_type': 'CHN_ENG',
}

# 网络图片文字文字识别接口
result = aipOcr.webImage(get_file_content(filePath),options)

# 如果图片是url 调用示例如下
# result = apiOcr.webImage('http://www.xxxxxx.com/img.jpg')

print(result)
```
#图片
![这里写图片描述](http://img.blog.csdn.net/20171117113726754?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#结果

```
{'log_id': 8544045531109655035, 'direction': 0, 'words_result_num': 4, 'words_result': [{'words': '【摘要】为了提高图像匹配的精确度,提出一种基于SIFT算法与 RANSAC算法相结合'}, {'words': '的方法对X射线图像进行匹配。通过最近邻次近邻比值法对特征点进行粗匹配,利用对极几'}, {'words': '何约束的 RANSAC算法剔除误匹配点对,从而实现精确匹配。实验结果表明了该方法的准确性'}, {'words': '和有效性。'}]}
```