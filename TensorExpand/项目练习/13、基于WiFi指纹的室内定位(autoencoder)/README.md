参考：http://blog.topspeedsnail.com/archives/10468

本帖基于论文：[Low-effort place recognition with WiFi fingerprints using Deep Learning](https://arxiv.org/pdf/1611.02049v1.pdf)

室内定位有很多种方式，利用WiFi指纹就是是其中的一种。在室内，可以通过WiFi信号强度来确定移动设备的大致位置，参看：https://www.zhihu.com/question/20593603。

# 使用WiFi指纹定位的简要流程

首先采集WiFi信号，这并不需要什么专业的设备，几台手机即可。Android手机上有很多检测WiFi的App，如Sensor Log。

把室内划分成网格块(对应位置)，站在每个块内分别使用Sensor Log检测WiFi信号，数据越多越好。如下：

```python
location1 : WiFi{"BSSID":"11:11:11:11:11:11",...."level":-33,"....} # 所在位置对应的AP,RSSI信号强度等信息
location2 : WiFi{"BSSID":"11:11:11:11:11:11",...."level":-27,"....}
location2 : WiFi{"BSSID":"22:22:22:22:22:22",...."level":-80,"....}
location3 : WiFi{"BSSID":"22:22:22:22:22:22",...."level":-54,"....}
...
```
无线信号强度是负值，范围一般在0<->-90dbm。值越大信号越强，-50dbm强于-70dbm，

数据采集完成之后，对数据进行预处理，制作成WiFi指纹数据库，参考下面的UJIIndoorLoc数据集。

开发分类模型(本帖关注点)。

最后，用户上传所在位置的wifi信息，分类模型返回预测的位置。

# TensorFlow练习: 基于WiFi指纹的室内定位
使用的数据集：https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc

下载数据集：

```python
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip
$ unzip UJIndoorLoc.zip
```

# 代码分析

```python
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np

test_dataset = pd.read_csv("validationData.csv",header = 0)
test_x = scale(np.asarray(test_dataset.ix[:2,0:520])) # ix 选取数据 或者 iloc
test_y = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))[:5] # map(str) 将所有元素转成字符串
print(test_y) # ['11' '24' '24' '24' '02']
test_y = np.asarray(pd.get_dummies(test_y)) # 生成类似one_hot 编码
print('\n',test_y)
'''
[[0 1 0]  # 11
 [0 0 1]  # 24
 [0 0 1]  # 24
 [0 0 1]  # 24
 [1 0 0]] # 02
'''
```

```python
import pandas as pd
import numpy as np

test_y=[0,3,2,1,4] # test_y=['0','3','2','1','4']
test_y = np.asarray(pd.get_dummies(test_y)) # 生成类似one_hot 编码

print(test_y)
'''
[[1 0 0 0 0]
 [0 0 0 1 0]
 [0 0 1 0 0]
 [0 1 0 0 0]
 [0 0 0 0 1]]
'''
```

# 模型流程
![这里写图片描述](http://img.blog.csdn.net/20180118111148254?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
