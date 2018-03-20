参考：

- [xdever/RFCN-tensorflow](https://github.com/xdever/RFCN-tensorflow)
- [PureDiors/pytorch_RFCN](https://github.com/PureDiors/pytorch_RFCN)
- [daijifeng001/caffe-rfcn](https://github.com/daijifeng001/caffe-rfcn)


----------

# TensorFlow implementation of RFCN

效仿R-CNN，采用流行的物体检测策略，包括区域建议和区域分类两步。用Faster R-CNN中的区域建议网络(RPN)提取候选区域，该RPN为全卷积网络。效仿Faster R-CNN，同享RPN和R-FCN的特点。


----------
论文可在https://arxiv.org/abs/1605.06409上找到。

# Building
ROI汇集和MS COCO加载器需要首先编译。为此，请在项目的根目录中运行`make`。如果您需要特殊的链接器/编译器选项，则可能需要编辑`BoxEngine/ROIPooling/Makefile`。

注意：如果您的系统上有多个python版本，并且您想使用与“python”不同的版本，请在调用make之前提供一个名为PYTHON的环境变量。例如：

```python
# python3编译
vim /etc/profile
# 加入以下语句
export PYTHON=python3

source /etc/profile # 这句一定要执行
# 再运行
make
```

# 训练自己的数据
参考[问题#7](https://github.com/xdever/RFCN-tensorflow/issues/7)

要在此repo上训练您自己的数据集：

- 训练阶段：
[1]用你自己的脚本替换`CocoDataset.py`，读取图像，加载坐标和类标签/索引
[2][在此](https://github.com/xdever/RFCN-tensorflow/blob/master/Dataset/BoxLoader.py#L48)更新类别的总数（Dataset/BoxLoader.py 48行位置修改）
[3]更新[1]中的main.py，导入脚本

- 推理阶段
[1]在test.py中，更新类别定义（更改categories）

##**或者**

- 【1】你可以制作一个脚本，将你的数据重新格式化为**coco format**，它只是按照[这种格式](http://cocodataset.org/#download)解析json文件中的值（4.1节）
- 【2】[在此](https://github.com/xdever/RFCN-tensorflow/blob/master/Dataset/BoxLoader.py#L48)更新类别的总数（Dataset/BoxLoader.py 48行位置修改）
- 【3】不需要按[1]更新main.py


----------
# Pascal VOC 2007数据下载
共20个类别
```python
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

```python
# 执行以下命令将解压到一个名为VOCdevkit的目录中
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
## 数据预览
1、VOC2007/Annotations

类别名与对象的矩形框位置

2、VOC2007/JPEGImages

![这里写图片描述](http://img.blog.csdn.net/20180312111000760?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

3、VOC2007/SegmentationClass

![这里写图片描述](http://img.blog.csdn.net/20180312111028031?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

4、VOC2007/SegmentationObject

![这里写图片描述](http://img.blog.csdn.net/20180312111039831?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


# train stage
# 将Pascal VOC转成COCO格式
- 参考[Pascal VOC转COCO数据](http://blog.csdn.net/wc781708249/article/details/79615210)  生成一个`new.json` 文件

注：按照SegmentationObject来提取Segmentation，并不是JPEGImages中的图片都有对应的Segmentation

```python
ls SegmentationObject/|wc # 632
ls JPEGImages/|wc # 9963
# JPEGImages中大部分图片没有对应的掩膜，即无法做训练数据
# 需从JPEGImages选出可以做训练的图片 执行以下程序
```

```python
# -*- coding:utf-8 -*-

import json
import shutil,os

json_file='new.json'
data=json.load(open(json_file))

# 图片名
img_name=[]
for img in data['images']:
    img_name.append(img['file_name'])

# mkdir train2014
for img in img_name:
    shutil.copy(os.path.join('./JPEGImages',img),'./train2014')

# print(len(img_name))
```

- 将VOC2007目录下的`JPEGImages` 文件夹复制到`./RFCN-tensorflow/data/Datasets/COCO` 目录下，并重命名为`train2014`(或者 如果执行了上一步，将VOC2007目录下的`train2014`文件夹复制到`./RFCN-tensorflow/data/Datasets/COCO` 目录下 ) ，再将`train2014` 复制一份成`val2014` （只是为了演示，其实两者是不同的）

- 将生成的`new.json` （使用[Pascal VOC转COCO数据](http://blog.csdn.net/wc781708249/article/details/79615210)得到的 ）复制到`./RFCN-tensorflow/data/Datasets/COCO/annotations` 目录，并重命名为`instances_train2014.json`

- 还需要一个`instances_val2014.json` 这里直接把`instances_train2014.json` 复制一份成`instances_val2014.json` 即可（实际上两者是不一样的）

最终 `./RFCN-tensorflow/data/Datasets/COCO` 目录结构如下：
```
<COCO>
├─  annotations
│    ├─  instances_train2014.json
│    └─  instances_val2014.json
|
|─  train2014
|       |--*.jpg
└─  val2014
        |--*.jpg
```

- 先手动下载模型文件
```python
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
# 将文件解压到./RFCN-tensorflow 目录下
tar zxf inception_resnet_v2_2016_08_30.tar.gz
```

- 修改`Dataset/BoxLoader.py` 48行位置

```python
def categoryCount(self):
   return 80 # 改成数据的实际类别数 Pascal VOC2017只有20个类别 所以这里修改成20（不算背景）
```

## 执行命名：
```python
python3 main.py -dataset ./RFCN-tensorflow/data/Datasets/COCO -name save
```


----------


# test stage
- 在test.py中，更新类别定义（更改categories 按照自己的数据中的categories 重新修改）

注：要按照类别ID顺序，依次排列，不可以随意

可以加上以下语句：

```python
# -*- coding:utf-8 -*-

import json

json_file='new.json'
data=json.load(open(json_file))

# 按类别顺序取出
cats={}
for cat in data['categories']:
    cats[cat['id']]=cat['name']

sorted(cats)
categories=list(cats.values())
print(categories)
'''
['diningtable', 'person', 'bottle', 'boat', 'train', 'bird', 'dog', 'cat', 'tvmonitor', 'cow', 'car', 'sofa', 'horse', 'chair', 'pottedplant', 'bicycle', 'motorbike', 'aeroplane', 'sheep', 'bus']
'''
```

打开`test.py` 将categories 换成上面的即可。

- 修改`test.py`82行 

```python
if not CheckpointLoader.loadCheckpoint(sess, None, opt.n, ignoreVarsInFileNotInSess=True):

# 改成
if not CheckpointLoader.loadCheckpoint(sess, opt.n, None,ignoreVarsInFileNotInSess=True):

# 即 None与opt.n对调一下 去Utils找到loadCheckpoint函数，会发现test.py中的调用参数顺序不对
```


## 执行命令

```python
python3 test.py -n save/save/ -i <input image> -o <output image>
```

## 运行结果

![这里写图片描述](http://img.blog.csdn.net/20180320140743102?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


![这里写图片描述](http://img.blog.csdn.net/20180320140755851?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
