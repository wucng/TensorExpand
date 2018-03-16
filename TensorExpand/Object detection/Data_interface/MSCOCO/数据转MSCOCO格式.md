参考：

- [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
- [COCO数据库](http://blog.csdn.net/happyhorizion/article/details/77894205)


----------
微软发布的COCO数据库, 除了图片以外还提供物体检测, 分割(segmentation)和对图像的语义文本描述信息. 

MS COCO API 官网- http://cocodataset.org/ 

Github网址 - [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)

[philferriere/cocoapi](https://github.com/philferriere/cocoapi)- support Windows build and python3


----------
# 安装pycocotools

- NOTE: pycocotools requires Visual C++ 2015 Build Tools（for Windows）
- download here if needed http://landinghub.visualstudio.com/visual-cpp-build-tools

- clone this repo

```
git clone https://github.com/philferriere/cocoapi.git
```
- use pip to install pycocotools

```python
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

----------
# 下载MS COCO数据
[参考简单的MS COCO数据集下载方法](http://blog.csdn.net/qq_33000225/article/details/78831102)

```python
sudo apt-get install aria2

aria2c -c http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/val2014.zip 
```
# 分析COCO数据特点
参考 [Dataset - COCO Dataset 数据特点](http://blog.csdn.net/zziahgf/article/details/72819043)

[COCO数据集annotation内容](http://blog.csdn.net/qq_30401249/article/details/72636414)

#### instances_train-val2014/annotations/instances_train2014.json



