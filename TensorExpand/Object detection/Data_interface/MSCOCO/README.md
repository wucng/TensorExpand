参考：

- [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
- [philferriere/cocoapi](https://github.com/philferriere/cocoapi)- support Windows build and python3
- [COCO数据库](http://blog.csdn.net/happyhorizion/article/details/77894205)


----------
微软发布的COCO数据库, 除了图片以外还提供**物体检测**, **分割**(segmentation)和对图像的**语义文本描述信息**，**关键点检测**（肢体检测）. 

![这里写图片描述](http://img.blog.csdn.net/20170908141707784?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaGFwcHlob3Jpemlvbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20170908144238610?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaGFwcHlob3Jpemlvbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

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
- [参考简单的MS COCO数据集下载方法](http://blog.csdn.net/qq_33000225/article/details/78831102)

```python
sudo apt-get install aria2

aria2c -c http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/val2014.zip 
```

## coco数据集下载链接
数据包括了**物体检测**和keypoints**身体关键点**的检测和**图片描述**。

- [MS coco数据集下载](http://blog.csdn.net/daniaokuye/article/details/78699138)

```python
http://images.cocodataset.org/zips/train2017.zip 
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip 
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip 
http://images.cocodataset.org/annotations/image_info_test2017.zip 
```

这些就是全部的microsoft coco数据集2017的链接了。


# 分析COCO数据特点
- [Dataset - COCO Dataset 数据特点](http://blog.csdn.net/zziahgf/article/details/72819043)

- [COCO数据集annotation内容](http://blog.csdn.net/qq_30401249/article/details/72636414)

- [COCO 标注详解](http://blog.csdn.net/yeyang911/article/details/78675942)


# 其他数据集
- [机器学习数据集(Dataset)汇总](http://blog.csdn.net/MyArrow/article/details/51828681)