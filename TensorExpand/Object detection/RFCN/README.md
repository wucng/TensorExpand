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

注意：如果您的系统上有多个python版本，并且您想使用与“python”不同的版本，请在调用make之前提供一个名为PYTHON的环境变量。例如：**python3编译**`export PYTHON=python3 & make`

尝试加载`.so`文件时，您可能会遇到未定义的符号问题。如果您自己构建了TensorFlow版本并且Makefile无法自动检测您的ABI版本，则会出现这种情况。日志中可能会遇到类似`“tensorflow.python.framework.errors_impl.NotFoundError：BoxEngine / ROIPooling / roi_pooling.so：undefined symbol：_ZN10tensorflow7strings6StrCatB5cxx11ERKNS0_8AlphaNumE”`的错误。在这种情况下，清理项目（清理）并使用`USE_OLD_EABI = 0`标志（`export USE_OLD_EABI = 0 & make`）重建它。

您可能想要在没有GPU支持的情况下构建ROI池。使用`USE_GPU = 0`标志关闭代码的CUDA部分。

您可能需要运行以下命令来安装python依赖项：

```
pip3 install --user -r packages.txt
```
# Testing
您可以使用`test.py`运行训练有素的模型。 模型路径应该没有文件扩展名（没有.data *和.index）。 一个例子：

![这里写图片描述](https://cloud.githubusercontent.com/assets/2706617/25061919/2003e832-21c1-11e7-9397-14224d39dbe9.jpg)

# Pretrained model
你可以从这里下载一个预训练模型：
http://xdever.engineerjs.com/rfcn-tensorflow-export.tar.bz2

将它解压到你的项目目录。 然后您可以使用以下命令运行网络：

```python
python3 test.py -n export/model -i <input image> -o <output image>
```
注意：这个预训练模型没有以任何方式进行超参数优化。 该模型可以（并且）在优化时具有更好的性能。 尝试不同的学习率和分类以回归损失余额。 最佳值与测试高度相关。

# Training the network
为了训练网络，您首先需要下载MS COCO数据集。 下载所需的文件并将其解压缩到具有以下结构的目录中：

```python
<COCO>
├─  annotations
│    ├─  instances_train2014.json
│    └─  ...
|
├─  train2014
└─  ...
```
运行这个命名： `python3 main.py -dataset <COCO> -name <savedir>`
- <COCO> - coco根目录的完整路径
- <savedir> - 保存文件的路径。 该目录及其子目录将自动创建。

<savedir>将具有以下结构：

```python
<savedir>
├─  preview
│    └─  preview.jpg - preview snapshots from training process.
|
├─  save - TensorFlow checkpoint directory
│    ├─  checkpoint
│    ├─  model_*.*
│    └─  ...
└─  args.json - saved command line arguments.
```
您可以随时停止训练过程并稍后恢复，只需运行`python3 main.py -name <savedir>`，而无需任何其他参数。 所有命令行参数将自动保存并重新加载。
# License
该软件在Apache 2.0许可下。 有关更多详细信息，请参阅http://www.apache.org/licenses/LICENSE-2.0。

# Notes
此代码要求TensorFlow> = 1.0（最后一个已知工作版本是1.4.1）。 使用python3.6进行测试，编译它应该可以使用python 2。

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

# 改写CocoDataset.py




# 附加
## 简单的MS COCO数据集下载方法
http://blog.csdn.net/qq_33000225/article/details/78831102

```python
sudo apt-get install aria2

aria2c -c http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/val2014.zip 
```
## cocodataset/cocoapi
https://github.com/cocodataset/cocoapi

## COCO数据库
http://blog.csdn.net/happyhorizion/article/details/77894205



