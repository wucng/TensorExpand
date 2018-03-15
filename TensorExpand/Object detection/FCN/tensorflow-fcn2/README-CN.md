参考：

[MarvinTeichmann/tensorflow-fcn](https://github.com/MarvinTeichmann/tensorflow-fcn)


----------
# Tensorflow中完全卷积网络的实现


----------
# 更新

有关如何将此代码集成到自己的语义分割管道的示例可以在我的[KittiSeg](https://github.com/MarvinTeichmann/KittiSeg)项目存储库中找到。

# tensorflow-fcn
这是Tensorflow中完全卷积网络的单文件Tensorflow实现。 代码可以很容易地集成到语义分割管道中。 该网络可以直接应用或进行微调以利用张量流训练码进行语义分割。

解卷积层被初始化为双线性上采样。 使用VGG权重的Conv和FCN层权重。 Numpy负载用于读取VGG权重。 不需要Caffe或Caffe-Tensorflow来运行它。 在使用此需求前，[VGG16]的.npy文件将被下载。 你可以在这里找到这个文件：ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy

没有Pascal VOC微调被应用于权重。 该模型旨在根据您自己的数据进行调整。 该模型可以直接应用于图像（见`test_fcn32_vgg.py`），但结果会相当粗糙。

# 要求

除了tensorflow以外，还需要以下软件包：

numpy scipy pillow  matplotlib

这些软件包可以通过运行`pip install -r requirements.txt`或`pip install numpy scipy pillow matplotlib`来安装。


----------
# Tensorflow 1.0rc
此代码需要运行`Tensorflow Version> = 1.0rc`。 如果你想使用旧版本，你可以尝试使用提交`bf9400c6303826e1c25bf09a3b032e51cef57e3b`。 此提交已经使用0.12,0.11和0.10的点对点版本进行了测试。

Tensorflow 1.0带有大量突破性API变化。 如果您当前正在运行较旧的tensorflow版本，我会建议创建一个新的virtualenv并使用以下命令安装1.0rc：

```python
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```
以上命令将安装支持gpu的linux版本。 对于其他版本请按照[这里](https://www.tensorflow.org/versions/r1.0/get_started/os_setup)的说明。


----------
# Usage
`python test_fcn32_vgg.py`来测试实现。

使用它来构建用于微调的VGG对象：

```python
vgg = vgg16.Vgg16()
vgg.build(images, train=True, num_classes=num_classes, random_init_fc8=True)
```
图像是一个形状为[None，h，w，3]的张量。 h和w可以有任意大小。

> Trick：tensor 可以是一个占位符，一个变量，甚至是一个常数。

请注意，`num_classes`会影响`score_fr`（原始fc8图层）的初始化方式。 对于微调，我建议使用选项`random_init_fc8 = True`。


----------
# Training
可以在[KittiSeg](https://github.com/MarvinTeichmann/KittiSeg)项目存储库中找到示例代码。

# Finetuning and training
对于训练使用`vgg.build(images, train=True, num_classes=num_classes)`构建图像是q队列产生图像批次。 在`vgg.up`的输出之上使用`softmax_cross_entropy`损失函数。 损失函数的实现可以在`loss.py`中找到。

为了训练图形，你需要一个输入生产者和一个训练脚本。 看看[TensorVision](https://github.com/TensorVision/TensorVision/blob/9db59e2f23755a17ddbae558f21ae371a07f1a83/tensorvision/train.py)，看看如何建立这些。

我使用Adam Optimizer以1e-6的学习率成功地调整了网络。


----------
# 内容
目前提供以下型号：

- FCN32
- FCN16
- FCN8


----------
# 备注
张量流的去卷积层允许提供形状。 因此原始实施的crop 层不是必需的。

我稍微改变了upscore 层的命名。
## Field of View
所提供模型的接受范围（也称为field of view）是：

```
( ( ( ( ( 7 ) * 2 + 6 ) * 2 + 6 ) * 2 + 6 ) * 2 + 4 ) * 2 + 4 = 404
```
# 前辈
使用[Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow)生成权重。 VGG实现基于[tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16)，numpy加载基于[tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)。 你不需要上面引用的任何代码来运行模型，你不需要caffe。

# 安装
从pip安装matplotlib需要安装以下软件包`libpng-dev`，`libjpeg8-dev`，`libfreetype6-dev`和`pkg-config`。 在Debian，Linux Mint和Ubuntu系统上键入：

```python
sudo apt-get install libpng-dev libjpeg8-dev libfreetype6-dev pkg-config 
pip install -r requirements.txt
```
# TODO

- 提供微调的FCN权重。
- 提供一般训练代码


----------


# 实践
## 1、git项目

```python
git clone https://github.com/MarvinTeichmann/tensorflow-fcn.git
```

# 2、数据显示
## 1、image 

shape [224,224,3] 像素值0~255 

![这里写图片描述](http://img.blog.csdn.net/20180315111720695?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 2、mask 

shape[224,224] 像素值0、1、2、3 （加上背景 共4类）

0为背景 1为矩形 2为圆 3为三角形

![这里写图片描述](http://img.blog.csdn.net/20180315111946039?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


