参考：

- [thtrieu/darkflow](https://github.com/thtrieu/darkflow)

----------
将darknet翻译为tensorflow。 加载训练后的权重，使用Tensorflow重新训练/微调，将常量graph def导出到移动设备


----------
# 安装

```python
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
python3 setup.py build_ext --inplace

pip3 install -e .
# 或
pip3 install .
```

Android demo on Tensorflow's [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java)


# 准备

- 模型下载

模型下载链接:https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU

或：[yolo.weights](https://pjreddie.com/media/files/yolo.weights)

将下载好的`yolo.weights`放到bin目录

-  训练数据

Pascal VOC 2007训练示例：

```python
# Download the Pascal VOC dataset:
curl -O https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# An example of the Pascal VOC annotation format:
vim VOCdevkit/VOC2007/Annotations/000001.xml

# Train the net on the Pascal dataset:
flow --model cfg/yolo-new.cfg --train --dataset ./VOC2007/JPEGImages --annotation ./VOC2007/Annotations --gpu 1.0
```
- Pascal VOC 2007目录结构

VOC2007只需保留以下结构
```python
<VOC2007>
|———  Annotations
|       └─  *.xml # 存放图片的 类别与边框
|
|———  JPEGImages
|       └─ *.jpg # 存放图片
|
|———  SegmentationClass #（删除）
|
|——— SegmentationObject # （删除）
|
|___ ImageSets #（删除）
```
 
完整下载：

```python
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

# 执行以下命令将解压到一个名为VOCdevkit的目录中
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```


# 只修改classes

如果你没有训练或微调任何东西（你只是想转发训练过的网络）

例如，如果你想只使用3类`tvmonitor`, `person`, `pottedplant`; 编辑`labels.txt`如下

```
tvmonitor
person
pottedplant
```

```
flow --model cfg/yolo-new.cfg --train --dataset ./VOC2007/JPEGImages --annotation ./VOC2007/Annotations --gpu 1.0 --labels labels.txt
```

就是这样。 darkflow会考虑其余的。 您还可以在darkflow使用`--labels`标志加载自定义标签文件（即`--labels myOtherLabelsFile.txt`）。 使用具有不同输出标签集合的多个模型时，这会很有帮助。 如果未设置此标志，默认情况下darkflow将从`labels.txt`加载（如果使用的是为COCO或VOC数据集设计的可识别的.cfg文件，那么自定义的标签文件将被忽略，COCO或VOC标签将会被加载，而自定义的`labels.txt` 会被忽略）。

# 设计网络
如果您正在使用其中一种原始配置，请跳过此步骤，因为它们已经在那里。 否则，请参阅以下示例：

```
...

[convolutional]
batch_normalize = 1
size = 3
stride = 1
pad = 1
activation = leaky

[maxpool]

[connected]
output = 4096
activation = linear

...
```


# `flow`具体使用

```python
# 查看帮助
flow --h
```

首先，让我们仔细研究一个非常有用的选项`--load`

```python
# 1. Load tiny-yolo.weights
flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights

# 2. To completely initialize a model, leave the --load option
flow --model cfg/yolo-new.cfg

# 3. It is useful to reuse the first identical layers of tiny for `yolo-new`
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights
# this will print out which layers are reused, which are initialized
```

来自默认文件夹`sample_img/`的所有输入图像都通过网络流入，并且预测被放入`sample_img/out/`中。 我们可以随时指定更多参数，例如检测阈值，批量大小，图像文件夹等。

```python
# Forward all images in sample_img/ using tiny yolo and 100% GPU usage
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --gpu 1.0
```
可以使用每个边界框的像素位置和像素位置的描述来生成json输出。 每个预测默认存储在`sample_img/out`文件夹中。 下面显示了一个示例json数组。

```python
# Forward all images in sample_img/ using tiny yolo and JSON output.
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --json
```

JSON output:

```
[{"label":"person", "confidence": 0.56, "topleft": {"x": 184, "y": 101}, "bottomright": {"x": 274, "y": 382}},
{"label": "dog", "confidence": 0.32, "topleft": {"x": 71, "y": 263}, "bottomright": {"x": 193, "y": 353}},
{"label": "horse", "confidence": 0.76, "topleft": {"x": 412, "y": 109}, "bottomright": {"x": 592,"y": 337}}]
```
- 标签：自我解释
- 置信度：介于0到1之间（对yolo的信心有多大的自信）
- topleft：框左上角的像素坐标。
- bottomright：框右下角的像素坐标。



# Training new model
训练很简单，因为你只需要添加选项 `--train`。 如果这是第一次训练新配置，则训练集和注释将被解析。 要指向训练集和注释，请使用选项 `--dataset`和 `--annotation`。 几个例子：

```python
# Initialize yolo-new from yolo-tiny, then train the net on 100% GPU:
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights --train --gpu 1.0

# Completely initialize yolo-new and train it with ADAM optimizer
flow --model cfg/yolo-new.cfg --train --trainer adam
```
在训练期间，脚本偶尔会将中间结果保存到Tensorflow checkpoints中，并存储在`ckpt/`中。 要在执行训练/测试之前恢复到任何检查点，请使用`--load [checkpoint_num]`选项，如果`checkpoint_num <0`，则darkflow将通过解析`ckpt/checkpoint`加载最新的保存。

```python
# Resume the most recent checkpoint for training
flow --train --model cfg/yolo-new.cfg --load -1

# Test with checkpoint at step 1500
flow --model cfg/yolo-new.cfg --load 1500

# Fine tuning yolo-tiny from the original one
flow --train --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights
```

# 训练自己的数据
下面的步骤假定我们想要使用tiny YOLO，我们的数据集有3个类

- 创建一个配置文件`tiny-yolo-voc.cfg`的副本并根据您的偏好重新命名它`tiny-yolo-voc-3c.cfg`（保留原始的`tiny-yolo-voc.cfg`文件是至关重要的，请参阅 以下解释）。

- 在`tiny-yolo-voc-3c.cfg`中，将[region]层（最后一层）中的类更改为要训练的类的数量。 在我们的例子中，类被设置为3。

```
...

[region]
anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
bias_match=1
classes=3
coords=4
num=5
softmax=1

...
```
- 在`tiny-yolo-voc-3c.cfg`中，将[convolutional]层（第二层到最后一层）中的滤镜更改为num *（classes + 5）。 在我们的例子中，num是5并且类是3，所以5 *（3 + 5）= 40，因此过滤器被设置为40。

```
...

[convolutional]
size=1
stride=1
pad=1
filters=40
activation=linear

[region]
anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52

...
```
- 更改`labels.txt`以包含要训练的标签（标签数量应与您在`tiny-yolo-voc-3c.cfg`文件中设置的分类数量相同）。 在我们的例子中，`labels.txt`将包含3个标签。

```
label1
label2
label3
```
- 训练时参考`tiny-yolo-voc-3c.cfg`模型。

```
flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images
```
- 为什么我应该保留原始的`tiny-yolo-voc.cfg`文件不变？

当darkflow发现您正在加载`tiny-yolo-voc.weights`时，它会在`cfg/`文件夹中查找`tiny-yolo-voc.cfg`，并将该配置文件与您使用`--model cfg/tiny-yolo-voc-3c.cfg`设置的新配置文件进行比较。 在这种情况下，除了最后两个图层外，每个图层都具有相同的权重数量，所以它会将权重加载到所有图层中，直到后两个图层，因为它们现在包含不同数量的权重。

# 相机/视频文件演示
- 对于完全在CPU上运行的演示：

```
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi
```
- 对于在GPU上运行100％的演示：

```
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi --gpu 1.0
```
要使用摄像头/摄像头，只需使用关键字`camera`替换`videofile.avi`即可。

要使用预测的边界框保存视频，请添加`--saveVideo`选项。

```python
# 检测摄像头版：
python3 flow --model cfg/yolo.cfg --load bin/yolo.weights --demo camera
# 检测视频版：
python3 flow --model cfg/yolo.cfg --load bin/yolo.weights --demo demo.avi
# 使用GPU版：
python3 flow --model cfg/yolo.cfg --load bin/yolo.weights --demo camera --gpu 1.0
```


# 从另一个python应用程序使用darkflow
请注意，`return_predict（img）`必须采用`numpy.ndarray`。 您的图片必须事先加载并传递给`return_predict（img）`。 传递文件路径不起作用。

`return_predict（img）`的结果将是一个字典列表，它表示每个检测到的对象的值与上面列出的JSON输出格式相同。

```python
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)
```
# 将构建的图保存到protobuf文件（.pb）

```python
## Saving the lastest checkpoint to protobuf file
flow --model cfg/yolo-new.cfg --load -1 --savepb

## Saving graph and weights to protobuf file
flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb
```
保存.pb文件时，.meta文件也将随之生成。 这个.meta文件是元字典中所有内容的JSON转储文件，其中包含用于后处理的信息nessecary，例如`anchors `和`labels`。 通过这种方式，您可以根据图表进行预测并进行后期处理，您可以在这两个文件中找到所需的所有内容 - 无需使用`.cfg`或任何标签文件进行标记。

创建的`.pb`文件可用于将图形迁移到移动设备（JAVA / C ++ / Objective-C ++）。 输入张量和输出张量的名称分别是`input`和`output`。 有关此protobuf文件的进一步用法，请参阅C ++ API的Tensorflow[官方文档](https://www.tensorflow.org/versions/r0.9/api_docs/cc/index.html)。 要运行它，比如iOS应用程序，只需将该文件添加到Bundle Resources中，并在源代码内更新此文件的路径。

此外，darkflow支持从`.pb`和`.meta`文件加载以生成预测（而不是从`.cfg`和checkpoint 或`.weights`加载）。

```python
## Forward images in sample_img for predictions based on protobuf file
flow --pbLoad built_graph/yolo.pb --metaLoad built_graph/yolo.meta --imgdir sample_img/
```
如果您想在使用`return_predict()`时加载`.pb`和`.meta`文件，您可以设置“`pbLoad`”和“`metaLoad`”选项来代替您通常设置的`model`和`load`选项。


# 其他

- AssertionError: Cannot capture source

 [opencv安装](https://blog.csdn.net/wc781708249/article/details/78499852) 
```python
# 更新opencv
pip3 install -U opencv-contrib-python
```