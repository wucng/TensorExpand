参考：https://pjreddie.com/darknet/yolo/


----------
You only look once（YOLO）是一个最先进的实时对象检测系统。 在Pascal Titan X上，它以30 FPS的速度处理图像，COCO test-dev上的图像分辨率为57.9％。

# 与其他探测器比较
YOLOv3非常快速和准确。 在5.0英寸测量的mAP中，YOLOv3与Focal Loss相当，但速度快了约4倍。 而且，只需更改模型的大小，您就可以轻松地在速度和精度之间进行权衡，无需再培训！

![这里写图片描述](https://pjreddie.com/media/image/map50blue.png)

# COCO数据集上的性能

![这里写图片描述](http://img.blog.csdn.net/20180330091554987?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# How It Works

Prior detection systems将分类器或定位器重新用于执行检测。 他们将模型应用于多个位置和尺度的图像。 图像的高评分区域被视为检测结果。

我们使用完全不同的方法。 我们将一个神经网络应用于整个图像。 该网络将图像划分为区域并预测每个区域的边界框和概率。 这些边界框由预测的概率加权。


我们的模型比classifier-based的系统有几个优点。 它在测试时查看整个图像，以便通过图像中的全局上下文来预测它的预测。 与单一图像需要数千个[R-CNN](https://github.com/rbgirshick/rcnn)等系统不同的是，它也可以通过单一网络评估进行预测。 这使其速度非常快，比R-CNN快1000倍，比[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn)快100倍。 有关完整系统的更多详情，请参阅我们的[论文](https://pjreddie.com/media/files/papers/YOLOv3.pdf)。

# V3中有什么新功能？
YOLOv3使用一些技巧来改善培训并提高性能，包括：多尺度预测，更好的骨干分类器等等。 完整的细节在我们的[论文](https://pjreddie.com/media/files/papers/YOLOv3.pdf)中！

# 使用预训练模型进行检测
本文将引导您使用预先训练的模型通过YOLO系统检测对象。 如果你还没有安装Darknet，[你应该先做](https://pjreddie.com/darknet/install/)。 或者不要阅读所有刚刚运行的内容：

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
您已经在`cfg/`子目录中拥有YOLO的配置文件。 您必须在此下载预先训练的[权重文件（237 MB）](https://pjreddie.com/media/files/yolov3.weights)。 或者只是运行这个：

```
wget https://pjreddie.com/media/files/yolov3.weights
```
然后运行探测器！

```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```
你会看到一些这样的输出：

```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    .......
  105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255  0.353 BFLOPs
  106 detection
truth_thresh: Using default '1.000000'
Loading weights from yolov3.weights...Done!
data/dog.jpg: Predicted in 0.029329 seconds.
dog: 99%
truck: 93%
bicycle: 99%
```
![这里写图片描述](https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.48.42_PM.png)

Darknet会打印出它检测到的物体，信心以及找到它们需要多长时间。 我们没有使用`OpenCV`编译Darknet，因此无法直接显示检测结果。 相反，它将它们保存在`predictions.png`中。 您可以打开它来查看检测到的对象。 由于我们在CPU上使用Darknet，每个图像大约需要6-12秒。 如果我们使用GPU版本，速度会更快。

如果您需要灵感，我已经包含了一些示例图片。 试试`data/eagle.jpg`，`data/dog.jpg`，`data/person.jpg`或`data/horses.jpg`！

detect命令是该命令的更一般版本的缩写。 它相当于命令：

```
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```
如果您只想在一张图像上运行检测，则不需要知道这一点，但知道是否要执行其他操作（如[稍后会看到](https://pjreddie.com/darknet/yolo/#demo)），这很有用。

# 多张图像

不要在命令行上提供图像，您可以将其留空，以便连续尝试多个图像。 相反，当配置和权重完成加载时，您会看到提示：

```
./darknet detect cfg/yolov3.cfg yolov3.weights
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    .......
  104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
  105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255  0.353 BFLOPs
  106 detection
Loading weights from yolov3.weights...Done!
Enter Image Path:
```

输入像`data/horses.jpg`这样的图像路径，让它预测该图像的方框。

![这里写图片描述](https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.53.04_PM.png)

一旦完成，它会提示您输入更多路径来尝试不同的图像。 完成后使用`Ctrl-C`退出程序。

# 更改检测阈值
默认情况下，YOLO仅显示检测到的具有.25或更高置信度的对象。 您可以通过将`-thresh <val>`标志传递给yolo命令来更改它。 例如，要显示所有检测，您可以将阈值设置为0：

```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0
```
所以这显然不是非常有用，但是您可以将其设置为不同的值以控制模型设置的阈值。
# 网络摄像头的实时检测
如果你看不到结果，在测试数据上运行YOLO并不是很有趣。 而不是在一堆图像上运行它，让我们在网络摄像头的输入上运行它！

要运行这个演示，你需要用[CUDA和OpenCV编译Darknet](https://pjreddie.com/darknet/install/#cuda)。 然后运行命令：

```
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
```
YOLO将显示当前的FPS和预测类以及在其上绘制边界框的图像。

您需要连接到OpenCV可以连接到的计算机的网络摄像头，否则它将无法工作。 如果您连接了多个网络摄像头并且想要选择使用哪一个，则可以传递标志`-c <num>`进行选择（默认情况下，OpenCV使用摄像头0）。

如果OpenCV可以读取视频，您还可以在视频文件上运行它：

```
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video file>
```
以上就是我们制作YouTube视频的方式。


# Training YOLO on VOC
如果你想玩不同的训练体制，超参数或数据集，你可以从头开始训练YOLO。 以下是如何使用Pascal VOC数据集的方法。

# Get The Pascal VOC Data
为了训练YOLO，你需要从2007年到2012年的所有VOC数据。你可以在[这里](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)找到数据的链接。 要获取所有数据，请创建一个目录以将其全部存储并从该目录运行：

```
# 数据统一存放到data目录
cd data

wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
现在将有一个`VOCdevkit/`子目录，其中包含所有VOC训练数据。

# Generate Labels for VOC
现在我们需要生成Darknet使用的标签文件。 Darknet需要为每个图像使用一个`.txt`文件，每个图像中的每个地面真实对象都有一行，如下所示：

```
<object-class> <x> <y> <width> <height>
```

其中x，y，width和height与图像的宽度和高度有关。 为了生成这些文件，我们将在Darknet的`scripts/`目录中运行`voc_label.py`脚本。 让我们再次下载，因为我们很懒。

```
wget https://pjreddie.com/media/files/voc_label.py
python voc_label.py
```
几分钟后，该脚本将生成所有必需的文件。 大多数情况下，它会在`VOCdevkit/VOC2007/labels/`和`VOCdevkit/VOC2012/labels/`中生成大量标签文件。 在你的目录中你应该看到：

```
ls data
2007_test.txt   VOCdevkit
2007_train.txt  voc_label.py
2007_val.txt    VOCtest_06-Nov-2007.tar
2012_train.txt  VOCtrainval_06-Nov-2007.tar
2012_val.txt    VOCtrainval_11-May-2012.tar
```
- `VOCdevkit/VOC2007`目录结构

```python
<VOC2007>
|———  Annotations
|       └─  *.xml # 存放图片的 类别与边框
|
|———  JPEGImages
|       └─ *.jpg # 存放图片
|
|———  labels # 运行`voc_label.py` 生成的
|       └─ *.txt # 每个txt文件对应一张图片的所有对象
|
|———  SegmentationClass # 使用不到
|
|——— SegmentationObject # 使用不到
|
|___ ImageSets
```


```python
./darknet/2007_train.txt  
# 存放图片的路径(绝对路径)，每一行对应一张图片的路径，格式：/home/wu/myfloder/darknet/VOCdevkit/VOC2007/JPEGImages/'image_id'.jpg

./darknet/VOCdevkit/VOC2007/labels/'image_id'.txt  
# image_id对应某张图片 必须与上面的image_id对应，该txt存放该图片(只是一张图片)的所有对象信息,格式：<object-class> <x> <y> <width> <height>  其中x,y为对象定位框的中点像素坐标，而不是左上角
```


像`2007_train.txt`这样的文本文件列出了该年的图像文件和图像集。 Darknet需要一个文本文件，其中包含要训练的所有图像。 在这个例子中，让我们训练除2007年测试集之外的所有东西，以便测试我们的模型。 跑：

```
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```
现在我们把2007年的所有训练和2012年的训练都列入了一个大list。 这就是我们必须要做的数据设置！

# 数据更改

```python
# 数据统一存放到data目录
cd data

# Pascal VOC 2007数据下载
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

# 执行以下命令将解压到一个名为VOCdevkit的目录中
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
执行脚本[voc_label.py](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/Object%20detection/darknet/voc_label.py)，该脚本已经被修改以适应于这里的数据

```python
# 查看结果
vim 2007_train.txt
'''
/home/wu/myfolder/darknet/VOCdevkit/VOC2007/JPEGImages/004658.jpg
/home/wu/myfolder/darknet/VOCdevkit/VOC2007/JPEGImages/005076.jpg
/home/wu/myfolder/darknet/VOCdevkit/VOC2007/JPEGImages/007521.jpg
/home/wu/myfolder/darknet/VOCdevkit/VOC2007/JPEGImages/009712.jpg
'''
# --------------------------------
ls VOCdevkit/VOC2007/labels/
'''
000764.txt  001531.txt  002298.txt  
003065.txt  003832.txt  004599.txt  
005366.txt  006133.txt
'''
# --------------------------------
cat VOCdevkit/VOC2007/labels/003835.txt
'''
6 0.463 0.8106666666666666 0.226 0.144
6 0.05 0.7653333333333333 0.076 0.064
5 0.226 0.6773333333333333 0.28400000000000003 0.30933333333333335
5 0.033 0.6933333333333334 0.066 0.16
'''
```

```
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
cp train.txt 2007_test.txt
```


# 修改Pascal数据的Cfg
现在去你的`Darknet`目录。 我们必须更改`cfg/voc.data`配置文件以指向您的数据：

```
classes= 20
train  = <path-to-voc>/train.txt
valid  = <path-to-voc>2007_test.txt
names = data/voc.names
backup = backup
```
您应该将`<path-to-voc>`替换为放置VOC数据的目录。


您还应该修改模型cfg进行训练而不是测试。 `cfg/yolov3-voc.cfg` 应该如下所示：

```python
[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=64
subdivisions=16
width=416
height=416
```
如果是测试 则修改成

```python
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width=416
height=416
```


# 下载预训练的卷积权重
对于训练，我们使用在Imagenet上预先训练的卷积权重。 我们使用[darknet53](https://pjreddie.com/darknet/imagenet/#darknet53)模型的权重。 您可以在这里下载卷积图层的[权重（155 MB）](https://pjreddie.com/media/files/darknet53.conv.74)。

```
wget https://pjreddie.com/media/files/darknet53.conv.74
```


# Train The Model

```python
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74

# or 不使用预训练模型（不做迁移学习或网络微调）
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg

# 如果想接着上次继续训练
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup
```
新的模型参数保存在`backup/`目录

# 模型验证
```
./darknet detector valid cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup
```
# 模型测试
记得修改`cfg/yolov3-voc.cfg`
```python
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width=416
height=416
```


```python
./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup data/dog.jpg

# or
./darknet detect cfg/yolov3-voc.cfg backup/yolov3-voc.backup data/dog.jpg

# or 
./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup
# 再根据提示输入图片路径

# 注，如果GPU内存不足，加上 -nogpu 表示不使用GPU
# or 将每批次训练的图片数缩小点
```


# Training YOLO on COCO
如果你想玩不同的训练体制，超参数或数据集，你可以从头开始训练YOLO。 以下是如何让它在[COCO数据集](http://cocodataset.org/#overview)上工作。

# Get The COCO Data
为了训练YOLO，你需要所有的COCO数据和标签。 脚本`scripts/get_coco_dataset.sh`将为您执行此操作。 找出你想放置COCO数据的地方并下载它，例如：

```
cp scripts/get_coco_dataset.sh data
cd data
bash get_coco_dataset.sh
```
现在您应该拥有为Darknet生成的所有数据和标签。

# Modify cfg for COCO
现在去你的Darknet目录。 我们必须更改`cfg/coco.data`配置文件以指向您的数据：

```
classes= 80
train  = <path-to-coco>/trainvalno5k.txt
valid  = <path-to-coco>/5k.txt
names = data/coco.names
backup = backup
```

您应该将`<path-to-coco>`替换为放置COCO数据的目录。

您还应该修改模型cfg进行训练而不是测试。 `cfg/yolo.cfg`应该如下所示：

```python
[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=64
subdivisions=8
....
```
如果是测试

```python
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=8
....
```



# Train The Model

```
./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74
```
如果你想使用多个gpus运行：

```
./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 0,1,2,3
```
如果您想停止并从检查点重新开始训练：

```
./darknet detector train cfg/coco.data cfg/yolov3.cfg backup/yolov3.backup -gpus 0,1,2,3
```


# 模型验证
```
./darknet detector valid cfg/coco.data cfg/yolov3.cfg backup/yolov3.backup
```
# 模型测试
记得修改`cfg/yolo.cfg`
```python
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=8
....
```

```python
./darknet detector test cfg/coco.data cfg/yolov3.cfg backup/yolov3.backup data/dog.jpg

# or
./darknet detect cfg/yolov3.cfg backup/yolov3.backup data/dog.jpg

# or 
./darknet detector test cfg/coco.data cfg/yolov3.cfg backup/yolov3.backup
# 再根据提示输入图片路径
```


# What Happened to the Old YOLO Site?
如果您使用YOLO版本2，您仍然可以在此处找到该网站：https://pjreddie.com/darknet/yolov2/
# 引用
如果您在工作中使用YOLOv3，请引用我们的论文！

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```