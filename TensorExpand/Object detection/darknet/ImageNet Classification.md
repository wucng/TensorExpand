参考：https://pjreddie.com/darknet/imagenet/


----------
您可以使用Darknet将图像分类为1000级[ImageNet](http://image-net.org/challenges/LSVRC/2015/index)挑战。 如果你还没有安装Darknet，你应该先做。

# Classifying With Pre-Trained Models
以下是安装Darknet，下载分类权重文件并在图像上运行分类的命令：

```
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
wget https://pjreddie.com/media/files/extraction.weights
./darknet classifier predict cfg/imagenet1k.data cfg/extraction.cfg extraction.weights data/dog.jpg
```
本示例使用Extraction模型，您可以在[下面](https://pjreddie.com/darknet/imagenet/#extraction)阅读更多关于它的内容。 运行此命令后，您应该看到以下输出：

```
0: Convolutional Layer: 224 x 224 x 3 image, 64 filters -> 112 x 112 x 64 image
1: Maxpool Layer: 112 x 112 x 64 image, 2 size, 2 stride
...
23: Convolutional Layer: 7 x 7 x 512 image, 1024 filters -> 7 x 7 x 1024 image
24: Convolutional Layer: 7 x 7 x 1024 image, 1000 filters -> 7 x 7 x 1000 image
25: Avgpool Layer: 7 x 7 x 1000 image
26: Softmax Layer: 1000 inputs
27: Cost Layer: 1000 inputs
Loading weights from extraction.weights...Done!
298 224
data/dog.jpg: Predicted in 3.756339 seconds.
malamute: 0.194782
Eskimo dog: 0.155007
Siberian husky: 0.143937
dogsled: 0.020943
miniature schnauzer: 0.020566
```
Darknet会在加载配置文件和权重时显示信息，然后对图像进行分类并打印图像的前10类。 Kelp 是一种混合品种的狗，但她有很多爱斯基摩犬，所以我们会认为这是成功的！

您也可以尝试使用其他图像，如秃鹰图像：

```
./darknet classifier predict cfg/imagenet1k.data cfg/extraction.cfg extraction.weights data/eagle.jpg
```
输出：

```
...
data/eagle.jpg: Predicted in 4.036698 seconds.
bald eagle: 0.797689
kite: 0.185116
vulture: 0.006402
prairie chicken: 0.001041
hen: 0.000888
```
如果您未指定图像文件，则会在运行时提示图像。 通过这种方式，您可以在不重新加载整个模型的情况下对多个行进行分类。 使用命令：

```
./darknet classifier predict cfg/imagenet1k.data cfg/extraction.cfg extraction.weights
```
然后你会看到如下提示：

```
....
27: Softmax Layer: 1000 inputs
28: Cost Layer: 1000 inputs
Loading weights from extraction.weights...Done!
Enter Image Path:
```
无论何时您对分类图像感到厌倦，您都可以使用`Ctrl-C`退出程序。

# Validating On ImageNet
您可以看到这些验证集编号随处可见。 也许你想仔细检查一下这些模型的实际工作情况。 我们开始做吧！

首先，您需要下载验证图像和cls-loc注释。 你可以把它们拿到[这里](http://image-net.org/download-images)，但你必须要开一个账户！ 一旦你下载了所有你应该有一个目录`ILSVRC2012_bbox_val_v3.tgz`和`ILSVRC2012_img_val.tar`。 首先我们解开它们：

```
tar -xzf ILSVRC2012_bbox_val_v3.tgz
mkdir -p imgs && tar xf ILSVRC2012_img_val.tar -C imgs
```
现在我们有图像和注释，但我们需要标记图像，以便Darknet可以评估其预测。 我们使用这个bash[脚本](https://github.com/pjreddie/darknet/blob/master/scripts/imagenet_label.sh)来做到这一点。 它已经在您的`scripts/`子目录中。 我们可以再次得到它并运行它：

```
wget https://pjreddie.com/media/files/imagenet_label.sh
bash imagenet_label.sh
```
这将产生两件事：一个名为`labelled/`的目录，其中包含重命名的图像的符号链接，另一个名为`inet.val.list`的文件包含标签图像的路径列表。 我们需要将这个文件移动到Darknet中的`data/`子目录中：

```
mv inet.val.list <path-to>/darknet/data
```
现在你终于准备好验证你的模型了！ 首先重新制作Darknet。 然后像这样运行验证程序：

```
./darknet classifier valid cfg/imagenet1k.data cfg/extraction.cfg extraction.weights
```
注意：如果你没有用[OpenCV](https://pjreddie.com/darknet/install/#opencv)编译Darknet，那么你将无法加载所有的ImageNet图像，因为它们中的一些是`stb_image.h`不支持的奇怪格式。

如果你不用[CUDA](https://pjreddie.com/darknet/install/#cuda)进行编译，你仍然可以在ImageNet上进行验证，但是它会花费很长时间。 不建议。

# Pre-Trained Models
以下是用于ImageNet分类的各种预先训练的模型。 精确度在ImageNet上作为单一作物验证准确度进行衡量。 GPU时序采用英特尔i7-4790K（4 GHz）上的Titan X CPU时序进行测量。

![这里写图片描述](http://img.blog.csdn.net/20180328152303218?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# AlexNet
这模型开始一场革命！ [原始模型](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)对于拆分GPU而言很疯狂，所以这是一些[后续工作的模型](http://arxiv.org/abs/1404.5997)。

- 精度1：精度57.0％
- 前5名准确度：80.3％
- 正向时间：1.5 ms / img
- CPU转发时间：0.3 s / img
- [cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/alexnet.cfg)
- [权重文件（285 MB）](https://pjreddie.com/media/files/alexnet.weights)

# Darknet Reference Model
这个模型被设计成小而强大。 它的性能与AlexNet相同，但其参数只有1/10。 它主要使用卷积层，最后没有大的完全连接层。 它的速度是CPU上的AlexNet的两倍，因此更适合某些视觉应用。

- 前1精度：61.1％
- 前5名准确度：83.0％
- 正向时间：1.5 ms / img
- CPU转发时间：0.16 s / img
- [cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/darknet.cfg)
- [权重文件（28 MB）](http://pjreddie.com/media/files/darknet.weights)

# VGG-16
Oxford的[Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)为ILSVRC-2014比赛开发了VGG-16模型。 它非常准确，广泛用于分类和检测。 我改编了来自[Caffe](http://caffe.berkeleyvision.org/)预训练[模型](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md)的这个版本。 它被训练了6个epochs ，以适应Darknet特定的图像预处理（而不是平均减法，Darknet将图像调整到-1和1之间）。

- Top-1准确度：70.5％
- 前5名准确度：90.0％
- 正向时间：10.7 ms / img
- CPU转发时间：4.9 s / img
- [cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/vgg-16.cfg)
- [权重文件（528 MB）](http://pjreddie.com/media/files/vgg-16.weights)

# Extraction

我将此模型作为[GoogleNet模型](http://arxiv.org/abs/1409.4842)的一个分支进行开发。 它不使用“inception”模块，仅使用1x1和3x3卷积层。

- 前1精度：72.5％
- 前5名准确度：90.8％
- 正向时间：6.4 ms / img
- CPU转发时间：0.95 s / img
- [cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/extraction.cfg)
- [权重文件（90 MB）](http://pjreddie.com/media/files/extraction.weights)

# Darknet19
我修改了Extraction网络，使其更快，更准确。 这个网络是将Darknet Reference网络和Extraction以及诸如Network In Network，Inception和Batch Normalization等众多出版物的想法合并在一起的。

- 前1精确度：72.9％
- 前5名准确率：91.2％
- 正向时间：6.0 ms / img
- CPU正向时序：0.66 s / img
- [cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg)
- [权重文件（80 MB）](https://pjreddie.com/media/files/darknet19.weights)

# Darknet19 448x448
我用更大的输入图像尺寸448x448训练了Darknet19 ，10个更多的epochs 。 由于整个图像较大，该模型性能明显提高但速度较慢。

- 精度1：精度76.4％
- 前5名准确度：93.5％
- 转发时间：11.0 ms / img
- CPU转发时间：2.8 s / img
- [cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/darknet19_448.cfg)
- [权重文件（80 MB）](http://pjreddie.com/media/files/darknet19_448.weights)

# Resnet 50
出于某种原因，即使他们如此懒惰，人们也喜欢这些网络。 随你。 [paper](https://arxiv.org/abs/1512.03385)

- 前1精确度：75.8％
- 前5名准确度：92.9％
- 转发时间：??MS / IMG
- CPU转发时间：??S / IMG
- [cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/resnet50.cfg)
- [权重文件（87 MB）](https://pjreddie.com/media/files/resnet50.weights)

# Resnet 152
For some reason people love these networks even though they are so sloooooow. Whatever. [Paper](https://arxiv.org/abs/1512.03385)

- Top-1 Accuracy: 77.6%
- Top-5 Accuracy: 93.8%
- Forward Timing: ?? ms/img
- CPU Forward Timing: ?? s/img
- [cfg file](https://github.com/pjreddie/darknet/blob/master/cfg/resnet152.cfg)
- [weight file (220 MB)](https://pjreddie.com/media/files/resnet152.weights)

# Densenet 201

我爱DenseNets！ 他们非常深刻，非常疯狂，工作得很好。 就像Resnet一样，由于他们太多层，所以仍然很慢，但至少他们工作得很好！ [Paper](https://arxiv.org/abs/1608.06993)

- Top-1 Accuracy: 77.0%
- Top-5 Accuracy: 93.7%
- Forward Timing: ?? ms/img
- CPU Forward Timing: ?? s/img
- [cfg file](https://github.com/pjreddie/darknet/blob/master/cfg/densenet201.cfg)
- [weight file (67 MB)](https://pjreddie.com/media/files/densenet201.weights)


----------
