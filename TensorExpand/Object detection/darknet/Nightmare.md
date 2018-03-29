
参考：https://pjreddie.com/darknet/nightmare/


----------
曾几何时，在一座大学建筑中，几乎与现在坐的不同，但不完全不同，西蒙扬，维达尔迪和齐瑟曼有一个[好主意](https://arxiv.org/pdf/1312.6034v2.pdf)。 他们认为，嘿，我们一直在向前运行这些神经网络，并且它们工作得很好，为什么不向后运行呢？ 这样我们就可以弄清楚电脑实际上在想什么......

由此产生的图像是如此可怕，如此怪诞，以至于他们可以一直听到Templeton的尖叫声。

![这里写图片描述](https://pjreddie.com/media/image/scream_vgg-conv_10_000002.png)

[许多](https://arxiv.org/pdf/1412.0035v1.pdf) [研究](https://arxiv.org/pdf/1412.1897v4.pdf) [人员](https://arxiv.org/pdf/1506.02753.pdf)已经扩展了他们的工作，其中包括谷歌公开的[博客文章](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html)。

这是我抄袭那些抄袭那些有好主意的其他人的人。

# Have A Nightmare With Darknet
如果你没有安装Darknet，那就先做！ 要有我们的噩梦，我们将使用VGG-16预训练模型。 但是，我们不需要整个模型，只需要卷积图层，所以我们可以使用`vgg-conv.cfg`文件（您应该已经在`cfg/`子目录中）。 您需要在此下载预训练的[权重](https://pjreddie.com/media/files/vgg-conv.weights)（57 MB）。

现在我们可以生成您在第一段中看到的尖叫图像：

```
./darknet nightmare cfg/vgg-conv.cfg vgg-conv.weights data/scream.jpg 10
```
该命令如下分解：首先我们有可执行程序和子程序，`./darknet nightmare`，随后是配置文件和权重文件`cfg/vgg-conv.cfg`， `vgg-conv.weights`。 最后，我们有我们想要修改的图像和我们想要定位的配置文件的图层，`data/scream.jpg` ，`10`。

这可能需要一段时间，特别是如果您仅使用CPU。 在我的机器上大约需要15分钟。 我强烈建议让CUDA更快地产生恶梦。 启用CUDA后，Titan X需要大约7秒钟的时间。

您可以尝试较低层以获得更具艺术感觉：

```
./darknet nightmare cfg/vgg-conv.cfg vgg-conv.weights data/dog.jpg 7
```
![这里写图片描述](https://pjreddie.com/media/image/dog_vgg-conv_6_000000.png)

或者用更高层次来获得更复杂的紧急行为：

```
./darknet nightmare cfg/vgg-conv.cfg vgg-conv.weights data/eagle.jpg 13
```
![这里写图片描述](https://pjreddie.com/media/image/eagle_vgg-conv_13_000000.png)


# 特殊选项
你可能会注意到你生成的尖叫声跟我的看起来不太一样。 那是因为我使用了一些特殊的选项！ 我使用的实际命令是：

```
./darknet nightmare cfg/vgg-conv.cfg vgg-conv.weights \
data/scream.jpg 10 -range 3 -iters 20 -rate .01 -rounds 4
```

Darknet在连续的循环中产生图像，其中前一轮的输出馈送到下一轮。每轮的图像也写入磁盘。每一轮都由一些迭代组成。在每次迭代中，Darknet会修改图像以在某个比例下增强所选图层。音阶是从一组八度音程中随机选择的。该层从一系列可能的层中随机选择。修改此过程的命令是：

- rounds n：更改轮次数（默认1）。更多回合意味着生成更多的图像，并且通常会更改原始图像。
- iters n：改变每轮的迭代次数（默认为10）。更多的迭代意味着每轮更多的图像变化。
- range n：改变可能的图层范围（默认1）。如果设置为1，则每次迭代时只选择给定的图层。否则，在范围内随机选择一层（例如，10-区3将在层9-11之间选择）。
- octaves n：更改可能的比例数（默认值4）。在一个八度处，只检查全尺寸图像。每增加一个八度音，都会增加一个较小版本的图像（前一个八度的3/4）。
- rate x：更改图像的学习速率（默认为.05）。越高意味着每次迭代对图像的更多变化，但也有一些不稳定性和不精确性。
- thresh x：更改要放大的要素的阈值（默认值为1.0）。只有距平均值超过x个标准偏差的特征才会在目标图层中放大。较高的阈值意味着较少的特征被放大。
- zoom x：每轮后更改应用于图像的缩放比例（默认为1.0）。您可以选择在每一轮之后添加（x <1）或缩小（x> 1）的缩放以应用于图像。
- rotate x：更改每轮后应用的旋转（默认为0.0）。每轮后可选择轮换。

这里有很多玩法！ 这里有一个例子，有多个回合和一个稳定的放大：

# A Smaller Model
VGG-16是一个非常大的型号，如果您的内存不足，请尝试使用此型号！

cfg文件位于`cfg/`子目录（或[此处](https://github.com/pjreddie/darknet/blob/master/cfg/jnet-conv.cfg)）中，您可以在此下载[权重](http://pjreddie.com/media/files/jnet-conv.weights)（72 MB）。

```
./darknet nightmare cfg/jnet-conv.cfg jnet-conv.weights \
data/yo.jpg 11 -rounds 4 -range 3
```
![这里写图片描述](https://pjreddie.com/media/image/yo_jnet-conv_11_000003.png)

# 与Deep Dream和GoogleNet比较

这些示例使用VGG-16网络。 尽管GoogleNet似乎注[重狗和slugs](http://i.imgur.com/ebk1Cdc.jpg)，但VGG喜欢制造獾，一种啮齿动物和一只猴子之间的奇怪交叉：

![这里写图片描述](https://pjreddie.com/media/image/badgermole.png)

VGG也没有GoogleNet的本地响应规范化层。 因此，它的噩梦往往会因色彩星爆而过度饱和。

![这里写图片描述](https://pjreddie.com/media/image/vid_00520.png)

