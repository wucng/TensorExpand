参考：https://pjreddie.com/darknet/tiny-darknet/


----------
我听到很多人在谈论[SqueezeNet](https://arxiv.org/abs/1602.07360)。

SqueezeNet很酷，但它只是优化参数计数。 当大多数高质量图像是10MB或更多时，为什么我们关心我们的模型是5MB还是50MB？ 如果你想要一个实际上是FAST的小型模型，为什么不检查[Darknet参考网络](https://pjreddie.com/darknet/imagenet/#reference)？ 它只有28 MB，但更重要的是，它只有8亿个浮点运算。 原来的[Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)是23亿美元。 Darknet速度是它的2.9倍，而且尺寸很小，精确度提高了4％。

那么SqueezeNet呢？ 当然权重只有4.8 MB，但正向传递仍然是22亿次操作。 Alexnet在分类方面是第一次传球，但我们不应该在网络不好的时候停滞不前！

但无论如何，人们超级挤压SqueezeNet，所以如果你真的坚持小网络，使用这个：

# Tiny Darknet

|Model	|Top-1	|Top-5	|Ops	|Size
|:---|:---|:---|:--|:--|
|AlexNet	|57.0	|80.3	|2.27 Bn	|238 MB
|Darknet Reference|	61.1	|83.0	|0.81 Bn |28 MB
|SqueezeNet	|57.5	|80.3	|2.17 Bn	|4.8 MB
|Tiny Darknet	|58.7	|81.7	|0.98 Bn	|4.0 MB

这里真正的赢家显然是 `Darknet reference model` ，但如果你坚持要一个小模型，请使用`Tiny Darknet`。 或者训练你自己的，这应该很容易！

以下是如何在Darknet中使用它（以及如何安装Darknet）：

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
wget https://pjreddie.com/media/files/tiny.weights
./darknet classify cfg/tiny.cfg tiny.weights data/dog.jpg
```
希望你看到这样的东西：

```
data/dog.jpg: Predicted in 0.160994 seconds.
malamute: 0.167168
Eskimo dog: 0.065828
dogsled: 0.063020
standard schnauzer: 0.051153
Siberian husky: 0.037506
```
这里是配置文件：[tiny.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/tiny.cfg)

该模型只是一些3x3和1x1卷积层：

```
layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   224 x 224 x   3   ->   224 x 224 x  16
    1 max          2 x 2 / 2   224 x 224 x  16   ->   112 x 112 x  16
    2 conv     32  3 x 3 / 1   112 x 112 x  16   ->   112 x 112 x  32
    3 max          2 x 2 / 2   112 x 112 x  32   ->    56 x  56 x  32
    4 conv     16  1 x 1 / 1    56 x  56 x  32   ->    56 x  56 x  16
    5 conv    128  3 x 3 / 1    56 x  56 x  16   ->    56 x  56 x 128
    6 conv     16  1 x 1 / 1    56 x  56 x 128   ->    56 x  56 x  16
    7 conv    128  3 x 3 / 1    56 x  56 x  16   ->    56 x  56 x 128
    8 max          2 x 2 / 2    56 x  56 x 128   ->    28 x  28 x 128
    9 conv     32  1 x 1 / 1    28 x  28 x 128   ->    28 x  28 x  32
   10 conv    256  3 x 3 / 1    28 x  28 x  32   ->    28 x  28 x 256
   11 conv     32  1 x 1 / 1    28 x  28 x 256   ->    28 x  28 x  32
   12 conv    256  3 x 3 / 1    28 x  28 x  32   ->    28 x  28 x 256
   13 max          2 x 2 / 2    28 x  28 x 256   ->    14 x  14 x 256
   14 conv     64  1 x 1 / 1    14 x  14 x 256   ->    14 x  14 x  64
   15 conv    512  3 x 3 / 1    14 x  14 x  64   ->    14 x  14 x 512
   16 conv     64  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x  64
   17 conv    512  3 x 3 / 1    14 x  14 x  64   ->    14 x  14 x 512
   18 conv    128  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 128
   19 conv   1000  1 x 1 / 1    14 x  14 x 128   ->    14 x  14 x1000
   20 avg                       14 x  14 x1000   ->  1000
   21 softmax                                        1000
   22 cost                                           1000
```