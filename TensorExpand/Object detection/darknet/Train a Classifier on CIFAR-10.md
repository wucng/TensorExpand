参考：https://pjreddie.com/darknet/train-cifar/


----------


这篇文章将教你如何在Darknet中从头开始训练分类器。 我们将使用[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)数据集，这是一个10级的小图像数据集。 让我们开始吧！

# Install Darknet
如果您尚未安装，请执行以下操作：

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
如果一切正常，太棒了！ 如果出现了问题......呃试图修复它？
# Get The Data
我们将使用CIFAR数据的[镜像](https://pjreddie.com/projects/cifar-10-dataset-mirror/)，因为我们需要图像格式的图片。 原始数据集采用二进制格式，但我希望本教程将其推广到您想要处理的任何数据集，因此我们将使用图像来完成。

让我们把数据放在`data/`文件夹中。 要做到这一点：

```
cd data
wget https://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz
```
现在让我们看看我们有什么：

```python
ls cifar
# labels.txt  test  train 
```
使用我们的数据列出两个目录，`train`和`test` ，以及带有标签`labels.txt`的文件。 你可以看看`labels.txt`，如果你想看看我们将学习什么类的类：

```python
cat cifar/labels.txt
'''
airplane # 0
automobile # 1
bird # 2
cat # 3
deer # 4
dog # 5
frog # 6
horse # 7
ship # 8
truck # 9
'''
```

我们还需要生成我们的路径文件。 这些文件将包含所有训练和验证（或在这种情况下为测试）数据的路径。 为此，我们将cd到我们的cifar目录中，找到所有图像，并将它们写入文件，然后返回到我们的基础darknet目录。

```
cd cifar
find `pwd`/train -name \*.png > train.list
find `pwd`/test -name \*.png > test.list
cd ../..
```

# Make A Dataset Config File

```python
cd darknet
mkdir backup
```

我们必须给Darknet一些关于CIFAR-10的元数据。 使用您最喜欢的编辑器，在`cfg/`目录中打开一个名为`cfg/cifar.data`的新文件。 其中你应该有这样的东西：

```
classes=10
train  = data/cifar/train.list
valid  = data/cifar/test.list
labels = data/cifar/labels.txt
backup = backup
top=2
```
- classes = 10：数据集有10个不同的类
- train = ...：在哪里找到训练文件的列表
- valid = ...：在哪里可以找到验证文件的列表
- labels = ...：在哪里可以找到可能的类的列表
- backup= ...：在培训期间保存备份重量文件的位置
- top = 2：在测试时间计算top-n精度（除top-1外）

# Make A Network Config File!
我们需要一个网络来训练。 在你的`cfg`目录下创建另一个名为`cfg/cifar_small.cfg`的文件。 在这个网络中：

```python
[net]
batch=128
subdivisions=1
height=28
width=28
channels=3
max_crop=32 # 28x28-->32x32
min_crop=32

hue=.1 # 色调
saturation=.75 # 饱和度
exposure=.75 # 曝光

learning_rate=0.1
policy=poly # 策略
power=4
max_batches = 5000 # 最大训练步数
momentum=0.9  # 动量
decay=0.0005 # 衰变
# [28,28,3]

[convolutional]
batch_normalize=1 # BN
filters=32 # 32个输出
size=3 # 3x3卷积核
stride=1 # 移动步数1
pad=1 # padding 1
activation=leaky # 激活函数 relu_leaky
# [28,28,32]

[maxpool]
size=2 # 池化核 2x2
stride=2 # 移动步数 2
# [14,14,32]

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
# [14,14,64]

[maxpool]
size=2
stride=2
# [7,7,64]

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky
# [7,7,128]

[convolutional]
filters=10
size=1
stride=1
pad=1
activation=leaky
# [7,7,10]

[avgpool]

[softmax]
groups=1

[cost]
type=sse
```
这是一个非常小的网络，所以它不能很好地工作，但对于这个例子来说这很好。 该网络只有4个卷积层和2个maxpooling层。

最后的卷积层有10个过滤器，因为我们有10个类。 它输出一个`7 x 7 x 10`的图像。 我们只需要总共10个预测，因此我们使用平均联合图层为每个通道获取图像上的平均值。 这会给我们10个预测。 我们使用[softmax](https://en.wikipedia.org/wiki/Softmax_function)将预测转换为概率分布和成本层来计算我们的错误。

# Train The Model
现在我们只需运行训练代码！

```
./darknet classifier train cfg/cifar.data cfg/cifar_small.cfg
```
您只是告诉Darknet您想使用以下数据和网络cfg文件来训练分类器。 对于CPU训练可能需要一个小时或更长时间，即使是这个小型网络。 如果您有GPU，则应按照[这些说明](https://pjreddie.com/darknet/install/#cuda)启用GPU训练。

# Restarting Training

如果你停止训练，你可以随时使用它保存的一个模型检查点重新启动它：

```
./darknet classifier train cfg/cifar.data cfg/cifar_small.cfg backup/cifar_small.backup
```

# valid(验证)

```
./darknet classifier valid cfg/cifar.data cfg/cifar_small.cfg backup/cifar_small.backup
```

# test

先修改`cfg/cifar.data`

```pythn
classes=10
train  = data/cifar/train.list
valid  = data/cifar/test.list
test  = data/cifar/test.list # 加上这句
labels = data/cifar/labels.txt
backup = backup
top=2
```

```
./darknet classifier test cfg/cifar.data cfg/cifar_small.cfg backup/cifar_small.backup
```