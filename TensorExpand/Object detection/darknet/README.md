参考：

- [yolo](https://pjreddie.com/darknet/yolo/)
- [darknet](https://pjreddie.com/darknet/)


----------

# Installing Darknet

Darknet易于安装，只有两个可选依赖项：

- [OpenCV](https://opencv.org/)如果你想要更多种类的支持图像类型。
- [CUDA](https://developer.nvidia.com/cuda-downloads)如果你想GPU计算。

两者都是可选的，所以我们从安装基本系统开始。 我只在Linux和Mac电脑上测试过。 如果它不适合你，给我发电子邮件或其他东西？

# Installing The Base System
首先在[这里](https://github.com/pjreddie/darknet)克隆Darknet git仓库。 这可以通过以下方式完成：

```
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```
如果这项工作你应该看到一大堆编译信息飞过：

```
mkdir -p obj
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
.....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast -lm....
```
如果您有任何错误，请尝试修复它们？ 如果一切似乎编译正确，请尝试运行它！

```
./darknet
```
你应该得到输出：`usage: ./darknet <function>`

现在查看你可以用darknet在[这里](https://pjreddie.com/darknet/)做的很酷的事情。

# Compiling With CUDA

CPU上的Darknet速度很快，但它在GPU上快500倍！ 你必须有一个[Nvidia GPU](https://developer.nvidia.com/cuda-gpus)，你必须安装[CUDA](https://developer.nvidia.com/cuda-downloads)。 我不会详细介绍CUDA安装，因为它很可怕。

一旦安装了CUDA，将基本目录中的`Makefile`的第一行改为：

```
GPU=1
```
现在您可以使项目和CUDA启用。 默认情况下，它将在系统中的第0个图形卡上运行网络（如果您正确安装了CUDA，则可以使用`nvidia-smi`列出您的图形卡）。 如果你想改变Darknet使用什么卡，你可以给它一个可选的命令行标志`-i <index>`，例如：

```
./darknet -i 1 imagenet test cfg/alexnet.cfg alexnet.weights
```

如果您使用CUDA进行编译，但是无论出于何种原因都想要执行CPU计算，则可以使用`-nogpu`来使用CPU：

```
./darknet -nogpu imagenet test cfg/alexnet.cfg alexnet.weights
```
享受你的新的，超快的神经网络！

# Compiling With OpenCV

默认情况下，Darknet使用[stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)来加载图像。 如果你想要更多奇怪格式的支持（比如CMYK jpegs，感谢奥巴马），你可以改用[OpenCV](https://opencv.org/)！ OpenCV还允许您查看图像和检测，而不必将它们保存到磁盘。

首先安装OpenCV。 如果你是从源代码做到这一点，它将会很长很复杂，所以请试着让包管理器为你做。

接下来，将`Makefile`的第二行更改为：

```
OPENCV=1
```
你完成了！ 要尝试一下，首先重新`make`该项目。 然后使用`imtest`例程来测试图像加载和显示：

```
./darknet imtest data/eagle.jpg
```
如果你在他们身上得到一堆带鹰的窗户，你就成功了！ 他们可能看起来像：

![这里写图片描述](https://pjreddie.com/media/image/Screen_Shot_2015-06-10_at_2.47.08_PM.png)

