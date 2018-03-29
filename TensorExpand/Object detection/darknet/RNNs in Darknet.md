参考：https://pjreddie.com/darknet/rnns-in-darknet/


----------
递归神经网络是表示随时间变化的数据的强大模型。 对于RNN的一个很好的介绍，我强烈推荐去年的Andrej Karpathy的[博客文章](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)，这是一个很好的资源，同时实施它们！

所有这些型号都使用相同的网络架构，一个带3个重复模块的vanilla RNN。

![这里写图片描述](http://img.blog.csdn.net/20180328102929047?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

每个模块由3个fully-connected 层组成。 输入层将信息从输入传播到当前状态。 经常性层通过时间从先前状态传播信息到当前状态。 既然我们希望输入层和复发层都影响当前状态，我们将它们的输出**相加**以得到当前状态。 最后，输出层将当前状态映射到该RNN模块的输出。

网络输入是ASCII字符的[1-hot encoding](https://en.wikipedia.org/wiki/One-hot)。 我们训练网络以预测角色流中的下一个角色。 输出被限制为使用softmax层的概率分布。

由于每个经常性图层都包含有关当前字符和过去字符的信息，因此它可以使用此上下文来预测单词或短语中的未来字符。 随着时间的推移展开的培训看起来像这样：

![这里写图片描述](http://img.blog.csdn.net/20180328103227827?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

尽管一次只能预测一个角色，但这些网络可能非常强大。 在测试时间，我们可以评估给定句子的可能性，或者我们可以让网络自己生成文本！

要生成文本，首先我们通过输入一些字符（如换行符`\n`）或一组字符（如句子）来播种网络。 然后我们将网络的输出作为最后一个字符并将其反馈回网络作为输入。 由于网络的输出是下一个字符的概率分布，我们可以从给定的分布中选取最可能的字符或样本，但采样往往会产生更有趣的结果。

# 用Darknet生成文本
首先你应该[安装Darknet](https://pjreddie.com/darknet/install/)。 由于您不需要CUDA或OpenCV，因此这与克隆GitHub存储库一样简单：

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
一旦你选择了你想使用的权重文件，你可以使用以下命令生成文本：

```
./darknet rnn generate cfg/rnn.cfg <weights>
```
您还可以将各种标志传递给此命令：

- len <int>：更改生成的文本的长度，默认为1,000
- seed <string>：使用给定的字符串对RNN进行种子处理，默认为“\n”
- srand <int>：种子随机数发生器，用于重复运行
- temp <float>：设置采样温度，默认为0.7

足够的闲聊，让我们冒充一些人！

# George R.R. Martin
一些大的破坏者在这里！ 例如，Jon不是一个“紫色女孩”，所以我认为这很好？

要生成这个文本，你必须下载这个权重文件：[grrm.weights](https://pjreddie.com/media/files/grrm.weights)（36 MB）。 然后运行这个命令：

```
./darknet rnn generate cfg/rnn.cfg grrm.weights -srand 0 -seed JON
```
你可以改变srand或seed 来生成不同的文本，所以去野外！ 我真的希望我不会因此被起诉......

OS X上的随机数生成器与Linux中的不同，因此如果运行相同的命令，则会得到不同的输出：

# William Shakespeare

权重文件：[shakespeare.weights](https://pjreddie.com/media/files/shakespeare.weights)

要生成这个文本运行：

```
./darknet rnn generate cfg/rnn.cfg shakespeare.weights -srand 0
```
本示例使用换行符（“\n”）的默认种子。 更改文本seed 或随机数seed 将更改生成的文本。

这个模型在莎士比亚的全部作品中进行了一个单一文件的训练：[shakespeare.txt](https://pjreddie.com/admin/core/file/31/)，最初来自[Project Gutenberg](http://www.gutenberg.org/ebooks/100)。

# Leo Tolstoy
权重文件：[tolstoy.weights](https://pjreddie.com/media/files/tolstoy.weights)

要生成这个文本运行：
```
./darknet rnn generate cfg/rnn.cfg tolstoy.weights -srand 0 -seed Chapter
```

# Immanuel Kant
权重文件：[kant.weights](https://pjreddie.com/media/files/kant.weights)

要生成上述文本，请运行：

```
./darknet rnn generate cfg/rnn.cfg kant.weights -srand 0 -seed Thus -temp .8
```

# Slack
我不打算发布这个模型，但你可以下载你自己的Slack日志并在它们上面训练一个模型！ 你怎么问？ 阅读....

# Train Your Own Model
您还可以在新的文本数据上训练自己的模型！ 培训配置文件是`cfg/rnn.train.cfg`。 所有你需要训练的是一个文本文件，其中所有的数据都是ASCII码。 然后你运行下面的命令：

```
./darknet rnn train cfg/rnn.train.cfg -file data.txt
```
该模型会将定期备份保存到函数`train_char_rnn`中`src/rnn.c`中指定的目录中，您可能希望将此目录更改为机器的合适位置。 要从备份重新开始训练，您可以运行：

```
./darknet rnn train cfg/rnn.train.cfg backup/rnn.train.backup -file data.txt
```
如果你想在大量数据上训练大型模型，你可能需要在快速GPU上运行它。 你可以在CPU上训练它，但可能需要一段时间，你已经被警告！