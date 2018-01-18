参考：
https://github.com/luyishisi/tensorflow

http://blog.topspeedsnail.com/archives/10542


----------


本项目最初学习于：

熊猫的博客：http://blog.topspeedsnail.com/archives/10542

参考：https://github.com/ryankiros/neural-storyteller（根据图片生成故事）

TensorFlow练习3:http://blog.topspeedsnail.com/archives/10443

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

本帖代码移植自char-rnn：https://github.com/karpathy/char-rnn

使用rnn进行唐诗学习

poetry.txt 是2w+的唐诗
每一行对应一首诗，冒号前为诗的标题，冒号后面为诗的内容

如：生成的诗： 
梦惆清候 隔预成柳 冒腻闲盖家肯未，赤中似何狂沧宴 正恩脂恩懒睡锦，笼血芳哀

后续思路：如果样本都是情诗，就可以生成情诗 ....

----------
做标签时 前一个词预测后一个词
 如：[寒随穷律变，] --> 寒随穷律变，]]  即 [-->寒, 寒-->随,随-->穷,穷-->律,律-->变,变-->，；，-->]
生成文本时 先输入 [ -->词1 词1-->词2 ... 词n-->]  得到诗句 词1，词2 ... 词n （也可以输入固定词）

# 思路流程
![这里写图片描述](http://img.blog.csdn.net/20180116175326772?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


RNN不像传统的神经网络-它们的输出输出是固定的，而RNN允许我们输入输出向量序列。RNN是为了对序列数据进行建模而产生的。

> 样本序列性：样本间存在顺序关系，每个样本和它之前的样本存在关联。比如说，在文本中，一个词和它前面的词是有关联的；在气象数据中，一天的气温和前几天的气温是有关联的。

例如本帖要使用RNN生成古诗，你给它输入一堆古诗词，它会学着生成和前面相关联的字词。如果你给它输入一堆姓名，它会学着生成姓名；给它输入一堆古典乐/歌词，它会学着生成古典乐/歌词，甚至可以给它输入源代码。

本帖代码移植自[char-rnn](https://github.com/karpathy/char-rnn)，它是基于Torch的洋文模型，稍加修改即可应用于中文。char-rnn使用文本文件做为输入、训练RNN模型，然后使用它生成和训练数据类似的文本。

使用的数据集：全唐诗(43030首)：https://pan.baidu.com/s/1o7QlUhO


- https://github.com/ryankiros/neural-storyteller（根据图片生成故事）


