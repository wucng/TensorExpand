参考：https://github.com/luyishisi/tensorflow

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
