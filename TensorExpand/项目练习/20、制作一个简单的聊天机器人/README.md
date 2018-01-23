参考：http://blog.topspeedsnail.com/archives/10735


----------
from tensorflow.models.rnn.translate import seq2seq_model 出问题 待解决

现在很多卖货公司都使用聊天机器人充当客服人员，许多科技巨头也纷纷推出各自的聊天助手，如苹果Siri、Google Now、Amazon Alexa、微软小冰等等。前不久有一个[视频](https://www.youtube.com/watch?v=JFiu5rfnhzo)比较了Google Now和Siri哪个更智能，貌似Google Now更智能。

本帖使用TensorFlow制作一个简单的聊天机器人。这个聊天机器人使用中文对话数据集进行训练（使用什么数据集训练决定了对话类型）。使用的模型为RNN(seq2seq)，和前文的《RNN生成古诗词》《RNN生成音乐》类似。

相关博文：

- [使用深度学习打造智能聊天机器人](http://blog.csdn.net/malefactor/article/details/51901115)
- [脑洞大开：基于美剧字幕的聊天语料库建设方案](http://www.shareditor.com/blogshow/?blogId=105)
- [中文对白语料](https://github.com/fateleak/dgk_lost_conv)
- https://www.tensorflow.org/versions/r0.12/tutorials/seq2seq/index.html
- https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_shakespeare.py

# 数据集
我使用现成的[影视对白数据集](https://github.com/fateleak/dgk_lost_conv)，跪谢作者分享数据。

下载数据集

```python
$ wget https://raw.githubusercontent.com/rustch3n/dgk_lost_conv/master/dgk_shooter_min.conv.zip
# 解压
$ unzip dgk_shooter_min.conv.zip
```

数据预处理：

创建词汇表，然后把对话转为向量形式，参看练习1和7：

# 训练
需要很长时间训练，这还是小数据集，如果用百GB级的数据，没10天半个月也训练不完。

使用的模型：[seq2seq_model.py](https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py)。


