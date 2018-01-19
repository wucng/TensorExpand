参考：http://blog.topspeedsnail.com/archives/10508


----------
我在GitHub看到了一个使用RNN生成经典音乐的项目：[biaxial-rnn-music-composition](https://github.com/hexahedria/biaxial-rnn-music-composition)，它是基于Theano的。本帖改为使用TensorFlow生成音乐，代码逻辑在很大程度上基于前者。

相关博文：

- https://deeplearning4j.org/restrictedboltzmannmachine.html
- https://magenta.tensorflow.org/2016/06/10/recurrent-neural-network-generation-tutorial/
- https://deepmind.com/blog/wavenet-generative-model-raw-audio
- Google的项目[Magenta](https://github.com/tensorflow/magenta)：生成音乐、绘画或视频
- http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/
- TensorFlow练习7: 基于RNN生成古诗词

数据集：首先准备一些MIDI音乐，可以去freemidi.org下载

另一个关于音乐的数据集 [MusicNet](https://homes.cs.washington.edu/~thickstn/start.html)

![这里写图片描述](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7-2016-11-25-%E4%B8%8B%E5%8D%881.51.59.png)

有了MIDI音乐，我们还需要一个可以操作MIDI的Python库：[python-midi](https://github.com/vishnubob/python-midi)。

安装python-midi：<font color=FF22FF size=5>（只能安装python2 、Python3 没法安装）

```python
$ git clone https://github.com/vishnubob/python-midi
$ cd python-midi
# $ git checkout feature/python3   # 如果使用Python3，checkout对应分支
$ python setup.py install

# 或
pip install python-midi
```
MacOS没有内置midi支持，可以使用timidity播放midi：

```python
$ brew install timidity
$ timidity SpongBobSquarPantsTheme.mid
```

> 你也许不知道，你每次执行brew命令，它都会给Google发送匿名统计数据。[Analytics.md](https://github.com/Homebrew/brew/blob/master/docs/Analytics.md)


tensorflow while_loop用法：

```python
# 代码取自Stack OverFlow
import tensorflow as tf
import numpy as np


def body(x):
    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    c = a + b
    return tf.nn.relu(x + c)

def condition(x):
    return tf.reduce_sum(x) < 100

x = tf.Variable(tf.constant(0, shape=[2, 2]))

with tf.Session():
    tf.global_variables_initializer().run()
    result = tf.while_loop(condition, body, [x])
    print(result.eval())
```

```python
# 代码取自Stack OverFlow
import tensorflow as tf
import numpy as np


def body(x):
    # tf.assign_add(x,tf.constant(1,dtype=tf.int32))
    x=x+tf.constant(1,dtype=tf.int32)
    return x

def condition(x):
    return x < 1000

x = tf.Variable(tf.constant(0))

with tf.Session():
    tf.global_variables_initializer().run()
    result = tf.while_loop(condition, body, [x])
    print(result.eval())

# while x<1000:
#     x+=1
```

# TensorFlow生成mid音乐完整代码：

生成的mid音乐：[auto_gen_music](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/auto_gen_music.mid)

音乐分类：https://github.com/despoisj/DeepAudioClassification

# 模型思路
![这里写图片描述](http://img.blog.csdn.net/20180119161515180?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

