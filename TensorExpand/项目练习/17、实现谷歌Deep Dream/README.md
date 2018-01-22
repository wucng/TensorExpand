参考：http://blog.topspeedsnail.com/archives/10667


----------
本帖使用谷歌的预训练的[Inception模型](https://github.com/tensorflow/models/tree/master/research/inception/inception)生成带有艺术感的图片。

Inception模型是Google用两个星期，使用上百万张带分类的图片训练出的模型，在做[图像识别](https://tensorflow.org/tutorials/image_recognition/)时，为了节省时间，通常使用预训练的Inception模型做为训练基础。

Deep Dream是取预训练模型的某一层（神经网络有59层，前几层学会底层特性，像线、角，经过层层抽象，最后几层可以表示更高层次的特性），然后最大化我们提供的图像和某个层相似的特性，最后生成非常有意思的图像。

关于Deep Dream：

- https://github.com/google/deepdream
- http://www.alanzucconi.com/2016/05/25/generating-deep-dreams/
- http://ryankennedy.io/running-the-deep-dream/
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream

下图是lena经过《[open_nsfw: 基于Caffe的成人图片识别模型](http://blog.topspeedsnail.com/archives/9440)》处理后生成的图像：

![这里写图片描述](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/CtyNj-uVIAAQ97P-562x1024.jpg)


- https://open_nsfw.gitlab.io（未满18岁，请绕行；自备钛合金）

下载预训练的Inception模型：

```python
$ wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
# 解压
$ unzip inception5h.zip
```

代码：

