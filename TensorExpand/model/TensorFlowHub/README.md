参考：

- https://tensorflow.google.cn/hub/
- [tensorflow/hub](https://github.com/tensorflow/hub) 通过重新使用TensorFlow模型的一部分来实现迁移学习的库
- [迁移学习/fine-tuning](https://blog.csdn.net/wc781708249/article/details/80051463)
- [TF-Slim 实现模型迁移/微调](https://blog.csdn.net/wc781708249/article/details/80085041) 
-  [TF-slim实现自己数据模型微调](https://blog.csdn.net/wc781708249/article/details/80095578) 

------

TensorFlow Hub是一个library，用于促进机器学习模型的可重用部分的发布，发现和使用。 一个模块是TensorFlow图形的一个独立部分，以及它的权重和资产，可以在称为转移学习的过程中在不同任务中重复使用。

模块包含已使用大型数据集为任务预先训练的变量。 通过在相关任务上重用模块，您可以：


- 用较小的数据集训练模型
- 改进泛化
- 大大加快了训练

下面是一个使用英文嵌入模块将字符串数组映射到其嵌入的示例：

```python
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
  embeddings = embed(["A long sentence.", "single-word", "http://example.com"])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))

```

# Installation 
TensorFlow Hub依赖于1.7版之前的TensorFlow版本中不存在的错误修复和增强功能。 您必须安装或升级您的TensorFlow软件包至少1.7才能使用TensorFlow Hub：

```python
$ pip install "tensorflow>=1.7.0"
$ pip install tensorflow-hub
```
本部分将进行更新，以在兼容版本可用时包含特定的TensorFlow版本要求。


