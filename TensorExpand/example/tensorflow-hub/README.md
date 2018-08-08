# [TensorFlow Hub](https://github.com/tensorflow/hub)

TensorFlow Hub是一个用于促进机器学习模型的可重用部分的发布，发现和消费的库。 特别是，它提供了模块，这些模块是经过预先训练的TensorFlow模型，可以在新任务中重复使用。

# Getting Started
https://tensorflow.google.cn/hub/
# Introduction
TensorFlow Hub是可重用机器学习模块的库。

模块是TensorFlow图形的独立部分，连同其权重和资产，可以在称为转移学习的过程中在不同任务中重复使用。Transfer learning可以：

- 使用较小的数据集训练模型，
- 改善泛化，和
- 加快训练速度。

```python
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
  module_url = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
  embed = hub.Module(module_url)
  embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))
```
# Installation
TensorFlow Hub依赖于1.7之前的TensorFlow版本中不存在的错误修复和增强功能。 您必须将TensorFlow软件包安装或升级到至少1.7才能使用TensorFlow Hub：

```
$ pip3 install "tensorflow>=1.7.0"
$ pip3 install tensorflow-hub
```
当兼容的版本可用时，此部分将更新为包含特定的TensorFlow版本要求。

# Tutorials
- [Image Retraining](https://tensorflow.google.cn/hub/tutorials/image_retraining)

- [Text Classification](https://tensorflow.google.cn/hub/tutorials/text_classification_with_tf_hub)

- [Additional Examples](https://github.com/tensorflow/hub/blob/master/examples/README.md)

# Key Concepts:
- [Using a Module](https://github.com/tensorflow/hub/blob/master/docs/basics.md)

- [Creating a New Module](https://github.com/tensorflow/hub/blob/master/docs/creating.md)
- [Fine-Tuning a Module](https://github.com/tensorflow/hub/blob/master/docs/fine_tuning.md)

# Modules
- Available Modules -- quick links: image, text, other
- [Common Signatures for Modules](https://github.com/tensorflow/hub/blob/master/docs/common_signatures/index.md)
