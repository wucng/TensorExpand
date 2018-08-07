- https://github.com/tensorpack/tensorpack
- [model zoon](http://models.tensorpack.com/FasterRCNN/)

---
[Tensorpack](https://github.com/tensorpack/tensorpack)是一个基于TensorFlow的神经网络训练界面。

# 特点
它是另一个TF高级API，具有速度，可读性和灵活性。

- 专注于训练速度。

	- Tensorpack免费提供速度 - 它以高效的方式使用TensorFlow，无需额外开销。在常见的CNN上，它比同等的Keras代码运行速度快[1.2~5倍](https://github.com/tensorpack/benchmarks/tree/master/other-wrappers)。

	- 数据并行多GPU /分布式培训策略现成可供使用。它的扩展性与[谷歌的官方基准一样](https://www.tensorflow.org/performance/benchmarks)。

	- 有关一些基准脚本，请参阅[tensorpack/benchmarkmark](https://github.com/tensorpack/benchmarks)。

- 2、专注于大型数据集。

 - 你通常[不需要tf.data](http://tensorpack.readthedocs.io/tutorial/input-source.html#tensorflow-reader-cons)。符号编程通常会使数据处理更加困难。 Tensorpack可帮助您使用自动并行化在纯Python中高效处理大型数据集（例如ImageNet）。

- 3、它不是模型包装器。

	 - 世界上有太多的符号功能包装器。 Tensorpack仅包含一些常见模型。但你可以在Tensorpack中使用任何符号函数库，包括`tf.layers / Keras / slim / tflearn / tensorlayer / ....`

请参阅[教程](http://tensorpack.readthedocs.io/tutorial/index.html#user-tutorials)以了解有关这些功能的更多信息。

# [Examples](https://github.com/tensorpack/tensorpack/blob/master/examples):

我们拒绝玩具的例子。 [Tensorpack实例](https://github.com/tensorpack/tensorpack/blob/master/examples)不是向您展示10个在玩具数据集上训练的任意网络，而是忠实地复制论文并关注复制数字，展示其实际研究的灵活性。

# Vision:

- 在ImageNet上训练[ResNet](https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet)和[其他模型](https://github.com/tensorpack/tensorpack/blob/master/examples/ImageNetModels)。
- [在COCO对象检测上训练更快 - RCNN / Mask-RCNN](https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN)
- [生成性对抗网络（GAN）变体](https://github.com/tensorpack/tensorpack/blob/master/examples/GAN)，包括DCGAN，InfoGAN，条件GAN，WGAN，BEGAN，DiscoGAN，图像到图像，CycleGAN。
- [DoReFa-Net：在ImageNet上训练二进制/低位宽CNN](https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net)
- [用于整体嵌套边缘检测（HED）的全卷积网络](https://github.com/tensorpack/tensorpack/blob/master/examples/HED)
- [MNIST上的空间变换网络](https://github.com/tensorpack/tensorpack/tree/master/examples/SpatialTransformer)
- [可视化CNN显着性图](https://github.com/tensorpack/tensorpack/tree/master/examples/Saliency)
- [关于MNIST的相似性学习](https://github.com/tensorpack/tensorpack/blob/master/examples/SimilarityLearning)


# Reinforcement Learning:
- [Atari游戏的深度Q-Network（DQN）变体](https://github.com/tensorpack/tensorpack/tree/master/examples/DeepQNetwork)，包括DQN，DoubleDQN，DuelingDQN。
- [Asynchronous Advantage Actor-Critic（A3C）在OpenAI Gym](https://github.com/tensorpack/tensorpack/blob/master/examples/A3C-Gym)上进行了演示

# Speech / NLP:
- [用于语音识别的LSTM-CTC](https://github.com/tensorpack/tensorpack/blob/master/examples/CTC-TIMIT)
- [char-rnn的乐趣](https://github.com/tensorpack/tensorpack/blob/master/examples/Char-RNN)
- [PennTreebank上的LSTM语言模型](https://github.com/tensorpack/tensorpack/tree/master/examples/PennTreebank)

# Install:
- Python 2.7 or 3.3+. Python 2.7 is supported until it retires in 2020.
- OpenCV的Python绑定（可选，但许多功能都需要）
- TensorFlow> = 1.3。 （如果您只想将`tensorpack.dataflow`作为数据处理库使用，则不需要TensorFlow）

```python
pip3/pip2 install --upgrade git+https://github.com/tensorpack/tensorpack.git
# or add `--user` to avoid system-wide installation.
```
# 引用Tensorpack：
如果您在研究中使用Tensorpack或希望参考这些示例，请引用：

```
@misc{wu2016tensorpack,
  title={Tensorpack},
  author={Wu, Yuxin and others},
  howpublished={\url{https://github.com/tensorpack/}},
  year={2016}
}
```