参考：

- [TensorFlow Tutorial #10](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb)
- [Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [thrillerist/TensorFlow-Tutorials](https://github.com/thrillerist/TensorFlow-Tutorials)


----------
我们以前在教程＃08和＃09中看到了如何使用所谓的转移学习在新的数据集上使用预先训练的神经网络，方法是将原始模型的输出重新路由到分类层之前， 我们创建了一个新的分类器。 由于原来的模型是“冻结的”，它的权重不能被进一步优化，所以模型中所有前面的层次所学的东西，都不能被微调到新的数据集。

本教程演示如何使用Keras API进行Tensorflow的Transfer Learning和Fine-Tuning。 我们将再次使用Tutorial＃09中介绍的Knifey-Spoony数据集。 我们以前使用的是Inception v3模型，但是我们将在本教程中使用VGG16模型，因为它的架构更易于使用。

注：使用2.6 GHz CPU和GTX 1070 GPU的笔记本电脑上执行此笔记本大约需要15分钟。 在CPU上运行它估计需要大约10个小时！

# 流程图
这个想法是重新使用预先训练的模型，在这种情况下，VGG16模型由几个卷积层（实际上是多个卷积层的块）组成，接着是一些完全连接/密集的层，然后是一个softmax输出层 为分类。

密集层负责结合卷积层的特征，这有助于最终的分类。 所以当VGG16模型用于其他数据集时，我们可能需要替换所有的密集层。 在这种情况下，我们添加另一个密集层和一个压缩层以避免过度拟合。

<font size=4 color=#FF00FF>**转移学习与微调之间的区别**</font>在于，在转移学习中，我们只优化我们添加的新分类层的权重，同时保留原始VGG16模型的权重。 在Fine-Tuning中，我们优化了我们添加的新分类图层的权重，以及VGG16模型中的部分或全部图层。

![这里写图片描述](https://github.com/Hvass-Labs/TensorFlow-Tutorials/raw/c68d9601a3a5d1a955e9ecf7d05a18fc2e5f56a6/images/10_transfer_learning_flowchart.png)

# 迁移学习与网络微调

![这里写图片描述](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/%E5%9B%BE%E7%89%87%E9%A1%B9%E7%9B%AE/10%E3%80%81%E5%BE%AE%E8%B0%83%E7%BD%91%E7%BB%9C/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%BE%AE%E8%B0%83.png)
