参考：

- [18_TFRecords_Dataset_API.ipynb](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb)
- [Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)


----------
# 介绍
在之前的教程中，我们使用了一个所谓的feed-dict来将数据输入到TensorFlow图中。 这是一个相当简单的输入法，但它也是一个性能瓶颈，因为数据是在训练步骤之间顺序读取的。 这使得GPU难以以100％的效率使用GPU，因为GPU必须等待新数据才能工作。
相反，我们希望在并行线程中读取数据，以便在GPU准备就绪时新的训练数据始终可用。 这曾经是TensorFlow中所谓的QueueRunner，这是一个非常复杂的系统。 如本教程中所述，现在可以使用Dataset API和称为TFRecords的二进制文件格式完成。
这建立在Estimator API的Tutorial＃17上。

# 结论
本教程展示了如何将TensorFlow的二进制文件格式的TFRecords与数据集和Estimator API一起使用。这应该可以简化使用非常大型数据集的模型训练过程，同时获得GPU的高使用率。但是，API在许多方面可能更简单。
# 练习
这些是一些可能有助于提高TensorFlow技能的练习建议。获得TensorFlow的实践经验对于学习如何正确使用它非常重要。
进行任何更改之前，您可能需要备份此笔记本。
- 训练卷积神经网络的时间更长。在分类刀剑数据集时它会更好吗？
在TFRecord中保存单热编码标签而不是类整数，并修改其余代码以使用它。
- 制作碎片，以便保存多个TFRecord文件而不是一个。
- 将jpeg文件保存在TFRecord中，而不是解码图像。然后，您需要解析parse（）函数中的jpeg-image。什么是专业人士和骗子这样做？
- 尝试使用其他数据集。
- 在图像大小不同的情况下使用数据集。你会在转换成TFRecords文件之前或之后调整大小吗？为什么？
- 尝试使用numpy输入函数而不是TFRecords作为Estimator API。性能差异是什么？
- 向朋友解释程序是如何工作的。
