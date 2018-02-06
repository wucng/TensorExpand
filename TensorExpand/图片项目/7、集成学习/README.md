参考：

- [TensorFlow 教程 #05 - 集成学习](https://zhuanlan.zhihu.com/p/26943434)
- [Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [thrillerist/TensorFlow-Tutorials](https://github.com/thrillerist/TensorFlow-Tutorials)


----------

题图来自[Combining Classifiers](http://cse-wiki.unl.edu/wiki/index.php/Combining_Classifiers)

这篇教程介绍了卷积神经网络的集成（**ensemble**）。我们使用多个神经网络，然后取它们输出的平均，而不是只用一个。

最终也是在MINIST数据集上识别手写数字。ensemble稍微地提升了测试集上的分类准确率，但差异很小，也可能是随机出现的。此外，ensemble误分类的一些图像在单独网络上却是正确分类的。

# 流程图
![这里写图片描述](https://pic1.zhimg.com/v2-d91feb5235804f19a08f0fc9699ea042_r.jpg)


下面的图表直接显示了之后实现的卷积神经网络中数据的传递。网络有两个卷积层和两个全连接层，最后一层是用来给输入图像分类的。关于网络和卷积的更多细节描述见教程 #02 。

本教程实现了5个这样的神经网络的集成，每个网络的结构相同但权重以及其他变量不同。

![这里写图片描述](https://pic3.zhimg.com/80/v2-d321f7bf4d4d2cbe2baf10d0f338e667_hd.jpg)



