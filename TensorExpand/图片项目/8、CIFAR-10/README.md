参考：

- [TensorFlow 教程 #06 - CIFAR-10](https://zhuanlan.zhihu.com/p/27017189)
- [Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [thrillerist/TensorFlow-Tutorials](https://github.com/thrillerist/TensorFlow-Tutorials)


----------

这篇教程介绍了如何创建一个在CIRAR-10数据集上进行图像分类的卷积神经网络。同时也说明了在训练和测试时如何使用不同的网络。

下面的图表直接显示了之后实现的卷积神经网络中数据的传递。首先有一个扭曲（distorts）输入图像的预处理层，用来人为地扩大训练集。接着有两个卷积层，两个全连接层和一个softmax分类层。在后面会有更大的图示来显示权重和卷积层的输出，教程 #02 有卷积如何工作的更多细节。

![这里写图片描述](https://pic4.zhimg.com/80/v2-26b648c6544f0e8124a48aaae8d10d88_hd.jpg)

研究一些[CIFAR-10上的更好的神经网络](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html) ，试着实现它们
