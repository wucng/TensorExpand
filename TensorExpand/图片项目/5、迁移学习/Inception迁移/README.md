参考：

- [TensorFlow 教程 #08 - 迁移学习](https://zhuanlan.zhihu.com/p/27093918)
- [Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [thrillerist/TensorFlow-Tutorials](https://github.com/thrillerist/TensorFlow-Tutorials)


----------
当新数据集里的所有图像都用Inception处理过，并且生成的transfer-values都保存到缓存文件之后，我们可以将这些transfer-values作为其它神经网络的输入。接着训练第二个神经网络，用来分类新的数据集，因此，网络基于Inception模型的transfer-values来学习如何分类图像。

Inception 输入大小 299x299

![这里写图片描述](https://pic2.zhimg.com/80/v2-598e98856a8e6af4950fc829ca3d923e_hd.jpg)

# 基本流程图

![这里写图片描述](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/%E5%9B%BE%E7%89%87%E9%A1%B9%E7%9B%AE/5%E3%80%81%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0/Inception%E8%BF%81%E7%A7%BB/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0.png)
