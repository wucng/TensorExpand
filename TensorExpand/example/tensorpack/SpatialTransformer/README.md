[Spatial Transformer Networks](https://github.com/tensorpack/tensorpack/tree/master/examples/SpatialTransformer)

给定堆叠在两个通道中的两个失真的MNIST数字的图像，训练网络以产生它们的总和。 这里，两个Spatial Transformer分支学习本地化两个数字并分别扭曲它们。

![这里写图片描述](https://github.com/tensorpack/tensorpack/raw/master/examples/SpatialTransformer/demo.jpg)

- 左：输入图像。
- 中：第一个STN分支的输出（定位第二个数字）。
- 右：第二个STN分支的输出。

训练（需要大约300个epochs才能达到8.8％的误差）：

```
./mnist-addition.py
```
使用预训练模型绘制上述可视化：

```
./mnist-addition.py --load mnist-addition.npz --view
```