# [Holistically-Nested Edge Detection](https://github.com/tensorpack/tensorpack/tree/master/examples/HED)(整体嵌套边缘检测)

HED是一种完全卷积的架构。 此代码通常也适用于其他FCN任务，例如语义分段和检测。

此脚本需要原始BSDS数据集并动态应用扩充。 如果没有，它会自动将数据集下载到`$TENSORPACK_DATASET/`。

它需要预训练的vgg16模型。 有关从vgg16 caffe模型转换的说明，请参阅[examples / CaffeModels](https://github.com/tensorpack/tensorpack/blob/master/examples/CaffeModels)中的文档。

要查看增强的训练图像：

```
./hed.py --view
```
# training:

```
./hed.py --load vgg16.npz
```
它需要大约100k步（在TitanX上大约10个小时）才能达到合理的性能。

推断（在out * .png处生成每个级别的热图）：

```
./hed.py --load pretrained.model --run a.jpg
```

![](../images/HED.png)