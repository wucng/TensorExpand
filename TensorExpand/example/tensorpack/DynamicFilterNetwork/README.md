[Dynamic Filter Networks](https://github.com/tensorpack/tensorpack/tree/master/examples/DynamicFilterNetwork)

在动态滤波器网络中重现“学习转向滤波器”实验。

输入图像由动态学习的滤波器进行卷积以匹配地面实况图像。 最后，滤波器会聚到真正的转向滤波器，用于产生基本事实。

这也演示了如何直接从Python将图像放入tensorboard。

```
./steering-filter.py --gpu 0
```