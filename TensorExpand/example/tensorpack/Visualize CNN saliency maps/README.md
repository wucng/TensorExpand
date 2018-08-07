[Visualize Saliency Maps & Class Activation Maps](https://github.com/tensorpack/tensorpack/tree/master/examples/Saliency)

# Saliency Maps
`saliency-maps.py`获取一个图像，并通过运行ResNet-50生成其显着图，并将其最大激活返回到输入图像空间。 类似的技术可用于可视化网络中每个过滤器所学习的概念。

```
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzvf resnet_v1_50_2016_08_28.tar.gz
./saliency-maps.py cat.jpg
```
![这里写图片描述](https://github.com/tensorpack/tensorpack/blob/master/examples/Saliency/guided-relu-demo.jpg)

Left to right:

- 原始的猫形象
- 显着图中的幅度
- 幅度与原始图像混合
- 正相关像素（保持原始颜色）
- 负相关像素（保持原始颜色）

# CAM
`CAM-resnet.py`对Preact-ResNet进行微调，使其具有2倍大的最后层特征映射，然后生成CAM可视化。

1、微调或重新训练ResNet：

```
./CAM-resnet.py --data /path/to/imagenet [--load ImageNet-ResNet18-Preact.npz] [--gpu 0,1,2,3]
```
2、在ImageNet验证集上生成CAM：

```
./CAM-resnet.py --data /path/to/imagenet --load ImageNet-ResNet18-Preact-2xGAP.npz --cam
```
![这里写图片描述](https://github.com/tensorpack/tensorpack/blob/master/examples/Saliency/CAM-demo.jpg)