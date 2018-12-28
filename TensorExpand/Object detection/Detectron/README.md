- https://github.com/facebookresearch/Detectron
- [预训练模型](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)

---
# Introduction
Detectron的目标是为物体检测研究提供高质量，高性能的代码库。 它旨在灵活，以支持新颖研究的快速实施和评估。 Detectron包括以下对象检测算法的实现：

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) -- Marr Prize at ICCV 2017
- [RetinaNet](https://arxiv.org/abs/1708.02002) -- Best Student Paper Award at ICCV 2017
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [RPN](https://arxiv.org/abs/1506.01497)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [R-FCN](https://arxiv.org/abs/1605.06409)

使用以下骨干网络架构：

- [ResNeXt{50,101,152}](https://arxiv.org/abs/1611.05431)
- [ResNet{50,101,152}](https://arxiv.org/abs/1512.03385)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) (with ResNet/ResNeXt)
- [VGG16](https://arxiv.org/abs/1409.1556)

可以容易地实现额外的骨干架构。 有关这些型号的更多详细信息，请参阅下面的[参考资料](https://github.com/facebookresearch/Detectron#references)。


# Model Zoo and Baselines
我们提供了大量基础结果和训练模型，可在Detectron[模型zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)中下载。

# Installation
参考[here](./install.md)

# Quick Start: Using Detectron
参考[here](./quick_start.md)

# 训练coco数据
参考[here](./train_inference.md)

如果训练自己的数据需将自己的数据转成COCO格式，参考[here](https://blog.csdn.net/wc781708249/article/details/79615210)
