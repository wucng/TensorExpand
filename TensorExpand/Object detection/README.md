参考：

- [从论文到测试：Facebook Detectron开源项目初探](https://www.jiqizhixin.com/articles/2018-01-23-7)
- [Facebook Detectron](https://github.com/facebookresearch/Detectron)
- [Mask R-CNN论文](https://arxiv.org/abs/1703.06870)
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN) for object detection and instance segmentation on Keras and TensorFlow
- [Mask RCNN](https://github.com/CharlesShang/FastMaskRCNN) in TensorFlow
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [FastER RCNN built on tensorflow](https://github.com/CharlesShang/TFFRCNN)
- [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn) (Python implementation)
- [Faster R-CNN](https://github.com/ShaoqingRen/faster_rcnn)
- [smallcorgi/Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)
- [cs231n学习笔记-CNN-目标检测、定位、分割](https://www.jianshu.com/p/cef69c6651a9?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)


----------
Mask R-CNN是何凯明大神最近的新作。Mask R-CNN是一种在有效检测目标的同时输出高质量的实例分割mask。是对faster r-cnn的扩展，与bbox识别并行的增加一个预测分割mask的分支。Mask R-CNN 可以应用到人体姿势识别。并且在实例分割、目标检测、人体关键点检测三个任务都取得了现在最好的效果。

作者：flyingmoth
链接：https://www.jianshu.com/p/e8e445b38f6f
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

一个特点就是 Mask-RCNN 的检测和分割是并行出结果的，而不像以前是分割完了之后再做分类，结果是很 amazing 的。

Mask-RCNN 大体框架还是 Faster-RCNN 的框架，可以说在基础特征网络之后又加入了全连接的分割子网，由原来的两个任务（分类+回归）变为了三个任务（分类+回归+分割）

主要改进点在：

1. 基础网络的增强，ResNeXt-101+FPN的组合可以说是现在特征学习的王牌了

2. 分割 loss 的改进，由原来的 FCIS 的 基于单像素softmax的多项式交叉熵变为了基于单像素sigmod二值交叉熵，softmax会产生FCIS的 ROI inside map与ROI outside map的竞争。但文章作者确实写到了类间的竞争， 二值交叉熵会使得每一类的 mask 不相互竞争，而不是和其他类别的 mask 比较 。


[Mask-RCNN技术解析](http://blog.csdn.net/linolzhang/article/details/71774168)

![这里写图片描述](http://img.blog.csdn.net/20170614225558493)

其中 黑色部分为原来的 Faster-RCNN，红色部分为在 Faster网络上的修改：

1）将 Roi Pooling 层替换成了 RoiAlign；

2）添加并列的 FCN 层（mask 层）；

       先来概述一下 Mask-RCNN 的几个特点（来自于 Paper 的 Abstract）：

1）在边框识别的基础上添加分支网络，用于 语义Mask 识别；

2）训练简单，相对于 Faster 仅增加一个小的 Overhead，可以跑到 5FPS；

3）可以方便的扩展到其他任务，比如人的姿态估计 等；

4）不借助 Trick，在每个任务上，效果优于目前所有的 single-model entries；
	包括 COCO 2016 的Winners。

# RCNN行人检测框架

![这里写图片描述](http://img.blog.csdn.net/20170614225604196)

  图中灰色部分是 原来的 RCNN 结合 ResNet or FPN 的网络，下面黑色部分为新添加的并联 Mask层，这个图本身与上面的图也没有什么区别，旨在说明作者所提出的Mask RCNN 方法的泛化适应能力 - 可以和多种 RCNN框架结合，表现都不错。

# Mask-RCNN 技术要点

●技术要点1 - 强化的基础网络

     通过 ResNeXt-101+FPN 用作特征提取网络，达到 state-of-the-art 的效果。

● 技术要点2 - ROIAlign

     采用 ROIAlign 替代 RoiPooling（改进池化操作）。引入了一个插值过程，先通过双线性插值到14*14，再 pooling到7*7，很大程度上解决了仅通过 Pooling 直接采样带来的 Misalignment 对齐问题。
   
    PS： 虽然 Misalignment 在分类问题上影响并不大，但在 Pixel 级别的 Mask 上会存在较大误差。

● 技术要点3 - Loss Function

     每个 ROIAlign 对应 K * m^2 维度的输出。K 对应类别个数，即输出 K 个mask，m对应 池化分辨率（7*7）。Loss 函数定义：

            Lmask(Cls_k) = Sigmoid (Cls_k)，    平均二值交叉熵 （average binary cross-entropy）Loss，通过逐像素的 Sigmoid 计算得到。

     Why K个mask？通过对每个 Class 对应一个 Mask 可以有效避免类间竞争（其他 Class 不贡献 Loss ）。

![这里写图片描述](http://img.blog.csdn.net/20170614225609072)

# Mask-RCNN 扩展
Mask-RCNN 在姿态估计上的扩展



>Keras [https://github.com/matterport/Mask_RCNN\]
>TensorFlow [https://github.com/CharlesShang/FastMaskRCNN]
>Pytorch [https://github.com/felixgwu/mask_rcnn_pytorch\]
>caffe [https://github.com/jasjeetIM/Mask-RCNN]
>MXNet [https://github.com/TuSimple/mx-maskrcnn]


-------
-------

[R-CNN读书笔记](https://www.jianshu.com/p/9bcbf6d98238)
[R-CNN 物体检测第一弹](https://www.jianshu.com/p/52e6e184b786)

[FAST R-CNN 论文笔记](https://www.jianshu.com/p/7b5486aafac0)
[R-CNN 物体检测第二弹（Fast R-CNN）](https://www.jianshu.com/p/7c35ba55ad61)

[FASTER RCNN 论文笔记](https://www.jianshu.com/p/a684160e99c2)
[R-CNN目标检测第三弹（Faster R-CNN）](https://www.jianshu.com/p/8f78a9350117)

[Mask R-CNN](https://www.jianshu.com/p/e8e445b38f6f)


----------


视频：

https://search.bilibili.com/all?keyword=rcnn

[深度学习计算机视觉Mask R-CNN](https://www.bilibili.com/video/av15949583/?from=search&seid=13375581300820582393)

[【 深度学习计算机视觉Faster R-CNN 】](https://www.bilibili.com/video/av15949356/?from=search&seid=13375581300820582393)

[斯坦福深度学习课程CS231N 2017中文字幕版+全部作业参考](https://www.bilibili.com/video/av17204303/?from=search&seid=14478893974543927880)


# 教程
[深度学习一行一行敲faster rcnn-keras版(目录)](https://zhuanlan.zhihu.com/p/31530023)

# github
- [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
- [yhenon/keras-frcnn](https://github.com/yhenon/keras-frcnn)
- [shekkizh/FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow)
- [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)
- [CasiaFan/tensorflow_retinanet](https://github.com/CasiaFan/tensorflow_retinanet)

# 目标分割算法
[从论文到测试：Facebook Detectron开源项目初探](https://www.jiqizhixin.com/articles/2018-01-23-7)

Fast RCNN、Faster RCNN、RFCN、FPN、RetinaNet、Mask RCNN


