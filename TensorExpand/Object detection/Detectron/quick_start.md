<font size=4>

[toc]

# Using Detectron

本文档提供了有关Detectron的简要教程，用于COCO数据集的推理和培训。

- 有关Detectron的一般信息，请参阅[README.md](https://github.com/facebookresearch/Detectron/blob/master/README.md)。
- 有关安装说明，请参阅[INSTALL.md](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)。

# Inference with Pretrained Models

## 1、图像文件目录
要在图像文件目录（本例中为`demo/*.jpg`）上运行推理，可以使用`infer_simple.py`工具。 在这个例子中，我们使用端到端训练的`Mask R-CNN`模型和模型动物园的`ResNet-101-FPN`骨干：

```python
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \ # 网络结构配置，必须与下面的wts对应
    --output-dir /tmp/detectron-visualizations \ # 图片保存的位置
    --image-ext jpg \ # 图片的后缀名
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \ # 在线提取的权重参数，如果预先下载好了，可以通过这里直接指定其路径
    demo # 要推理的图片目录（这里是`demo/*.jpg`）
 
# 默认是保存成PDF格式，如果要改变保存的格式，如保存成.jpg,可以修改output-ext：
# --output-ext jpg
```
Detectron应自动从`--wts`参数指定的URL下载模型。 此工具将在`--output-dir`指定的目录中输出PDF格式的检测的可视化。 以下是您应该看到的输出示例（有关演示图像的版权信息，请参阅[demo/NOTICE](https://github.com/facebookresearch/Detectron/blob/master/demo/NOTICE)）。

**注意：**

当在您自己的高分辨率图像上进行推断时，Mask R-CNN可能很慢，因为花费了大量时间将预测的掩模上采样到原始图像分辨率（这尚未优化）。 如果`tools/infer_simple.py`报告的`misc_mask`时间很长（例如，大于20-90ms），则可以诊断此问题。 解决方案是首先调整图像的大小，使短边大约为`600-800px`（确切的选择无关紧要），然后在调整大小的图像上进行推断。

## 1.1 拓展到keypoints

在[model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) 下载对应的模型，在下载链接中可以查看对应的配置文件（`cfg`对应的文件路径）

<font color=#FF0000 size=5>注：使用`keypoint_rcnn_R-50-FPN_s1x.pkl`会报错，需使用`e2e_keypoint_rcnn_R-50-FPN_s1x.pkl` (即需使用`e2e`对应的模型)</font>
```python
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml \ # 网络结构配置，必须与下面的wts对应
    --output-dir /tmp/detectron-visualizations \ # 图片保存的位置
    --image-ext jpg \ # 图片的后缀名
    --wts https://s3-us-west-2.amazonaws.com/detectron/37697714/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml.08_44_03.qrQ0ph6M/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \ # 在线提取的权重参数，如果预先下载好了，可以通过这里直接指定其路径
    demo # 要推理的图片目录（这里是`demo/*.jpg`）
```

## 2. COCO Dataset（推荐第一种方式）
参考[此处](https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data)COCO数据配置，

```python
# test_net.py会自动从coco_2014_minival加载预推理的图片，所有只需将要做推理的图片放在这个目录下即可
mkdir -p xx/detectron/datasets/data/coco/coco_2014_minival 
cp *.jpg xx/detectron/datasets/data/coco/coco_2014_minival
```

此示例显示如何使用单个GPU进行推理，从模型zoo运行端到端训练的Mask R-CNN模型。 根据配置，这将对`coco_2014_minival`中的所有图像进行推断（必须正确安装）。

```
python2 tools/test_net.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    TEST.WEIGHTS https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    NUM_GPUS 1
```



使用`$N` GPU（例如，N = 8）使用相同模型运行推理。

```
python2 tools/test_net.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    NUM_GPUS $N
```
在NVIDIA Tesla P100 GPU上，对于此示例，每张图像的推理大约需要130-140 ms。
# Training a Model with Detectron
参考[此处](https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data)COCO数据配置，来配置自己的数据集

这是一个很小的教程，展示了如何在COCO上训练模型。 该模型将是使用`ResNet-50-FPN`骨干的端到端训练的``Faster R-CNN`。 出于本教程的目的，我们将使用较短的训练计划和较小的输入图像大小，以便训练和推理相对较快。 因此，与我们的[基线](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)相比，COCO上的box AP将相对较低。 提供此示例仅用于指导目的（即，不用于与出版物进行比较）。

## 1. Training with 1 GPU

```
python2 tools/train_net.py \
    --cfg configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    OUTPUT_DIR /tmp/detectron-output
```
预期结果：

- 输出（模型，验证集检测等）将保存在`/tmp/detectron-output`下
- 在Maxwell一代GPU（例如M40）上，训练大约需要4.2小时
- 推理时间应约为80ms /图像（也在M40上）
- `coco_2014_minival`上的Box AP应该在22.1％左右（在3次运行中测得+/- 0.1％stdev）

## 2. Multi-GPU Training
我们还提供了配置来说明使用学习计划的2,4和8 GPU训练，这些学习计划大致相当于上面1 GPU使用的学习计划。 配置位于：`configs/getting_started/tutorial_{2,4,8}gpu_e2e_faster_rcnn_R-50-FPN.yaml`。 例如，启动具有2个GPU的培训作业将如下所示：

```python
python2 tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/getting_started/tutorial_2gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    OUTPUT_DIR /tmp/detectron-output
```
请注意，我们还添加了`--multi-gpu-testing`标志，以指示Detectron在训练结束后并行化多个GPU（本例中为2;请参阅配置文件中的`NUM_GPUS`）的推理。

**预期结果：**

- 训练大约需要2.3小时（2 x M40）
- 推理时间应约为`80ms/image`（但在2个GPU上并行，因此总时间的一半）
- `coco_2014_minival`上的Box AP应该在22.1％左右（在3次运行中测得+/- 0.1％stdev）

要了解如何调整学习计划（“线性缩放规则”），请学习这些教程配置文件并阅读我们的论文 [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)。 除了本教程之外，我们发布的所有配置都使用了8个GPU。 如果您将使用少于8个GPU进行培训（或执行任何其他更改小批量大小的操作），则必须了解如何根据线性缩放规则操作培训计划。

注意：

此训练示例使用相对较低的GPU计算模型，因此Caffe2 Python操作的开销相对较高。 结果，随着GPU的数量从2增加到8的缩放相对较差（例如，使用8个GPU的训练需要大约0.9小时，仅比使用1个GPU快4.5倍）。 随着更大，使用更多GPU计算的重型模型，缩放得到改善。