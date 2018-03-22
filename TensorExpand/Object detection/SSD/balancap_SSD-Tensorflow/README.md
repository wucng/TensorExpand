参考：

- [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)
- [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [rykov8/ssd_keras](https://github.com/rykov8/ssd_keras)
- [zhreshold/mxnet-ssd](https://github.com/zhreshold/mxnet-ssd)


----------
# SSD: Single Shot MultiBox Detector in TensorFlow
SSD是用于单个网络的对象检测的统一框架。 它最初是在[这篇研究文章](http://arxiv.org/abs/1512.02325)中介绍的。

该存储库包含原始[Caffe代码](https://github.com/weiliu89/caffe/tree/ssd)的TensorFlow重新实现。 目前，它仅实现基于VGG的SSD网络（具有300和512个输入），但该项目的架构是模块化的，并且应该便于其他SSD变体（例如ResNet或Inception）的实施和培训。 现在的TF检查点已经从SSD Caffe模型直接转换而来。

该组织受TF-Slim模型库的启发，其中包含流行体系结构（ResNet，Inception和VGG）的实现。 因此，它分为三个主要部分：

- 数据集：与流行数据集（Pascal VOC，COCO，...）的接口以及将前者转换为TF-Records的脚本;

- 网络：定义SSD网络，以及常见的编码和解码方法（我们参考关于这个确切主题的论文）;

- 预处理：受原始VGG和Inception实施启发的预处理和数据增强例程。


----------
# SSD minimal example
[SSD笔记本](https://github.com/balancap/SSD-Tensorflow/blob/master/notebooks/ssd_notebook.ipynb)包含SSD TensorFlow管道的最小示例。 稍后，检测由两个主要步骤组成：在图像上运行SSD网络并使用常用算法（top-k filtering和非最大抑制算法）对输出进行后处理。

以下是成功检测结果的两个示例：

![这里写图片描述](https://github.com/balancap/SSD-Tensorflow/raw/master/pictures/ex1.png)

要运行笔记本，您首先必须将`./checkpoint`中的检查点文件解压缩

```
unzip ssd_300_vgg.ckpt.zip
```
然后开始一个jupyter笔记本

```
jupyter notebook notebooks/ssd_notebook.ipynb
```


----------
# Datasets
目前的版本只支持Pascal VOC数据集（2007和2012）。 为了用于训练SSD模型，前者需要使用`tf_convert_data.py`脚本转换为TF-Records：

```
DATASET_DIR=./VOC2007/test/
OUTPUT_DIR=./tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}
```

请注意，上面命令生成了一个TF-Records集合，而不是一个文件，以便在训练期间简化洗牌。

# Evaluation on Pascal VOC 2007
SSD模型的当前TensorFlow实现具有以下性能：
| Model| Training data| Testing data|mAP|FPS
| ------------- |:-------------:| -----:|
| [SSD-300 VGG-based](https://drive.google.com/open?id=0B0qPCUZ-3YwWZlJaRTRRQWRFYXM) | VOC07+12 trainval	|VOC07 test | 0.778 |-
| [SSD-300 VGG-based](https://drive.google.com/open?id=0B0qPCUZ-3YwWZlJaRTRRQWRFYXM) | VOC07+12+COCO trainval | VOC07 test |0.817|-
| [SSD-512 VGG-based](https://drive.google.com/open?id=0B0qPCUZ-3YwWT1RCLVZNN3RTVEU) | VOC07+12+COCO trainval | VOC07 test |0.837|-


----------
我们正在努力重现与[原始Caffe](https://github.com/weiliu89/caffe/tree/ssd)实施相同的性能！

在下载并提取之前的检查点之后，运行以下命令可以重现评估指标：

```
EVAL_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1
```
评估脚本提供召回 - 精度曲线的估计值，并根据Pascal VOC 2007和2012指南计算mAP指标。

另外，如果想要试验/测试不同的Caffe SSD检查点，前者可以转换为TensorFlow检查点，如下所示：

```python
CAFFE_MODEL=./ckpts/SSD_300x300_ft_VOC0712/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel
python caffe_to_tensorflow.py \
    --model_name=ssd_300_vgg \
    --num_classes=21 \
    --caffemodel_path=${CAFFE_MODEL}
```

# Training
脚本`train_ssd_network.py`负责训练网络。 与TF-Slim模型类似，可以将许多选项传递给训练过程（数据集，优化器，超参数，模型...）。 特别是，可以提供一个可用作起点的检查点文件，以便对网络进行微调。

# Fine-tuning existing SSD checkpoints
微调SSD模型的最简单方法是使用预先训练的SSD网络（VGG-300或VGG-512）。 例如，可以从前者开始修改模型，如下所示：

```
DATASET_DIR=./tfrecords
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32
```

请注意，除了训练脚本标志之外，还可能需要在`ssd_vgg_preprocessing.py`或/和网络参数（要素图层，锚点框......）中试验数据增强参数（随机裁剪，分辨率......） 在`ssd_vgg_300 / 512.py`中

此外，训练脚本可以与评估程序结合使用，以监控验证数据集上保存的检查点的性能。 为此，可以将训练和验证脚本传递给GPU内存上限，以便两者可以在同一台设备上并行运行。 如果某些GPU内存可用于评估脚本，则前者可以并行运行，如下所示：

```
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500
```

# 微调在ImageNet上训练的网络
人们还可以尝试建立一个基于标准架构（VGG，ResNet，Inception，...）的新SSD模型，并在其上设置multibox图层（具有特定的锚点，比率...）。 为此，您可以通过加载原始体系结构的权重来微调网络，并随机初始化网络的其余部分。 例如，在[VGG-16架构](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)的情况下，可以训练一个新的模型如下：

```
DATASET_DIR=./tfrecords
TRAIN_DIR=./log/
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```

因此，在前一个命令中，训练脚本随机初始化属于`checkpoint_exclude_scopes`的权重，并从检查点文件`vgg_16.ckpt`加载网络的其余部分。 请注意，我们还使用`trainable_scopes`参数指定首先训练新的SSD组件，并保持VGG网络的其余部分不变。 一旦网络收敛到良好的第一个结果（例如〜0.5 mAP），您可以对整个网络进行微调，如下所示：

```
DATASET_DIR=./tfrecords
TRAIN_DIR=./log_finetune/
CHECKPOINT_PATH=./log/model.ckpt-N
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.00001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=32
```

在[TF-Slim](https://github.com/tensorflow/models/tree/master/research/slim)模型页面上可以找到许多流行深层架构的预训练权重。

