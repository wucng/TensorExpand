参考：

- [使用Keras和Tensorflow设置和安装Mask RCNN](http://blog.csdn.net/wc781708249/article/details/79438972) 
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN) for object detection and instance segmentation on Keras and TensorFlow


----------
[toc]

# Mask R-CNN为对象检测和分割
这是Python 3，Keras和TensorFlow上Mask R-CNN的实现。 该模型为图像中的每个对象实例生成边界框和分割掩码。 它基于特征金字塔网络（FPN）和ResNet101主干网。

该存储库包括：

- 在FPN和ResNet101上构建的Mask R-CNN的源代码。
- MS COCO的训练代码
- 预先训练的MS COCO权重
- Jupyter notebooks在每一步都可以看到检测管道
- 用于多GPU训练的ParallelModel类
- MS COCO指标评估（AP）
- 训练您自己的数据集的示例


----------
该代码被记录并设计为易于扩展。 如果您在研究中使用它，请考虑引用此存储库。 如果您从事3D视觉工作，您可能会发现我们最近发布的[Matterport3D](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/)数据集也很有用。 该数据集是由我们的客户拍摄的3D重建空间创建的，这些客户同意将这些空间公开供学术使用。 你可以在[这里](https://matterport.com/gallery/)看到更多的例子。

# 使用此模型的项目

如果将此模型扩展到其他数据集或构建使用它的项目，我们很乐意听取您的意见。

- [图像到OSM](https://github.com/jremillard/images-to-osm)：使用TensorFlow，Bing和OSM查找卫星图像中的特征。 目标是通过添加高质量的棒球，足球，网球，足球和篮球场来改进OpenStreetMap。
- [4K视频演示](https://www.youtube.com/watch?v=OOT3UIXZztE)：Karol Majek的4K视频演示。

![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/4k_video.gif)

# 入门
- [demo.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/demo.ipynb)是最简单的开始。 它展示了一个使用MS COCO预先训练的模型来分割自己图像中的对象的例子。 它包括在**任意图像**上运行对象检测和实例分割的代码。

- [train_shapes.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/train_shapes.ipynb)显示了如何在您自己的数据集上训练Mask R-CNN。 这款notebook引入了一个玩具数据集（Shapes）来演示新数据集的训练。

- （[model.py](https://github.com/matterport/Mask_RCNN/blob/master/model.py)，[utils.py](https://github.com/matterport/Mask_RCNN/blob/master/utils.py)，[config.py](https://github.com/matterport/Mask_RCNN/blob/master/config.py)）：这些文件包含主Mask RCNN实现。

- [inspect_data.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/inspect_data.ipynb)。 该notebook可视化不同的预处理步骤以准备训练数据。

- [inspect_model.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/inspect_model.ipynb)这个notebook深入到执行检测和分割对象的步骤。 它提供了管道每一步的可视化。

[inspect_weights.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/inspect_weights.ipynb)这款notebook检查训练好的模型的权重并查找异常和奇怪的模式。

# 逐步检测
为了帮助debugging和理解模型，有3个笔记本（inspect_data.ipynb，inspect_model.ipynb，inspect_weights.ipynb）提供了大量的可视化，并允许逐步运行模型来检查每个点的输出。 这里有一些例子：

## 1.锚点分类和过滤
可视化第一阶段区域提案网络的每一步，并显示正面和负面的锚点以及锚点框架细化

![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/detection_anchors.png)

## 2.边界框细化
这是第二阶段最终检测框（虚线）和应用于它们的细化（实线）的示例。
![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/detection_refinement.png)

## 3.Mask生成

生成的mask的示例。 然后将它们缩放并放置在正确位置的图像上。
![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/detection_masks.png)

## 4.层次激活

通常检查不同层的激活以寻找麻烦迹象（全零或随机噪声）是有用的。
![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/detection_activations.png)

## 5.权重直方图
另一个有用的调试工具是检查权重直方图。 这些都包含在inspect_weights.ipynb笔记本中。
![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/detection_histograms.png)

## 6.登录到TensorBoard

TensorBoard是另一个伟大的调试和可视化工具。 该模型配置为记录损失并在每个时期结束时保存权重。
![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/detection_tensorboard.png)

## 7.将不同的部分组合成最终结果
![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/detection_final.png)

# MS COCO训练
我们为MS COCO提供预先训练的权重，使其更容易启动。 您可以使用这些权重作为起点来在网络上训练自己的变体。 训练和评估代码在coco.py中。 您可以在Jupyter notebook导入此模块（请参阅提供的笔记本中的示例），也可以直接从命令行运行它，如下所示：

```python
# Train a new model starting from pre-trained COCO weights
python3 coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 coco.py train --dataset=/path/to/coco/ --model=last
```

您还可以运行COCO评估代码：

```python
# Run COCO evaluation on the last trained model
python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
```

训练计划，学习率和其他参数应在coco.py中设置。

# 训练自己的数据集
为了在您自己的数据集上训练模型，您需要对两类进行分类：

`Config`此类包含默认配置。 对它进行子类化并修改需要更改的属性。

`Dataset`此类提供了一种使用任何数据集的一致方法。 它允许您使用新的数据集进行训练，而无需更改模型的代码。 它还支持同时加载多个数据集，如果要检测的对象在一个数据集中不全部可用，这非常有用。

`Dataset`类本身就是基类。 要使用它，创建一个新的类继承它，并添加特定于您的数据集的函数。 请参阅`utils.py`中的基础`Dataset`类以及在train_shapes.ipynb和coco.py中进行扩展的示例。

# 与官方文件的差异

这个实现大部分遵循Mask RCNN论文，但是有些情况下我们偏离了代码简单性和泛化性。这些是我们意识到的一些差异。如果您遇到其他差异，请告诉我们。

- **图像大小调整**：为了支持每批次训练多幅图像，我们将所有图像的大小调整为相同大小。例如，MS COCO上的1024x1024图像。我们保留宽高比，所以如果图像不是正方形，我们用零填充它。在这篇论文中，调整大小的方法是：最小边是800px，最大边是1000px。

- **边界框**：一些数据集提供边界框，一些仅提供掩码。为了支持对多个数据集进行训练，我们选择忽略数据集附带的边界框，然后动态生成它们。我们选取封装所有像素的最小盒子作为边界框。这简化了实现，并且还使得应用某些图像增强变得容易，否则将很难应用于边界框，例如图像旋转。

	为了验证这种方法，我们将计算出的边界框与COCO数据集提供的边界框进行了比较。我们发现〜2％的边界框相差1px或更多，〜0.05％相差5px或更多，只有0.01％相差10px或更多。

- **学习率**：本文使用的学习率为0.02，但我们发现它太高，并且经常导致权重爆炸，尤其是使用小批量时。这可能与Caffe和TensorFlow计算梯度之间的差异（批次与GPU之间的平均值与平均值之间的差异）有关。或者，也许官方模型使用渐变裁剪来避免这个问题。我们确实使用渐变裁剪，但不要太积极。我们发现无论如何，较小的学习速度会更快地收敛，所以我们就这样做。

- **Anchor Strides**：金字塔的最低级别相对于图像的跨度为4px，所以锚点每隔4个像素间隔创建一次。为了减少计算和内存负载，我们采用了2的锚定步长，它将锚点数量减少了4，并且对精度没有显着影响。

# 贡献

欢迎对这个资料库作出贡献。 你可以贡献的东西的例子：

- 速度提升。 就像在TensorFlow或Cython中重写一些Python代码一样。
- 在其他数据集上进行培训。
- 精度提高。
- 可视化和示例。

您也可以加入我们的团队，帮助我们构建更多像这样的项目。

# 要求

- Python 3.4+
- TensorFlow 1.3+
- Keras 2.0.8+
- Jupyter Notebook
- Numpy，skimage，scipy，Pillow，cython，h5py

## MS COCO要求：

要在MS COCO上进行训练或测试，您还需要：

- pycocotools（安装说明如下）
- [MS COCO数据集](http://cocodataset.org/#home)
- 下载5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)和35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)子集。 [faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md)实施细节。

如果您使用Docker，则代码已验证可在此[Docker容器](https://hub.docker.com/r/waleedka/modern-deep-learning/)上使用。

# 安装

1、克隆这个存储库

2、从[发布页面](https://github.com/matterport/Mask_RCNN/releases)下载预先训练好的COCO权重（mask_rcnn_coco.h5）。

3、（可选）在MS COCO上训练或测试从其中一个回购站安装`pycocotools`。 他们是原始pycocotools for Python3和Windows的补丁（官方repo似乎不再活跃）。

- Linux: https://github.com/waleedka/coco
- Windows：https://github.com/philferriere/cocoapi。 您必须在您的路径上安装Visual C ++ 2015构建工具（有关更多详细信息，请参阅回购）

# 更多示例

![](https://github.com/matterport/Mask_RCNN/raw/master/assets/sheep.png)

![这里写图片描述](https://github.com/matterport/Mask_RCNN/raw/master/assets/donuts.png)
