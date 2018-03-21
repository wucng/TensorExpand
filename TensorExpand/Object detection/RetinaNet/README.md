参考：

- [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
- [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)
- [kuangliu/pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet)
- [CasiaFan/tensorflow_retinanet](https://github.com/CasiaFan/tensorflow_retinanet)


----------
# Keras实现RetinaNet对象检测

Keras实现RetinaNet物体检测，如林崇益，Priya Goyal，Ross Girshick，Kaiming He和PiotrDollár在[密集物体检测中的描述](https://arxiv.org/abs/1708.02002)。


----------

# Installation
- 1、克隆这个存储库。`git clone https://github.com/fizyr/keras-retinanet.git`
- 2、在存储库中，执行`pip3 install . --user`。 请注意，由于应该如何安装tensorflow，这个软件包并没有定义对tensorflow的依赖关系，因为它会尝试安装（至少在Arch Linux导致错误的安装）。 请确保`tensorflow `按照您的系统要求进行安装。 另外，确保安装`Keras` 2.1.3或更高版本。
- 3、或者，如果想 训练/测试MS COCO数据集，请安装`pycocotools`，运行`pip3 install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`。

python3 与windows安装pycocotools[参考这里](http://blog.csdn.net/wc781708249/article/details/79438972#t4)

----------
# Training
`keras-retinanet`可以使用[这个](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py)脚本来训练。 请注意，训练脚本使用相对导入，因为它位于`keras_retinanet`包内。 如果您想调整脚本以供您在此存储库之外使用，则需要将其切换为使用绝对导入。

如果您正确安装了`keras-retinanet`，则train 脚本将作为`retinanet-train`安装。 但是，如果您对keras-retinanet存储库进行本地修改，则应直接从存储库运行脚本。 这将确保您的本地更改将被train 脚本使用。

默认主干是`resnet50`。 您可以在运行脚本中使用`--backbone=xxx`参数来更改此参数。 xxx可以是resnet模型（`resnet50`，`resnet101`，`resnet152`）或`mobilenet`模型（`mobilenet128_1.0`，`mobilenet128_0.75`，`mobilenet160_1.0`等）的骨干之一。 不同的选项由每个模型在其相应的Python脚本（`resnet.py`，`mobilenet.py`等）中定义。

# Usage
- 对于[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)训练，请运行：

```python
# Running directly from the repository:
keras_retinanet/bin/train.py pascal /path/to/VOCdevkit/VOC2007

# Using the installed script:
retinanet-train pascal /path/to/VOCdevkit/VOC2007
```
- 训练[MS COCO](http://cocodataset.org/#home), 运行:

```python
# Running directly from the repository:
keras_retinanet/bin/train.py coco /path/to/MS/COCO

# Using the installed script:
retinanet-train coco /path/to/MS/COCO
```
预训练的MS COCO模型可以在[这里](https://github.com/fizyr/keras-retinanet/releases/download/0.2/resnet50_coco_best_v2.0.1.h5)下载。 使用椰子树的结果如下所示（注意：根据论文，这种配置应该达到0.357的mAP）。

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.368
```

- 训练 [OID](https://github.com/openimages/dataset)，运行

```python
# Running directly from the repository:
keras_retinanet/bin/train.py oid /path/to/OID

# Using the installed script:
retinanet-train oid /path/to/OID

# You can also specify a list of labels if you want to train on a subset
# by adding the argument 'labels_filter':
keras_retinanet/bin/train.py oid /path/to/OID --labels_filter=Helmet,Tree
```

- 训练[KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php)，运行

```python
# Running directly from the repository:
keras_retinanet/bin/train.py kitti /path/to/KITTI

# Using the installed script:
retinanet-train kitti /path/to/KITTI

If you want to prepare the dataset you can use the following script:
https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/prepare_kitti_data.py
```
- 对于[**自定义数据集**]的训练，可以使用CSV文件作为传递数据的方式。 有关这些CSV文件格式的更多详情，请参阅下文。 要使用您的CSV进行训练，请运行：

```python
# Running directly from the repository:
keras_retinanet/bin/train.py csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes

# Using the installed script:
retinanet-train csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes
```

通常，在您自己的数据集上进行的训练步骤如下：

- 1、通过调用实例`keras_retinanet.models.resnet50_retinanet`并编译它来创建一个模型。 经验上，下面的编译参数已经被发现运行良好：

```python
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
```
- 2、创建用于训练和测试数据的生成器（示例显示在[keras_retinanet.preprocessing.PascalVocGenerator](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)中）。

- 3、使用`model.fit_generator`开始训练。

# Testing
[本笔记本](https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb)中可以看到一个测试网络的例子。 一般来说，输出可以从网络中检索如下：

```python
_, _, detections = model.predict_on_batch(inputs)
```
如果检测结果为检测结果，shaped `(None, None, 4 + num_classes)`（对于（(x1, y1, x2, y2, cls1, cls2, ...)）。

加载模型可以通过以下方式完成：

```python
from keras_retinanet.models.resnet import custom_objects
model = keras.models.load_model('/path/to/model.h5', custom_objects=custom_objects)
```
对于形状为1000x800x3的图像，NVIDIA Pascal Titan X的执行时间大约为75毫秒。

# CSV datasets
`CSVGenerator`提供了一种简单的方法来定义您自己的数据集。 它使用两个CSV文件：一个文件包含annotations ，另一个文件包含类名到ID映射。

# Annotations format
带annotations 的CSV文件应每行对应一个注释（annotation ）。 带有多个边界框的图像应该为每个边界框使用一行。 请注意，像素值的索引从0开始。每行的预期格式为：

```python
path/to/image.jpg,x1,y1,x2,y2,class_name
path/to/image.jpg,x1,y1,x2,y2,class_name
# 如果图片路径一样，表示同一张图片，不同的对象
```
某些图像可能不包含任何标记的对象。 要将这些图像作为反面示例添加到数据集中，请添加一个注释，其中x1，y1，x2，y2和class_name全部为空：（相当于背景图片）

```python
path/to/image.jpg,,,,,
```
一个完整的例子：

```python
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat # 02图片第一个对象
/data/imgs/img_002.jpg,22,5,89,84,bird # 02图片第二个对象
/data/imgs/img_003.jpg,,,,, # 背景图片，没有任何要检查的对象
```
这定义了一个包含3个图像的数据集。 `img_001.jpg`包含一头母牛。 `img_002.jpg`包含一只猫和一只鸟。 `img_003.jpg`不包含有趣的对象/动物。

# Class mapping format
类映射文件的类名应该每行对应一个映射。 每行应使用以下格式：

```python
class_name,id
```
对于类的索引从0开始。不要包含背景类，因为它是隐式的。

例如：

```python
cow,0
cat,1
bird,2
# 因为这里的背景对应的类别为 空 ，
# 而mask RCNN 与 FCN（RFCN）都是使用0来表示背景
```

# Debugging
创建您自己的数据集并不总是在框中起作用。 有一个`debug.py`工具可帮助查找最常见的错误。

特别有用的是`--annotations`标志，它在你的数据集的图像上显示你的注释。 当没有可用的锚点时，注释以绿色着色，并有可用的锚点并以红色着色。 如果注释没有可用的锚点，则意味着它不会对训练作出贡献。 少量注释显示为红色是正常的，但如果大多数注释或全部注释都是红色，则需要注意。 最常见的问题是注释太小或太奇怪（伸出）。

# Results
## MS COCO
### Status
下面显示了使用`keras-retinanet`的示例输出图像。

![这里写图片描述](https://github.com/delftrobotics/keras-retinanet/raw/master/images/coco2.png)

![这里写图片描述](https://github.com/delftrobotics/keras-retinanet/raw/master/images/coco3.png)



# Notes

- 该存储库需要Keras 2.1.3或以上。
- 该存储库使用OpenCV 3.4进行[测试](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml)。
- 该存储库使用Python 2.7和3.6进行[测试](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml)。
- 警告，如`UserWarning: Output "non_maximum_suppression_1" missing from loss dictionary.` 可以安全地被忽略。 这些警告表明没有损失连接到这些输出，但它们旨在作为用户网络的输出（即产生的网络检测）而不是丢失输出。

对这个项目的贡献是值得欢迎的。

# Discussions
请随时加入`#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/)频道进行讨论和提问。
