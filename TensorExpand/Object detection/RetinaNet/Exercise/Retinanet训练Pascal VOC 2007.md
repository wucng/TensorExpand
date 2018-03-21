参考：

- [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
- [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)
- [kuangliu/pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet)
- [CasiaFan/tensorflow_retinanet](https://github.com/CasiaFan/tensorflow_retinanet)


----------
# 安装Retinanet
- 1、克隆这个存储库。`git clone https://github.com/fizyr/keras-retinanet.git`
- 2、在存储库中，执行`pip3 install . --user`。 请注意，由于应该如何安装tensorflow，这个软件包并没有定义对tensorflow的依赖关系，因为它会尝试安装（至少在Arch Linux导致错误的安装）。 请确保`tensorflow `按照您的系统要求进行安装。 另外，确保安装`Keras` 2.1.3或更高版本。
- 3、或者，如果想 训练/测试MS COCO数据集，请安装`pycocotools`，运行`pip3 install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`。

python3 与windows安装pycocotools[参考这里](http://blog.csdn.net/wc781708249/article/details/79438972#t4)


----------
# Pascal VOC 2007

- Pascal VOC 2007数据下载与数据预览参考[这里](http://blog.csdn.net/wc781708249/article/details/79615210#t0)


- VOC2007目录结构

```python
<VOC2007>
|———  Annotations
|       └─  *.xml # 存放图片的 类别与边框
|
|———  JPEGImages
|       └─ *.jpg # 存放图片
|
|———  SegmentationClass # 类别掩膜
|
|——— SegmentationObject # 对象掩膜
|
|___ ImageSets
```
`Retinanet`只需要 `Annotations`(存放图片的 类别与边框)与 `JPEGImages`（存放图片）和`ImageSets`中的`Main`文件夹，其他的都使用不到（可以删除）

## train
先去[这里](https://github.com/fizyr/keras-retinanet/releases/)下需要的预训练的模型存放在`./keras_retinanet/snapshots/`

```python
# train

python3 keras_retinanet/bin/train.py pascal /path/to/VOCdevkit/VOC2007

# 使用 --backbone=xxx 选择网络结构，默认是resnet50

# xxx可以是resnet模型（`resnet50`，`resnet101`，`resnet152`）
# 或`mobilenet`模型（`mobilenet128_1.0`，`mobilenet128_0.75`，`mobilenet160_1.0`等）

# 也可以使用models目录下的 resnet.py，mobilenet.py等来自定义网络
```
## test
参考：[ResNet50RetinaNet.ipynb](https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb)

- 将`labels_to_names`换成Pascal VOC 2007对应的类别
参考 http://blog.csdn.net/wc781708249/article/details/79624990#t9

```python
class_name=['diningtable', 'person', 'bottle', 'boat', 'train', 'bird', 'dog', 'cat', 'tvmonitor', 'cow', 'car', 'sofa', 'horse', 'chair', 'pottedplant', 'bicycle', 'motorbike', 'aeroplane', 'sheep', 'bus']
class_name=sorted(class_name)
labels_to_names=dict(zip(range(len(class_name)),class_name))
```
- `model_path` 改成保存的模型路径 如：`resnet50_pascal_01.h5`（snapshots存放模型）


# MS COCO数据
这里直接将`Pascal VOC 2007`转成`MS COCO`格式来训练

参考：[Pascal VOC转COCO数据](http://blog.csdn.net/wc781708249/article/details/79615210) 

- 在`./keras-retinanet`建一个目录用来存放COCO格式数据`mkdir COCO`

- 将[Pascal VOC转COCO数据](http://blog.csdn.net/wc781708249/article/details/79615210) 生成的`new.json` 复制`instances_train2017.json`、`instances_val2017.json` 存放在`./keras-retinanet/COCO/annotations/`

- 将`JPEGImages`复制成`train2017`、`val2017` 放在`./keras-retinanet/COCO/images`

- 最终`COCO`目录如下：

```python
<COCO>
|———— annotations
|         |—— instances_train2017.json
|         └  instances_val2017.json
|
|____ images
         |—— train2017
                └─ *.jpg
         |___ val2017
                └─ *.jpg
```

## train
先去[这里](https://github.com/fizyr/keras-retinanet/releases/)下需要的预训练的模型存放在`./keras_retinanet/snapshots/`
```python
# train

python3 keras_retinanet/bin/train.py coco /path/to/MS/COCO

# 使用 --backbone=xxx 选择网络结构，默认是resnet50

# xxx可以是resnet模型（`resnet50`，`resnet101`，`resnet152`）
# 或`mobilenet`模型（`mobilenet128_1.0`，`mobilenet128_0.75`，`mobilenet160_1.0`等）

# 也可以使用models目录下的 resnet.py，mobilenet.py等来自定义网络
```

```python
# 为了演示，只运行200步
python3 keras_retinanet/bin/train.py --steps 200 coco ./COCO
```

## test
参考：[ResNet50RetinaNet.ipynb](https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb)

- 将`labels_to_names`换成Pascal VOC 2007对应的类别
参考 http://blog.csdn.net/wc781708249/article/details/79624990#t9

```python
# 需根据JSON文件字段 categories 对类别标记的id顺序排列
class_name=['diningtable', 'person', 'bottle', 'boat', 'train', 'bird', 'dog', 'cat', 'tvmonitor', 'cow', 'car', 'sofa', 'horse', 'chair', 'pottedplant', 'bicycle', 'motorbike', 'aeroplane', 'sheep', 'bus']
labels_to_names=dict(zip(range(len(class_name)),class_name)) # Retinanet要求类别从0开始，背景默认为空，不用显示标记，Mask RCNN 、FCN、RFCN 类别从1开始，0默认为背景
```
- `model_path` 改成保存的模型路径,如：`./snapshots/resnet50_coco_01.h5`


# CSV数据
这里将`Pascal VOC 2007`转成`CSV`格式训练

- 在`./keras-retinanet` 新建一个目录`mkdir CSV`
- 在CSV目录中放入`annotations.csv` 每行对应一个对象，格式如下：

```python
path/to/image.jpg,x1,y1,x2,y2,class_name

/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat # 02图片第一个对象
/data/imgs/img_002.jpg,22,5,89,84,bird # 02图片第二个对象
/data/imgs/img_003.jpg,,,,, # 背景图片，没有任何要检查的对象
```
这定义了一个包含3个图像的数据集。 img_001.jpg包含一头母牛。 img_002.jpg包含一只猫和一只鸟。 img_003.jpg不包含有趣的对象/动物。

- 在CSV目录 放入`classes.csv` 类名与id对应，索引从0开始。不要包含背景类，因为它是隐式的。具体格式如下：

```python
class_name,id

cow,0
cat,1
bird,2
# 因为这里的背景对应的类别为 空 ，
# 而mask RCNN 与 FCN（RFCN）都是使用0来表示背景
```
- CSV目录结构如下：
```python
<CSV>
|———— annotations.csv # 必须
|———— classes.csv # 必须
|
|____ JPEGImages  # （可选），这样 annotations.csv可以使用图片的相对路径       
         └─ *.jpg
```

## Pascal VOC 2007转成CSV格式

完整程序参考：PascalVOC2CSV.py
```python
for p in fp:
    if '<filename>' in p:
        self.filen_ame = p.split('>')[1].split('<')[0]

    if '<object>' in p:
        # 类别
        d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
        self.supercategory = d[0]
        if self.supercategory not in self.label:
            self.label.append(self.supercategory)

        # 边界框
        x1 = int(d[-4]);
        y1 = int(d[-3]);
        x2 = int(d[-2]);
        y2 = int(d[-1])

        self.annotations.append([os.path.join('JPEGImages', self.filen_ame), x1, y1, x2, y2, self.supercategory])
```
## train

```python
# Running directly from the repository:
python3 keras_retinanet/bin/train.py csv ./CSV/annotations.csv ./CSV/classes.csv

# 使用 --backbone=xxx 选择网络结构，默认是resnet50

# xxx可以是resnet模型（`resnet50`，`resnet101`，`resnet152`）
# 或`mobilenet`模型（`mobilenet128_1.0`，`mobilenet128_0.75`，`mobilenet160_1.0`等）

# 也可以使用models目录下的 resnet.py，mobilenet.py等来自定义网络
```

## test
参考：[ResNet50RetinaNet.ipynb](https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb)

- 将`labels_to_names`换成Pascal VOC 2007对应的类别
参考 http://blog.csdn.net/wc781708249/article/details/79624990#t9

```python
# 需根据classes.csv文件中类别名与id对应顺序排列

class_name=['diningtable', 'person', 'bottle', 'boat', 'train', 'bird', 'dog', 'cat', 'tvmonitor', 'cow', 'car', 'sofa', 'horse', 'chair', 'pottedplant', 'bicycle', 'motorbike', 'aeroplane', 'sheep', 'bus']
class_name=sorted(class_name)
labels_to_names=dict(zip(range(len(class_name)),class_name)) # Retinanet要求类别从0开始，背景默认为空，不用显示标记，Mask RCNN 、FCN、RFCN 类别从1开始，0默认为背景
```
- `model_path` 改成保存的模型路径,如：`./snapshots/resnet50_csv_01.h5`

