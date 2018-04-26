参考：

- [davidsandberg/facenet](https://github.com/davidsandberg/facenet)


----------

本页介绍如何在自己的数据集上训练自己的分类器。 这里假定你已经遵循例如 该指南在[LFW上进行验证](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw)以安装依赖关系，克隆FaceNet库，设置python路径等并对齐LFW数据集（至少对于LFW实验）。 在下面的例子中，使用了冷冻模型20170216-091149。 使用冻结图显着加快了模型的加载速度。

---------------------------------------
# Train a classifier on LFW
对于这个实验，我们使用LFW图像的子集来训练分类器。 LFW数据集分为训练和测试集。 然后加载预训练模型，然后使用此模型为选定图像生成特征。 预训练模型通常在更大的数据集上进行训练以提供良好的性能（本例中为MS-Celeb-1M数据集的一个子集）。

- 将数据集分解为训练集和测试集
- 加载预训练模型进行特征提取
- 计算数据集中图像的嵌入
- mode=TRAIN: 
	- 使用来自数据集的train部分的嵌入来训练分类器
	- 将训练好的分类模型保存为python pickle
- mode=CLASSIFY:
	- 加载分类模型
	- 使用来自数据集测试部分的嵌入来测试分类器


### 在数据集的训练集部分上训练分类器的步骤如下：

```python
CUDA_VISIBLE_DEVICES=1 python3 src/classifier.py TRAIN ./datasets/lfw/lfw_mtcnnpy_160 ../20170512-110547/20170512-110547.pb ./models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
```

训练的输出如下所示：

```
Number of classes: 19
Number of images: 665
Loading feature extraction model
Model filename: ../20170512-110547/20170512-110547.pb
Calculating features for images
Training classifier
Saved classifier model to file "./models/lfw_classifier.pkl"
```



### 训练好的分类器可以稍后用于使用测试集进行分类：

```python
CUDA_VISIBLE_DEVICES=1 python3 src/classifier.py CLASSIFY ./datasets/lfw/lfw_mtcnnpy_160 ../20170512-110547/20170512-110547.pb ./models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
```
这里使用数据集的测试集部分进行分类，并显示分类结果和分类概率。 该子集的分类准确度为〜0.98。

```
Number of classes: 19
Number of images: 1202
Loading feature extraction model
Model filename: /home/david/models/export/model-20170216-091149.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "/home/david/lfw_classifier.pkl"
   0  Ariel Sharon: 0.583
   1  Ariel Sharon: 0.611
   2  Ariel Sharon: 0.670
...
...
...
1198  Vladimir Putin: 0.588
1199  Vladimir Putin: 0.623
1200  Vladimir Putin: 0.566
1201  Vladimir Putin: 0.651
Accuracy: 0.978
```

# Train a classifier on your own dataset
所以也许你想自动分类你的私人照片收藏。 或者你有一个安全摄像头，你想自动识别你的家庭成员。 那么很可能你想在你自己的数据集上训练一个分类器。 在这种情况下，`classifier.py`程序也可以用于此目的。 我通过复制LFW数据集的子集创建了自己的训练集和测试数据集。 在这个例子中，每个class的前5张图像被用于训练，接下来的5张图像被用于测试。

使用的类是：

```
    Ariel_Sharon
    Arnold_Schwarzenegger
    Colin_Powell
    Donald_Rumsfeld
    George_W_Bush
    Gerhard_Schroeder
    Hugo_Chavez
    Jacques_Chirac
    Tony_Blair
    Vladimir_Putin
```

### 分类器的训练与以前类似：

```
CUDA_VISIBLE_DEVICES=1 python3 src/classifier.py TRAIN ~/datasets/my_dataset/train/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000
```
分类器的训练需要几秒钟（在加载预先训练的模型之后），输出如下所示。 由于这是一个非常简单的数据集，因此准确性非常好。

```
Number of classes: 10
Number of images: 50
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Training classifier
Saved classifier model to file "/home/david/models/my_classifier.pkl"
```
这就是包含测试集的目录的组织方式：

```
/home/david/datasets/my_dataset/test
├── Ariel_Sharon
│   ├── Ariel_Sharon_0006.png
│   ├── Ariel_Sharon_0007.png
│   ├── Ariel_Sharon_0008.png
│   ├── Ariel_Sharon_0009.png
│   └── Ariel_Sharon_0010.png
├── Arnold_Schwarzenegger
│   ├── Arnold_Schwarzenegger_0006.png
│   ├── Arnold_Schwarzenegger_0007.png
│   ├── Arnold_Schwarzenegger_0008.png
│   ├── Arnold_Schwarzenegger_0009.png
│   └── Arnold_Schwarzenegger_0010.png
├── Colin_Powell
│   ├── Colin_Powell_0006.png
│   ├── Colin_Powell_0007.png
...
...
```

### 测试集上的分类可以使用以下方式运行：

```
CUDA_VISIBLE_DEVICES=1 python3 src/classifier.py CLASSIFY ~/datasets/my_dataset/test/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000
```

```
Number of classes: 10
Number of images: 50
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "/home/david/models/my_classifier.pkl"
   0  Ariel Sharon: 0.452
   1  Ariel Sharon: 0.376
   2  Ariel Sharon: 0.426
...
...
...
  47  Vladimir Putin: 0.418
  48  Vladimir Putin: 0.453
  49  Vladimir Putin: 0.378
Accuracy: 1.000
```
本代码旨在为如何使用人脸识别器提供一些启发和想法，但它本身并不是一个有用的应用程序。 实际应用中可能需要的一些额外的东西包括：

- 在人脸检测和分类管道中包含人脸检测
- 使用分类概率的阈值来找到未知的人，而不是仅仅使用具有最高概率的类

