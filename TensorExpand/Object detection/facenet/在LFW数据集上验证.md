
参考：

- [davidsandberg/facenet](https://github.com/davidsandberg/facenet)
- https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw

----------
# 1、 Install dependencies

在下面的描述中，假设

1）Tensorflow已经[安装](https://github.com/davidsandberg/facenet/wiki#1-install-tensorflow)，并且

2）Facenet[回购](https://github.com/davidsandberg/facenet)已被克隆（`git clone https://github.com/davidsandberg/facenet.git`），并且

3）所需的[python模块](https://github.com/davidsandberg/facenet/blob/master/requirements.txt)已安装。

--------
# 2、Download the LFW dataset
## 1）从[这里](vis-www.cs.umass.edu/lfw/lfw.tgz)下载未对齐的图像
在这个例子中，文件被下载到`~/Downloads`。
## 2) 将未对齐的图像解压缩到本地存储
假设你有一个用于存储数据集的目录`~/datasets`

```python
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C lfw/raw --strip-components=1
```
# 3、Set the python path
将环境变量`PYTHONPATH`设置为指向克隆的repo的`src`目录。 这通常是这样做的

```
export PYTHONPATH=[...]/facenet/src
```
`[...]`应该将其替换为克隆的facenet仓库所在的目录。

# 4.对齐LFW数据集
可以使用`align`模块中的`align_dataset_mtcnn`完成LFW数据集的对齐。

LFW数据集的对齐方式如下所示(将以下语句写入shell脚本执行 如：`sh align.sh`)：

```python
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
~/datasets/lfw/raw \
~/datasets/lfw/lfw_mtcnnpy_160 \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
& done
```
与面部检测器给出的边界框相比，参数`margin`控制应该裁剪多少更宽的对齐图像。 图像大小为160像素裁剪32个像素;图像大小为182裁剪44像素，这是用于下面的模型的训练的图像大小。

**注**：如果使用的不是最新的tensorflow，如:1.4.1版 运行时需将`./facenet/src/align/detect_face.py`中210、212行的`keepdims` 换成`keep_dims` 


# 5、Download pre-trained model (optional)
如果您没有自己想要测试的训练有素的模型，并且简单的前进方法是下载预先训练好的模型以运行测试。 一个这样的模型可以在[这里](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)找到。 下载并提取模型并放入您最喜欢的模型目录（在本例中我们使用`~/models/facenet/`）。 提取存档后，应该有一个新的文件夹`20180402-114759`与内容。

```
20180402-114759.pb
model-20180402-114759.ckpt-275.data-00000-of-00001
model-20180402-114759.ckpt-275.index
model-20180402-114759.meta
```
# 6. Run the test
测试运行`validate_on_lfw`：

```python
python src/validate_on_lfw.py \
~/datasets/lfw/lfw_mtcnnpy_160 \
~/models/facenet/20180402-114759 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization
```

这会

a）加载模型，

b）用图像对加载和解析文本文件，

c）计算测试集中所有图像（以及它们的水平翻转版本）的嵌入，

d）计算准确度，验证率（@ FAR = -10e-3），曲线下面积（AUC）
和等误差率（EER）性能指标。

测试的典型输出如下所示：

```
Model directory: /home/david/models/20180402-114759/
Metagraph file: model-20180402-114759.meta
Checkpoint file: model-20180402-114759.ckpt-275
Runnning forward pass on LFW images
........................
Accuracy: 0.99650+-0.00252
Validation rate: 0.98367+-0.00948 @ FAR=0.00100
Area Under Curve (AUC): 1.000
Equal Error Rate (EER): 0.004
```
