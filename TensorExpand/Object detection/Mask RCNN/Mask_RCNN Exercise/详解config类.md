基础配置类。 对于自定义配置，请创建一个从这个继承的子类并重写属性那需要改变。


----------


```python
# 命名配置。 例如，'COCO'，'Experiment 3'，...等。如果你的代码需要根据哪一个做不同的事情，那么这很有用实验正在运行。

NAME = None # 子类中覆盖
# 如 NAME='shapes'
```

----------
```python
# 输入图像resing
# 图像被调整大小，使得最小的边> = IMAGE_MIN_DIM和
# 最长边是<= IMAGE_MAX_DIM。 如果两种情况都不能
# 一起满足IMAGE_MAX_DIM被执行。
IMAGE_MIN_DIM = 800
IMAGE_MAX_DIM = 1024

# 例如
IMAGE_MIN_DIM =128
IMAGE_MAX_DIM =128
```

----------

```python
# 使用的GPU数量。 对于CPU训练，请使用1
GPU_COUNT = 1
```


----------

```python
# 在每个GPU上训练的图像数量。 通常可以使用12GB的GPU处理1024x1024画面的2张图片。
# 根据您的GPU内存和图像大小进行调整。 使用最高您的GPU可以处理的编号，以获得最佳性能。
IMAGES_PER_GPU = 2

# 例如 4G GPU 128x128
# 4=128*128*n;12=1024*1024*2 ==>n=1024*1024*2*4//(128*128*12)=42
IMAGES_PER_GPU= int(1024*1024*4//(IMAGE_MAX_DIM*IMAGE_MAX_DIM*12)*(2/3))
```


----------

```python
# 每个epoch的训练步数这不需要与训练集的大小相匹配。Tensorboard更新保存在每个epoch的末尾，因此将其设置的数字越小意味着获得更频繁的TensorBoard更新。验证统计数据也会在每个epoch末期和他们计算可能需要一段时间，所以不要设置太小以免花费验证统计信息很多时间。
STEPS_PER_EPOCH = 1000

# num_images 图像数
batch_size=GPU_COUNT *IMAGES_PER_GPU
STEPS_PER_EPOCH =int(num_images/batch_size*(2/3))
```


----------

```python
# 在每个训练epoch结束时运行的验证步骤数量。更大的数字可以提高验证统计的准确性，但速度会变慢降低训练。
VALIDATION_STEPS = 50

# 例如
VALIDATION_STEPS=STEPS_PER_EPOCH//(1000//50)
```


----------

```python
# FPN金字塔每层的步幅。 这些值是基于Resnet101骨干网的。
BACKBONE_STRIDES = [4,8,16,32,64] # 与使用的神经网络相关，如果使用Resnet101则不变
```


----------

```python
# 分类类别的数量（包括背景）
NUM_CLASSES = 1 # 子类中覆盖 

# 如：
NUM_CLASSES = 1+3 # 必须包含一个背景（背景作为一个类别）
```


----------

```python
# 方形锚点的长度（以像素为单位）
RPN_ANCHOR_SCALES =（32,64,128,256,512） # 以输入图像1204x1024设置的，如果输入图像大小发生变化必须做相应的调整

# 例如
scale=1024//IMAGE_MAX_DIM
RPN_ANCHOR_SCALES = (32//scale, 64//scale, 128//scale, 256//scale, 512//scale)  # anchor side in pixels

```

----------

```python
# 每个cell处锚点的比率（宽度/高度）
# 值1表示方形锚点，0.5表示宽锚点
RPN_ANCHOR_RATIOS = [0.5,1,2] # 不变
```


----------

```python
# 锚定步伐
# 如果为1，则为骨干特征映射中的每个单元创建锚点。
# 如果是2，则为每个其他单元格创建锚点，依此类推。
RPN_ANCHOR_STRIDE = 1 # 不变
```


----------

```python
# 过滤RPN提议的非最大抑制阈值。
# 你可以在训练过程中减少这些以产生更多的提议。
RPN_NMS_THRESHOLD = 0.7  # 0.6
```


----------

```python
# 每张图片用于RPN训练的锚点数量
RPN_TRAIN_ANCHORS_PER_IMAGE = 256

# 例如
scale=1024//IMAGE_MAX_DIM
RPN_TRAIN_ANCHORS_PER_IMAGE =256//scale

```


----------

```python
# 投资回报率在非最大压缩后保持（训练和推理）
POST_NMS_ROIS_TRAINING = 2000 # 不变
POST_NMS_ROIS_INFERENCE = 1000 # 不变
```


----------

```python
# 如果启用，则将实例掩码调整为更小的大小以减小内存负载。 建议使用高分辨率图像时使用。
USE_MINI_MASK = True
MINI_MASK_SHAPE =（56,56）＃（高度，宽度）的迷你mask

# 如
scale=1024//IMAGE_MAX_DIM
MINI_MASK_SHAPE =(56//scale,56//scale)
```


----------


----------

```python
# 如果为True，则用零填充图像，使其成为（max_dim by max_dim）
IMAGE_PADDING = True # 当前，False选项不受支持  不变
```


----------

```python
# 图像均值（RGB）
MEAN_PIXEL = np.array（[123.7，116.8，103.9]）# 不变
# 所以输入的图像需保持像素值0~255，不需事先转成0.~1.,因为会自动做归一化处理
```


----------

```python
# 每个图像的分辨率/mask头的ROI数量
# mask RCNN文件使用512，但通常RPN不会生成
# 有足够的正建议来填补这一点并保持积极的位置：否定的
# 比例为1：3。 您可以通过调整来增加投标的数量RPN网管门限。
TRAIN_ROIS_PER_IMAGE = 200

# 例如
scale=1024//IMAGE_MIN_DIM
TRAIN_ROIS_PER_IMAGE = 200//scale
```


----------

```
# 用于训练分级器/mask头的正ROI百分比
ROI_POSITIVE_RATIO = 0.33 # 不变
```


----------

```
# Pooled ROIs
POOL_SIZE = 7
MASK_POOL_SIZE = 14
MASK_SHAPE = [28,28] # 不变
```


----------

```python
# 在一个图像中使用的地面实况实例的最大数量
MAX_GT_INSTANCES = 100 # 不变
```


----------

```python
# RPN和最终检测的边界框细化标准差。
RPN_BBOX_STD_DEV = np.array（[0.1,0.1,0.2,0.2]） # 不变
BBOX_STD_DEV = np.array（[0.1,0.1,0.2,0.2]） # 不变
```


----------

```python
# 最大检测次数
DETECTION_MAX_INSTANCES = 100

# 例如
scale=1024//IMAGE_MIN_DIM
DETECTION_MAX_INSTANCES =100*scale*2//3
```


----------

```python
# 接受检测到的实例的最小概率值低于此阈值的ROI将被跳过
DETECTION_MIN_CONFIDENCE = 0.7

# 例如
DETECTION_MIN_CONFIDENCE = 0.6
```


----------

```python
# 检测的非最大抑制阈值
DETECTION_NMS_THRESHOLD = 0.3
```


----------

```python
# 学习效率和动力
# Mask RCNN文件使用lr = 0.02，但在TensorFlow上引起权重爆炸。 可能由于优化器的差异执行。
LEARNING_RATE = 0.001
LEARNING_MOMENTUM = 0.9
```


----------

```python
# 重量衰减正则化
WEIGHT_DECAY = 0.0001
```


----------

```python
# 使用RPN投资回报率或外部生成的投资回报率进行训练
# 在大多数情况下保持真实。 如果您想训练，请设置为False
# 由代码生成的ROI分支，而不是来自ROI的ROIRPN。 例如，调试分类器头，而不必训练RPN。
USE_RPN_ROIS = True
```


----------


