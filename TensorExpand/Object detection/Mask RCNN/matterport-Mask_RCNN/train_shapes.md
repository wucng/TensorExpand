参考：

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [使用Keras和Tensorflow设置和安装Mask RCNN](http://blog.csdn.net/wc781708249/article/details/79438972)


----------
[train_shapes.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/train_shapes.ipynb)显示了如何在您自己的数据集上训练Mask R-CNN。 这款笔记本引入了一个玩具数据集（Shapes）来演示新数据集的训练。
# Mask R-CNN - Train on Shapes Dataset
这个notebook展示了如何在你自己的数据集上训练Mask R-CNN。 为了简单起见，我们使用了可以快速训练的形状（正方形，三角形和圆形）的综合数据集。 不过，你仍然需要一个GPU，因为网络主干是一个Resnet101，这对于在CPU上训练来说太慢了。 在GPU上，您可以在几分钟内开始好的结果，并在不到一个小时的时间内获得好的结果。

Shapes数据集的代码包含在下面。 它可以即时生成图像，因此不需要下载任何数据。 它可以生成任何尺寸的图像，所以我们选择一个较小的图像大小来加快训练速度。

```python
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

# %matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
```

# Configurations

```python
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    玩具形状数据集的训练配置。
     派生自基础Config类并覆盖特定值
     到玩具形状数据集。
    """
    # Give the configuration a recognizable name
    # 给配置一个可识别的名字
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    # 训练1个GPU和每个GPU 8个图像。 我们可以在每个上放置多个图像
    # GPU，因为图像很小。 批量大小为8(GPUs * images/GPU)。
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    # 类别数量（包括背景）
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # 使用小图像进行更快速的训练。 设置小方面的限制
    # the large side, and that determines the image shape.
    # 大的一面，这决定了图像的形状。
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    # 使用较小的锚点是因为我们的图像和对象很小
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # 由于图像很小，因此减少了每个图像的训练ROI
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # 几件物品。 旨在允许ROI采样选择33％的正面投资回报率。
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    # 因为数据很简单，所以使用一个epoch 
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # 由于epoch 很小，因此使用小的验证步骤
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()
```
# 笔记本首选项

```python
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    返回要用于的Matplotlib轴数组
     笔记本中的所有可视化。 提供一个
     控制图形大小的中心点。
    
     更改默认大小属性以控制大小
     的渲染图像
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
```
# Dataset
创建一个合成数据集

扩展Dataset类并添加一个方法来加载形状数据集load_shapes（），并覆盖以下方法：

- load_image()
- load_mask()
- image_reference()

```python
class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    生成形状合成数据集。 数据集由简单组成
     随机放置在空白表面上的形状（三角形，正方形，圆形）。
     图像即时生成。 无需文件访问。
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images. 生成所需数量的合成图像。
        count: number of images to generate. 数量
        height, width: the size of the generated images. 尺寸
        """
        # Add classes 添加类别
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        # 生成图像的随机规格（即颜色和形状大小和位置列表）。 这比较紧凑比实际图片。图像在load_image（）中即时生成。
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        创建具有多种形状的图像的随机规格。
         返回图像的背景颜色和形状列表
         可用于绘制图像的规格。
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        # 生成几个随机形状并记录它们边界框
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
```

```python
# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()
```

```python
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
```

![这里写图片描述](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAywAAACnCAYAAADzEdgbAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAADBtJREFUeJzt3X/M9XVdx/HXG4lFuCaQmRXLrJyyGlQDFE2gVILZ7Ieu%0AmrkyN2gJNHTWbM0WWow5cRO1qYCrVqvWUmKBBoKJ/LrR2AqXRawmSwcBMUxvIOjdH+d74HB539fN%0AfcN1nc91zuOxMc75nnN9z+fc9/c+13mez+ecU90dAACAER207AEAAADsjWABAACGJVgAAIBhCRYA%0AAGBYggUAABiWYAEAAIa1FsFSVd9TVVdt2Hb7Aezniqo6Zjp9elXdu3DZBVX1+iexj/Oq6j+q6u8W%0Ath1bVZ+tqk9X1dVV9bxp+/dW1d9X1bVV9amq+s5N9vv8qvpcVT1QVScubH9vVd1YVTdU1W8tbH97%0AVe2qqpuq6tz9/bMAeDKq6jlV9e79uP61mz3WAbB+1iJYJhu/cOZAvoDmuiQvnU6fmOTzVXX0dP6l%0A0+X78oEkJ2/Y9uUkp3b3yUnek+S8afuvJ7m4u09J8sdJzt5kv19O8ookf7Vh+/u7+yXT+F4zRdAz%0Ak7yxu4+ftv9aVR36JMbOmqmqdXqMYAt0913d/baN2zc5tnw5GABPsE5PRmqfV6j6YFX9Us18oqqO%0A23CV65O8bDp9TJI/TPKyqjokyXO6+0v7uo3uvisbfiF3993d/bXp7ENJ/nc6/YUkh0+nj0hyd1Ud%0AUlXXVdULplcub66qb+3uB7v7/o33s7vvmP7fSR5J8miS3Un+s6oOS/ItSb6+cJvsIFV19DR79qmq%0A+tuqetF0TFxeVX9RVe+Yrnf7ws98pKpePp3+RFVdM820nTBt+92q+mhVfTzJ66rq5dPs3zVV9cGl%0A3FF2lKo6f+G4PGM+w72HY+vkaXb5mqp6z/zHF/bzB9OMy/VVdfoy7gsAy3fwsgewjX60qq6ZTu8t%0AXs5Nck1mUXJ1d9+y4fKbk1xSVQcn+b8kn0lyYWZhsStJqurFSc7PN75KeF53f3qzAU4B8a4kvzpt%0AujrJJ6vqTUkOSXJ8dz9cVW9M8kdJ7k9yTnc/sNl+p32/Pskd86iqqiuT/Etmfxbv6u5H9rUPhnRq%0Akku7++KqqiQfS3J2d++qqg8vXG9vr1r/THfvrqoXZjb79xPT9ge7+6eTpKr+IclJ3f3Vqrqwqk7v%0A7iu26P6ww1XVaUmO6u4Tp/PPT/LahassHlv/nOTHuvue6fhd3M+pSZ7V3adMM8A3JnHcAayhdQqW%0Az3X3q+ZnqupfN16hux+qqo8muSDJc/dy+d1JfjbJrd19b1V9R2aB89npOjclOWV/BzdF0J8nOb+7%0AvzhtviDJb3f3ZVX185mF0Fnd/W9V9e9JDu/um5/Evl+R5JeTvHo6/wNJfi7J85I8I8lnqurj3f2V%0A/R03S3dpkt+pqj9J8k9Jvj/JPLRvTvJd0+nFJ4OVJFX1zUneV1UvyCzAF983cMN0nW/L7Di5bHpC%0AeViSLwb27geTXLtw/tENl8+PrWcnuae770kemwVOHo/rH0py8vRCUyX5pqo6orvv27KRs3aq6s2Z%0ABfXt3X3GssfDenIc7ts6Lwn7hlmWqnpukjdlNstx/l72c32S35z+n8zeO/K6TO9fqaoXT0sYFv+7%0ApqpO3nDbi8seKsmfJvlYd1++4fbmb+y/J9PysKp6ZWaxeU9V/dRm93Va5nNektd298MLlz/Q3Y90%0A90NJHkzyzL3cX8b2cHe/rbvfkOSVSe5KMl/KuLik8f5pCeEzkhw7bfvJJI9090mZvV9q8d/Eo0ky%0APZm8I8mru/uU6X1Pl2zd3WEF3JbkpIXzG3/PzI+t/0pyxBTF88fB5PHj8AtJPtndPz69j+8YscLT%0Arbs/MD22eZLI0jgO922dZlg2fdP99Mvy0syWWN1SVX9WVad195Ubfu66zJaO3TSdvz7Ja7r7tmTf%0AMyxTRf9CkhfW7JPCzkzyI0lOS/LsqnpDkn/s7t9I8vtJPlRVj2T2d3Xm9KrkO5O8KrNXxa+qqs8n%0A+WqSv07yoiRHV9UV3f17SS6e7utlVdVJ3trdt1bVLVV14zSsa7t7vz81jSH8YlX9SmZ/x1/JLLYv%0Aqap7MovcuXcnuSqzJ5N3TdtuTPL26Ti8YZPbeEuSy6d/I49mdvzf9nTeCVZHd185vTflhszeH/eX%0Am1z9zUn+pqoeTHJrkrdmemye9vOSqrp22nZnZjPFAKyZenwWHlgl0/uWvq+7z9vnlQEABrVOS8IA%0AAIAdxgwLAAAwLDMsAADAsAQLAAAwrKV9Sti5h1xoLdowOoce9eE9XlKV7G3V4O47zzzgW3zvw2/Z%0A25d3bqtDf/gsx+Ea233r+5d+HDoGx3Lfrov2uL2qsqcl1N2dI08454Bvb4RjMHEcrjvHISPY7Dhc%0Ap4815gmeGCm1yUPV3i479KgPPXb6qcQLwDItRkpt8mC4p8uq6rGff6rxAsCeCZa183iobBYpT8bi%0Az8/jRbgAO8U8NDaLlCdj/vPzeBEuAE8vwbI2nr5Q2ZP5PoULMLqnK1T2pKqEC8DTzJvu18IsVqq2%0AJlYWzW9jcbkYwCju23XRY1GxlaoqBx10UO69+X1bejsA60CwrLzHY2U7iRZgNPNY2U6iBeCpEywr%0AbTmxMidagFEsI1bmRAvAUyNYVtZyY2VOtADLtsxYmRMtAAdOsKykMWJlTrQAyzJCrMyJFoADI1hW%0AzlixMidagO02UqzMiRaA/SdYVsqYsTInWoDtMmKszIkWgP0jWFbMoL+fAbbVqLEyN/r4AEYiWFbG%0A418MOTKzLMBWm38x5MjmXy4JwL4JlhWyU16w2ynjBHamnTJ7sVPGCbBsgmUl7IzZFYCtZtYCYPUI%0AlhXhhToAsxYAq0iwAAAAwxIsO57lYACJ5WAAq0qwrAArIAAsBwNYVYIFAAAYlmABAACGJVgAAIBh%0ACRYAAGBYgmVH8wlhAIlPCANYZYJlR6vsvvOMZQ8CYOmOPOGcZQ8BgC0iWADY8bp72UMAYIsIFgAA%0AYFiCBQAAGJZgAQAAhiVYAACAYQmWHa/y9S/5pDCAI44/e9lDAGALCBYAVoJPCgNYTYIFAAAYlmBZ%0ACZaFASSWhQGsIsECwMqwLAxg9QiWlbEzZlm6syPGCexcO2GWpbtz+HFnLXsYADuCYGEJatkDAFaY%0AWRaA1SJYVsrYsyzdye47xx0fsDpGnmXp7qHHBzCag5c9gHXQSb7239+9bbf3P3nntt3WPnXn2499%0Ax0KsmF1Zpvt2XbTsISxFd+fIE85Z9jDYRqMtuaqq3LfrIrECcAAEy3ap9X2iLlbGUWt8HMIydbdY%0AAThAloSx5cQKwNjL1ABGJljYBmIFAIADI1gAAIBhCRYAAGBYggUAABiWYAEAAIYlWAAAgGEJFgAA%0AYFiCBQAAGJZgAQAAhiVYAACAYQkWAABgWIIFAAAYlmABAACGJVgAAIBhCRYAAGBYggUAABiWYAEA%0AAIYlWAAAgGEJFgAAYFiCBQAAGJZgAQAAhiVYAACAYQkWAABgWIIFAAAYlmABAACGJVgAAIBhCRYA%0AAGBYggUAABiWYAEAAIYlWAAAgGEJFgAAYFiCBQAAGJZgAQAAhiVYAACAYQkWAABgWIIFAAAYlmAB%0AAACGJVgAAIBhCRYAAGBYggUAABiWYAEAAIYlWAAAgGEJFgAAYFiCBQAAGJZgAQAAhiVYAACAYQkW%0AAABgWIIFAAAYlmABAACGJVgAAIBhCRYAAGBYggUAABiWYAEAAIYlWAAAgGEJFgAAYFiCBQAAGJZg%0AAQAAhiVYAACAYQkWAABgWIIFAAAY1sHLHsA6qCSHPevOZQ8DcvhxZy17CAAA+0WwbJNa9gAAAGAH%0AsiQMAAAYlmABAACGJVgAAIBhCRYAAGBYggUAABiWYAEAAIYlWAAAgGEJFgAAYFiCBQAAGJZgAQAA%0AhiVYAACAYQkWAABgWIIFAAAYlmABAACGJVgAAIBhCRYAAGBYggUAABiWYAEAAIYlWAAAgGEJFgAA%0AYFiCBQAAGJZgAQAAhiVYAACAYQkWAABgWNXdyx4DAADAHplhAQAAhiVYAACAYQkWAABgWIIFAAAY%0AlmABAACGJVgAAIBhCRYAAGBYggUAABiWYAEAAIYlWAAAgGEJFgAAYFiCBQAAGJZgAQAAhiVYAACA%0AYQkWAABgWIIFAAAYlmABAACGJVgAAIBhCRYAAGBYggUAABjW/wM61eeL24P+VAAAAABJRU5ErkJg%0Agg==%0A)

# Ceate Model

```python
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
```

```python
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
```
# 训练
分两个阶段训练：

- 只有头部。 在这里，我们冻结所有的骨干层，只训练随机初始化层（即我们没有使用MS COCO预先训练的权重）。 为了仅训练头层，将layers='heads'传递给train() 函数。
- 微调所有图层。 对于这个简单的例子，这不是必要的，但我们将其包括在内以显示过程。 只需传递layers="all"来训练所有图层。

```python
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
'''
训练头部分支
传递layers="heads"会冻结除头部图层之外的所有图层
你也可以传递一个正则表达式来选择哪些图层按名称模式进行训练。
'''
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
```

```python
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
'''
微调所有图层
传递layers="all"训练所有图层。 你也可以
传递正则表达式来选择要使用的图层按名称模式训练。
'''
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")
```

```python
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
'''
保存重量
＃通常不需要，因为回调在每个epoch后都会保存
＃取消注释以手动保存
'''
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)
```
# Detection

```python
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
```

```python
# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))
```

```python
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())
```
# Evaluation

```python
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
```

# 完整代码

```python
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

# %matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    玩具形状数据集的训练配置。
     派生自基础Config类并覆盖特定值
     到玩具形状数据集。
    """
    # Give the configuration a recognizable name
    # 给配置一个可识别的名字
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    # 训练1个GPU和每个GPU 8个图像。 我们可以在每个上放置多个图像
    # GPU，因为图像很小。 批量大小为8(GPUs * images/GPU)。
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    # 类别数量（包括背景）
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # 使用小图像进行更快速的训练。 设置小方面的限制
    # the large side, and that determines the image shape.
    # 大的一面，这决定了图像的形状。
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    # 使用较小的锚点是因为我们的图像和对象很小
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # 由于图像很小，因此减少了每个图像的训练ROI
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # 几件物品。 旨在允许ROI采样选择33％的正面投资回报率。
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    # 因为数据很简单，所以使用一个epoch
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # 由于epoch 很小，因此使用小的验证步骤
    VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    返回要用于的Matplotlib轴数组
     笔记本中的所有可视化。 提供一个
     控制图形大小的中心点。

     更改默认大小属性以控制大小
     的渲染图像
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    生成形状合成数据集。 数据集由简单组成
     随机放置在空白表面上的形状（三角形，正方形，圆形）。
     图像即时生成。 无需文件访问。
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images. 生成所需数量的合成图像。
        count: number of images to generate. 数量
        height, width: the size of the generated images. 尺寸
        """
        # Add classes 添加类别
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        # 生成图像的随机规格（即颜色和形状大小和位置列表）。 这比较紧凑比实际图片。图像在load_image（）中即时生成。
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        创建具有多种形状的图像的随机规格。
         返回图像的背景颜色和形状列表
         可用于绘制图像的规格。
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        # 生成几个随机形状并记录它们边界框
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes


# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')


# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
'''
保存重量
＃通常不需要，因为回调在每个epoch后都会保存
＃取消注释以手动保存
'''
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
```